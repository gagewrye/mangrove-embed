"""
Pull AlphaEarth embeddings + v3 11-class labels straight to local parquet via the Earth Engine
Python API. No Code Editor, no Drive exports, no assets.

Labels are a port of mangrove_training_knn.js (v3):
  1 mangrove (GMW v3 extent/core) · 2 salt_marsh (GWL 186 AND WorldCover 90) ·
  3 tidal_flat (GWL 187 AND WC 60/80/90) · 4 water (JRC >= 10 mo) · 5 seasonal_water (1-9 mo) ·
  6 built_up (WC 50 or GHSL >= 50%) · 7 barren · 8 coastal_forest · 9 cropland ·
  10 herbaceous_wetland · 11 other_land
  Ambiguous pixels (another dataset says mangrove, label doesn't) are masked.
  Mangroves anywhere in the region; other classes only within coast_buffer_m of mangrove.
Sampling is two-stage, as in v3: stratified candidates are drawn at 100 m (cheap: 100x fewer
pixels to scan), then labels are re-checked and embeddings pulled at 10 m at those points only.
Candidates whose 10 m label differs from the 100 m one are dropped, so every kept label is an
exact 10 m match.

Per region:
  1. Tile the bbox; one cheap request gets approximate per-tile class counts at 100 m.
  2. Allocate the per-class budget across tiles proportional to class area.
  3. Pull tiles concurrently with computeFeatures. Timeouts/memory errors -> split tile into
     quadrants and retry; rate limits -> backoff.
  4. Cache each tile in data/raw/<region>/; reruns skip finished tiles (resumable).
  5. Merge + subsample to exact budget -> data/processed/<region>_{labeled,unlabeled}.parquet.
"""
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import BANDS, CLASS_IDS
from .config import Config

log = logging.getLogger(__name__)

EMB = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
OVERSAMPLE = 1.3
CANDIDATE_SCALE = 100   # m; stage-1 candidate draw
CANDIDATE_FACTOR = 3    # candidates per needed point (some fail the 10 m re-check)
MAX_SPLIT_DEPTH = 3


# ------------------------------------------------------------------ pure helpers (testable)
def feature_columns(scales):
    return [f"s{s}_{b}" for s in scales for b in BANDS] + ["longitude", "latitude"]


def tile_grid(bbox, deg):
    x0, y0, x1, y1 = bbox
    nx, ny = max(1, math.ceil((x1 - x0) / deg)), max(1, math.ceil((y1 - y0) / deg))
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    return [[x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy]
            for j in range(ny) for i in range(nx)]


def quadrants(t):
    x0, y0, x1, y1 = t
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return [[x0, y0, xm, ym], [xm, y0, x1, ym], [x0, ym, xm, y1], [xm, ym, x1, y1]]


def allocate(counts, budget):
    """Split `budget` across tiles proportional to counts; every nonzero tile gets >= 1."""
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return np.zeros(len(counts), dtype=int)
    alloc = np.ceil(budget * OVERSAMPLE * counts / counts.sum()).astype(int)
    alloc[(counts > 0) & (alloc == 0)] = 1
    return alloc


# ------------------------------------------------------------------ Earth Engine graph
def class_image(cfg: Config, proj):
    """Exclusive 11-class label image + coastal mask (port of v3 JS, section 2)."""
    y = cfg.year
    gmw_extent = (ee.ImageCollection("projects/sat-io/open-datasets/GMW/extent/GMW_V3")
                  .filter(ee.Filter.eq("system:index", f"gmw_v3_{y}")).first()
                  .select(0).unmask(0).gt(0))
    gmw_core = (ee.Image("projects/sat-io/open-datasets/GMW/core/gmw_v3_mng_core")
                .select(0).unmask(0).eq(1))
    mangrove = gmw_core if cfg.mangrove_label == "core" else gmw_extent

    gwl = (ee.ImageCollection("projects/sat-io/open-datasets/GWL_FCS30")
           .filter(ee.Filter.eq("system:index", f"GWL_FCS30_{y}")).first()
           .select(0).unmask(0))
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select(0).unmask(0)
    wc_is = lambda codes: wc.remap(codes, [1] * len(codes), 0).eq(1)

    salt_marsh = gwl.eq(186).And(wc.eq(90))
    tidal_flat = gwl.eq(187).And(wc_is([60, 80, 90]))
    seas = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality").unmask(0)
    water = seas.gte(10)
    seasonal_water = seas.gte(1).And(seas.lt(10))
    ghsl = (ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S_10m").first()
            .select("built_surface").unmask(0))
    built = wc.eq(50).Or(ghsl.gte(50))

    ambiguous = mangrove.Not().And(wc.eq(95).Or(gwl.eq(185)).Or(gmw_extent).Or(gmw_core))

    grid = ee.Projection("EPSG:4326").atScale(250)
    coastal = (mangrove.selfMask()
               .reduceResolution(ee.Reducer.max(), maxPixels=1024)
               .reproject(grid).unmask(0)
               .focalMax(math.ceil(cfg.coast_buffer_m / 250), "circle", "pixels")
               .reproject(grid).gt(0))

    cls = (ee.Image(0)
           .where(wc_is([20, 30, 70, 100]), 11)
           .where(wc.eq(90), 10)
           .where(wc.eq(40), 9)
           .where(wc.eq(10), 8)
           .where(wc.eq(60), 7)
           .where(built, 6)
           .where(seasonal_water, 5)
           .where(water, 4)
           .where(tidal_flat, 3)
           .where(salt_marsh, 2)
           .where(mangrove, 1)
           .updateMask(ambiguous.Not())
           .rename("class").toInt()
           .setDefaultProjection(proj))
    labels = cls.updateMask(cls.eq(1).Or(coastal)).selfMask()
    return labels, coastal


def build_images(bbox, cfg: Config):
    region = ee.Geometry.Rectangle(bbox)
    col = (ee.ImageCollection(EMB)
           .filterDate(f"{cfg.year}-01-01", f"{cfg.year + 1}-01-01")
           .filterBounds(region))
    # mosaic() drops native projection; restore UTM 10 m so pixel kernels = 10 m pixels
    proj = col.filterBounds(region.centroid(1)).first().select(0).projection()
    emb = col.mosaic().select(BANDS).setDefaultProjection(proj)

    feats = []
    for s in cfg.scales:
        img = emb if s == 0 else emb.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(s, "pixels"))
        feats.append(img.rename([f"s{s}_{b}" for b in BANDS]))
    feat_img = ee.Image.cat(feats + [ee.Image.pixelLonLat()])
    labels, coastal = class_image(cfg, proj)
    return proj, feat_img, labels, coastal


def tile_class_counts(tiles, labels, proj):
    """Approximate per-tile pixel counts per class at 100 m -> (n_tiles, n_classes)."""
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(t), {"tid": i})
                               for i, t in enumerate(tiles)])
    out = labels.reduceRegions(collection=fc, reducer=ee.Reducer.frequencyHistogram(),
                               crs=proj.crs(), scale=100, tileScale=4)
    rows = _with_retry(lambda: out.getInfo())["features"]
    counts = np.zeros((len(tiles), len(CLASS_IDS)))
    for f in rows:
        p = f["properties"]
        for k, v in (p.get("histogram") or {}).items():
            c = int(float(k))
            if c in CLASS_IDS:
                counts[p["tid"], c - 1] = v
    return counts


# ------------------------------------------------------------------ fetching
class TooBig(Exception):
    pass


def _with_retry(fn, tries=6):
    for k in range(tries):
        try:
            return fn()
        except ee.EEException as e:
            msg = str(e).lower()
            if any(s in msg for s in ("timed out", "memory limit", "too many pixels",
                                      "computation too complex", "aggregations")):
                raise TooBig(str(e))
            if any(s in msg for s in ("too many", "rate", "quota", "429", "503", "unavailable",
                                      "internal error")) and k < tries - 1:
                time.sleep(min(60, 2 ** k + np.random.rand()))
                continue
            raise
    raise RuntimeError("unreachable")


def _pull(fc, cols):
    df = _with_retry(lambda: ee.data.computeFeatures(
        {"expression": fc.select(cols), "fileFormat": "PANDAS_DATAFRAME"}))
    return df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=cols)


def fetch_labeled(tile, points, feat_img, labels, proj, cols, seed, depth=0):
    points = [int(p) for p in points]
    if sum(points) == 0:
        return pd.DataFrame(columns=cols)
    try:
        candidates = labels.stratifiedSample(
            numPoints=0, classBand="class", classValues=CLASS_IDS,
            classPoints=[p * CANDIDATE_FACTOR for p in points],
            region=ee.Geometry.Rectangle(tile), scale=CANDIDATE_SCALE, projection=proj,
            seed=seed, tileScale=4, geometries=True)
        checked = (feat_img.addBands(labels.rename("class10"))
                   .sampleRegions(collection=candidates, properties=["class"], scale=10,
                                  projection=proj, tileScale=16, geometries=False)
                   .filter(ee.Filter.equals(leftField="class", rightField="class10")))
        return _pull(checked, cols)
    except TooBig:
        if depth >= MAX_SPLIT_DEPTH:
            raise
        sub = [math.ceil(p / 4) for p in points]
        return pd.concat([fetch_labeled(q, sub, feat_img, labels, proj, cols, seed + 7 * (i + 1),
                                        depth + 1) for i, q in enumerate(quadrants(tile))],
                         ignore_index=True)


def fetch_unlabeled(tile, n, feat_img, coastal, proj, cols, seed, depth=0):
    if n == 0:
        return pd.DataFrame(columns=cols)
    try:
        points = coastal.selfMask().rename("coastal").sample(
            region=ee.Geometry.Rectangle(tile), scale=CANDIDATE_SCALE, projection=proj,
            numPixels=int(n), seed=seed, tileScale=4, geometries=True)
        fc = feat_img.sampleRegions(collection=points, scale=10, projection=proj,
                                    tileScale=16, geometries=False)
        return _pull(fc, cols)
    except TooBig:
        if depth >= MAX_SPLIT_DEPTH:
            raise
        return pd.concat([fetch_unlabeled(q, math.ceil(n / 4), feat_img, coastal, proj, cols,
                                          seed + 7 * (i + 1), depth + 1)
                          for i, q in enumerate(quadrants(tile))], ignore_index=True)


# ------------------------------------------------------------------ orchestration
def _cached(path: Path, fn):
    if path.exists():
        return pd.read_parquet(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(path)
    return df


def extract_region(name, bbox, cfg: Config):
    out_lab = cfg.processed_dir / f"{name}_labeled.parquet"
    out_unl = cfg.processed_dir / f"{name}_unlabeled.parquet"
    if out_lab.exists() and out_unl.exists():
        log.info("%s: already processed, skipping", name)
        return

    proj, feat_img, labels, coastal = build_images(bbox, cfg)
    tiles = tile_grid(bbox, cfg.tile_deg)
    raw = cfg.raw_dir / name
    raw.mkdir(parents=True, exist_ok=True)

    cnt = _cached(raw / "tile_counts.parquet", lambda: pd.DataFrame(
        tile_class_counts(tiles, labels, proj), columns=[str(c) for c in CLASS_IDS])).values
    alloc = np.stack([allocate(cnt[:, j], cfg.n_per_class) for j in range(cnt.shape[1])], 1)
    a_unl = allocate(cnt.sum(1), cfg.n_unlabeled * 2)
    log.info("%s: %d tiles (%d with mangrove); approx px per class at 100 m: %s",
             name, len(tiles), int((cnt[:, 0] > 0).sum()),
             dict(zip(CLASS_IDS, cnt.sum(0).astype(int).tolist())))

    lab_cols = feature_columns(cfg.scales) + ["class"]
    unl_cols = feature_columns(cfg.scales)
    jobs = []
    for i, t in enumerate(tiles):
        s = cfg.seed + 1000 * i
        if alloc[i].sum() > 0:
            jobs.append((raw / f"lab_{i:04d}.parquet",
                         lambda t=t, i=i, s=s: fetch_labeled(t, alloc[i], feat_img, labels, proj,
                                                             lab_cols, s)))
        if a_unl[i] > 0:
            jobs.append((raw / f"unl_{i:04d}.parquet",
                         lambda t=t, i=i, s=s: fetch_unlabeled(t, a_unl[i], feat_img, coastal,
                                                               proj, unl_cols, s + 1)))

    failed = []
    with ThreadPoolExecutor(cfg.workers) as ex:
        futs = {ex.submit(_cached, p, fn): p for p, fn in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc=name):
            try:
                f.result()
            except Exception as e:  # keep going; report at end
                failed.append((futs[f].name, str(e)[:200]))
    if failed:
        for n, e in failed:
            log.error("%s/%s failed: %s", name, n, e)
        raise RuntimeError(f"{name}: {len(failed)} tile(s) failed; rerun to retry (cache kept).")

    lab = pd.concat([pd.read_parquet(p) for p in sorted(raw.glob("lab_*.parquet"))],
                    ignore_index=True).dropna()
    lab["class"] = lab["class"].astype(int)
    lab = pd.concat([g.sample(min(len(g), cfg.n_per_class), random_state=cfg.seed)
                     for _, g in lab.groupby("class")], ignore_index=True)
    unl = pd.concat([pd.read_parquet(p) for p in sorted(raw.glob("unl_*.parquet"))],
                    ignore_index=True).dropna()
    unl = unl.sample(min(len(unl), cfg.n_unlabeled), random_state=cfg.seed)
    for df in (lab, unl):
        df["region"] = name
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(out_lab, index=False)
    unl.to_parquet(out_unl, index=False)
    log.info("%s: labeled per class %s | unlabeled %d", name,
             lab["class"].value_counts().sort_index().to_dict(), len(unl))


def extract(cfg: Config, regions=None):
    if not cfg.project:
        raise SystemExit("No Earth Engine project. Set `project:` in the config, EE_PROJECT, or --project.")
    ee.Initialize(project=cfg.project)
    names = regions or list(cfg.regions)
    failed = []
    for name in names:
        try:
            extract_region(name, cfg.regions[name], cfg)
        except Exception as e:
            log.error("%s: %s", name, e)
            failed.append(name)
    if failed:
        raise SystemExit(f"Incomplete regions: {failed}. Rerun `mangrove extract` to resume.")
