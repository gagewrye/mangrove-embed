"""
Dense chips for segmentation: 64-band AlphaEarth embeddings + 11-class v3 label raster.

Per region:
  1. Chip centers, drawn at 100 m (cheap): half on mangrove pixels, half on random coastal-zone
     pixels (so chips also cover mangrove-free coast). Greedy de-duplication keeps chips from
     overlapping more than ~50%.
  2. Each chip is pulled with ee.data.computePixels on the embedding's native UTM 10 m grid
     (no resampling): `chip_px` x `chip_px` pixels, bands = A00..A63, class, valid.
  3. Saved to data/chips/<region>/chip_NNNN.npz (x: float16 [64,H,W], y: uint8 [H,W]) with
     index.csv. Cached per chip; rerun to resume.

Labels in y: 1..11 = v3 classes, 255 = ignore (ambiguous, outside the coastal zone, or no
embedding). Training maps 1..11 -> 0..10.
"""
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm

from . import BANDS
from .config import Config
from .extract import CANDIDATE_SCALE, _with_retry, build_images

log = logging.getLogger(__name__)
IGNORE = 255


# ------------------------------------------------------------------ pure helpers (testable)
def dedupe(xy, min_dist):
    """Greedy: keep points at least min_dist apart (input order = priority)."""
    kept = []
    for p in xy:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_dist ** 2 for q in kept):
            kept.append(p)
    return np.array(kept).reshape(-1, 2)


def snap_origin(cx, cy, grid_x0, grid_y0, px, res=10.0):
    """Upper-left corner of a px-wide chip centered near (cx, cy), aligned to the native grid."""
    half = px * res / 2
    x0 = grid_x0 + round((cx - half - grid_x0) / res) * res
    y1 = grid_y0 + round((cy + half - grid_y0) / res) * res
    return x0, y1


def to_arrays(structured, n_bands=64):
    """computePixels structured array -> (x float16 [64,H,W], y uint8 [H,W])."""
    x = np.stack([structured[b] for b in BANDS[:n_bands]]).astype(np.float32)
    valid = structured["valid"].astype(bool)
    cls = structured["class"].astype(np.int32)
    y = np.where(valid & (cls >= 1) & (cls <= 11), cls, IGNORE).astype(np.uint8)
    x[:, ~valid] = 0.0
    return x.astype(np.float16), y


# ------------------------------------------------------------------ Earth Engine
def chip_centers(name, bbox, cfg: Config, labels, coastal, proj):
    region = ee.Geometry.Rectangle(bbox)
    def pts(img, n, seed):
        fc = img.selfMask().rename("v").sample(region=region, scale=CANDIDATE_SCALE,
                                                projection=proj, numPixels=int(n), seed=seed,
                                                tileScale=4, geometries=True)
        feats = _with_retry(lambda: fc.getInfo())["features"]
        return [f["geometry"]["coordinates"] for f in feats]
    # numPixels is approximate (pre-mask); ask for extra
    mang = pts(labels.eq(1), cfg.chip_n_mangrove * 8, cfg.seed)
    rand = pts(coastal, cfg.chip_n_random * 8, cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(mang); rng.shuffle(rand)
    return mang, rand


def fetch_chip(img, crs, x0, y1, px):
    req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": px, "height": px},
                    "affineTransform": {"scaleX": 10, "shearX": 0, "translateX": x0,
                                        "shearY": 0, "scaleY": -10, "translateY": y1},
                    "crsCode": crs}}
    return _with_retry(lambda: ee.data.computePixels(req))


def extract_chips_region(name, bbox, cfg: Config):
    out = Path(cfg.data_dir) / "chips" / name
    index_path = out / "index.csv"
    out.mkdir(parents=True, exist_ok=True)
    proj, feat_img, labels, coastal = build_images(bbox, cfg)
    pinfo = _with_retry(lambda: proj.getInfo())
    crs = pinfo["crs"]
    t = pinfo.get("transform", [10, 0, 0, 0, -10, 0])
    grid_x0, grid_y0 = t[2], t[5]

    if index_path.exists():
        index = pd.read_csv(index_path)
    else:
        mang, rand = chip_centers(name, bbox, cfg, labels, coastal, proj)
        fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        def utm(lonlat):
            return np.array([fwd.transform(lo, la) for lo, la in lonlat]).reshape(-1, 2)
        min_d = cfg.chip_px * 10 * 0.5
        m_xy = dedupe(utm(mang), min_d)[: cfg.chip_n_mangrove]
        r_xy = dedupe(np.vstack([m_xy, utm(rand)]), min_d)[len(m_xy):][: cfg.chip_n_random]
        rows = []
        for kind, xy in (("mangrove", m_xy), ("random", r_xy)):
            for cx, cy in xy:
                x0, y1 = snap_origin(cx, cy, grid_x0, grid_y0, cfg.chip_px)
                rows.append(dict(kind=kind, x0=x0, y1=y1))
        index = pd.DataFrame(rows)
        inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        half = cfg.chip_px * 5
        index["lon"], index["lat"] = zip(*[inv.transform(x + half, y - half)
                                          for x, y in zip(index.x0, index.y1)])
        index.insert(0, "chip", [f"chip_{i:04d}" for i in range(len(index))])
        index["region"], index["crs"] = name, crs
        index.to_csv(index_path, index=False)
    log.info("%s: %d chips (%s), %s, %d px", name, len(index),
             index.kind.value_counts().to_dict(), crs, cfg.chip_px)

    emb = feat_img.select([f"s0_{b}" for b in BANDS], BANDS)
    img = ee.Image.cat([emb.toFloat(),
                        labels.unmask(IGNORE).rename("class").toFloat(),
                        emb.select(0).mask().rename("valid").toFloat()])

    def job(r):
        path = out / f"{r.chip}.npz"
        if path.exists():
            return
        x, y = to_arrays(fetch_chip(img, crs, r.x0, r.y1, cfg.chip_px))
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, x=x, y=y)
        tmp.rename(path)

    failed = []
    with ThreadPoolExecutor(cfg.workers) as ex:
        futs = {ex.submit(job, r): r.chip for r in index.itertuples()}
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{name} chips"):
            try:
                f.result()
            except Exception as e:
                failed.append((futs[f], str(e)[:200]))
    for c, e in failed:
        log.error("%s/%s failed: %s", name, c, e)
    if failed:
        raise RuntimeError(f"{name}: {len(failed)} chip(s) failed; rerun to retry.")


def extract_chips(cfg: Config, regions=None):
    if not cfg.project:
        raise SystemExit("No Earth Engine project. Set `project:` in the config, EE_PROJECT, or --project.")
    ee.Initialize(project=cfg.project)
    failed = []
    for name in regions or list(cfg.regions):
        try:
            extract_chips_region(name, cfg.regions[name], cfg)
        except Exception as e:
            log.error("%s: %s", name, e)
            failed.append(name)
    if failed:
        raise SystemExit(f"Incomplete regions: {failed}. Rerun `mangrove chips` to resume.")
