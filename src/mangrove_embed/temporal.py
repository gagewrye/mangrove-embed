"""
Matched multi-year chips for change prediction: the same 10 m pixels in every AlphaEarth year.

Per region:
  1. Footprints = the existing segmentation chips (data/chips/<region>/index.csv, kinds
     mangrove/random) plus `chip_n_change` new chips centered on pixels where GMW extent differs
     between any two label years (kind=change), so gain and loss are not vanishingly rare.
  2. Each footprint is pulled once per year on its UTM 10 m grid. Every embedding year shares
     that grid, so pixel (i, j) is the same ground location in every year (no resampling).
  3. Labels: the 11-class v3 image for each GMW year (2017-2020), with the coastal zone fixed to
     `coastal_year` so the label mask does not move with mangrove extent. Years without GMW get
     y = 255 everywhere (pseudo-labels are made later, offline).
  4. Saved to data/temporal/<region>/chip_NNNN.npz: x float16 [T, 64, H, W], y uint8 [T, H, W],
     years int16 [T]; index.csv as for chips. Cached per chip; rerun to resume.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm

from . import BANDS
from .chips import IGNORE, dedupe, fetch_chip, snap_origin, to_arrays
from .config import Config
from .extract import CANDIDATE_SCALE, EMB, _with_retry, build_images, class_image, gmw_extent_image

log = logging.getLogger(__name__)
YEARS = list(range(2017, 2026))       # AlphaEarth annual embeddings
LABEL_YEARS = [2017, 2018, 2019, 2020]  # GMW v3 years with an embedding


# ------------------------------------------------------------------ pure helpers (testable)
def stack_years(per_year, years):
    """[(x [64,H,W], y [H,W]) per year] -> npz dict with a leading year axis."""
    return dict(x=np.stack([x for x, _ in per_year]), y=np.stack([y for _, y in per_year]),
                years=np.asarray(years, dtype=np.int16))


def add_change_rows(index, centers_utm, n, px, grid_x0, grid_y0, region, crs):
    """Append up to n change chips, each at least half a chip from every existing chip."""
    half = px * 5
    existing = np.c_[index.x0 + half, index.y1 - half]
    xy = dedupe(np.vstack([existing, centers_utm]), px * 10 * 0.5)[len(existing):][:n]
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    start = int(index.chip.str[5:].astype(int).max()) + 1
    rows = []
    for i, (cx, cy) in enumerate(xy):
        x0, y1 = snap_origin(cx, cy, grid_x0, grid_y0, px)
        lon, lat = inv.transform(x0 + half, y1 - half)
        rows.append(dict(chip=f"chip_{start + i:04d}", kind="change", x0=x0, y1=y1,
                         lon=lon, lat=lat, region=region, crs=crs))
    return pd.concat([index, pd.DataFrame(rows, columns=index.columns)], ignore_index=True)


# ------------------------------------------------------------------ Earth Engine
def change_centers(bbox, proj, years, n, seed):
    """Centers (lon, lat) of pixels where GMW extent differs between any two of `years`."""
    stack = ee.Image.cat([gmw_extent_image(y) for y in years])
    changed = stack.reduce(ee.Reducer.max()).neq(stack.reduce(ee.Reducer.min()))
    img = (changed.selfMask().rename("v").toInt().setDefaultProjection(proj)
           .reduceResolution(ee.Reducer.max(), maxPixels=1024)
           .reproject(proj.atScale(CANDIDATE_SCALE)))
    fc = img.stratifiedSample(numPoints=0, classBand="v", classValues=[1], classPoints=[int(n)],
                              region=ee.Geometry.Rectangle(bbox), scale=CANDIDATE_SCALE,
                              projection=proj, seed=seed, tileScale=4, geometries=True)
    feats = _with_retry(lambda: fc.getInfo())["features"]
    return [f["geometry"]["coordinates"] for f in feats]


def year_image(bbox, cfg, proj, year, coastal_year):
    region = ee.Geometry.Rectangle(bbox)
    emb = (ee.ImageCollection(EMB).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
           .filterBounds(region).mosaic().select(BANDS).setDefaultProjection(proj))
    if year in LABEL_YEARS:
        labels, _ = class_image(cfg, proj, year=year, coastal_year=coastal_year)
        cls = labels.unmask(IGNORE)
    else:
        cls = ee.Image.constant(IGNORE)
    return ee.Image.cat([emb.toFloat(), cls.rename("class").toFloat(),
                         emb.select(0).mask().rename("valid").toFloat()])


def plan_region(name, bbox, cfg, n_change, coastal_year, label_years):
    out = Path(cfg.data_dir) / "temporal" / name
    index_path = out / "index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    src = Path(cfg.data_dir) / "chips" / name / "index.csv"
    if not src.exists():
        raise SystemExit(f"{src} missing: run `mangrove chips` for {name} first.")
    index = pd.read_csv(src)
    if n_change > 0:
        proj, _, _, _ = build_images(bbox, replace(cfg, year=coastal_year, scales=[0]))
        pinfo = _with_retry(lambda: proj.getInfo())
        crs, t = pinfo["crs"], pinfo.get("transform", [10, 0, 0, 0, -10, 0])
        assert crs == index.crs.iloc[0], (crs, index.crs.iloc[0])
        ll = change_centers(bbox, proj, label_years, n_change * 6, cfg.seed + 7)
        fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        utm = np.array([fwd.transform(lo, la) for lo, la in ll]).reshape(-1, 2)
        index = add_change_rows(index, utm, n_change, cfg.chip_px, t[2], t[5], name, crs)
        n_got = (index.kind == "change").sum()
        if n_got < n_change:
            log.warning("%s: only %d/%d change chips (little GMW change)", name, n_got, n_change)
    out.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)
    return index


def extract_temporal_region(name, bbox, cfg: Config, years, n_change, coastal_year):
    label_years = [y for y in LABEL_YEARS if y in years] or LABEL_YEARS
    index = plan_region(name, bbox, cfg, n_change, coastal_year, label_years)
    out = Path(cfg.data_dir) / "temporal" / name
    log.info("%s: %d chips (%s), years %d-%d", name, len(index),
             index.kind.value_counts().to_dict(), min(years), max(years))
    region = ee.Geometry.Rectangle(bbox)
    proj = (ee.ImageCollection(EMB).filterDate(f"{coastal_year}-01-01", f"{coastal_year + 1}-01-01")
            .filterBounds(region.centroid(1)).first().select(0).projection())
    imgs = {y: year_image(bbox, cfg, proj, y, coastal_year) for y in years}

    def job(r):
        path = out / f"{r.chip}.npz"
        if path.exists():
            return
        per_year = [to_arrays(fetch_chip(imgs[y], r.crs, r.x0, r.y1, cfg.chip_px)) for y in years]
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **stack_years(per_year, years))
        tmp.rename(path)

    failed = []
    with ThreadPoolExecutor(cfg.workers) as ex:
        futs = {ex.submit(job, r): r.chip for r in index.itertuples()}
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{name} temporal"):
            try:
                f.result()
            except Exception as e:
                failed.append((futs[f], str(e)[:200]))
    for c, e in failed:
        log.error("%s/%s failed: %s", name, c, e)
    if failed:
        raise RuntimeError(f"{name}: {len(failed)} chip(s) failed; rerun to retry.")


def extract_temporal(cfg: Config, regions=None, years=None, n_change=100, coastal_year=2020):
    if not cfg.project:
        raise SystemExit("No Earth Engine project. Set `project:` in the config, EE_PROJECT, or --project.")
    ee.Initialize(project=cfg.project)
    years = sorted(years or YEARS)
    failed = []
    for name in regions or list(cfg.regions):
        try:
            extract_temporal_region(name, cfg.regions[name], cfg, years, n_change, coastal_year)
        except Exception as e:
            log.error("%s: %s", name, e)
            failed.append(name)
    if failed:
        raise SystemExit(f"Incomplete regions: {failed}. Rerun `mangrove temporal-chips` to resume.")
