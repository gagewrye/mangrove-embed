"""Is low classifier confidence physical vulnerability, or just confusability with neighbours?

Independent outcome: Sentinel-2 canopy damage (NDVI / NDMI change after the storm minus the same
calendar windows one year earlier; see storm_check.py), on mangrove pixels sampled evenly across
pre-storm confidence bins. If low-confidence pixels lose more canopy, confidence tracks
vulnerability; if damage is flat across bins, confidence only tracks how close a pixel sits to the
mangrove / not-mangrove boundary.

usage: python scripts/storm_damage_by_confidence.py [--per-bin 400]
Writes results/change_report/storm_damage_by_confidence.csv
"""
import argparse
import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from storm_check import window_values  # noqa: E402

from mangrove_embed import config  # noqa: E402
from mangrove_embed.report import M, _jm_zone, load  # noqa: E402
from mangrove_embed.seg.train import load_index  # noqa: E402

YEARS = list(range(2017, 2026))
BINS = [0, 0.8, 0.95, 0.99, 0.999, 1.0001]
LABELS = ["<0.80", "0.80-0.95", "0.95-0.99", "0.99-0.999", ">0.999"]
CASES = {
    "Melissa 2025 (Jamaica SW)": ("jamaica", "SW", "2025-10-28", [2023, 2024], [2025]),
    "Beryl 2024 (Jamaica S-central)": ("jamaica", "S-central", "2024-07-03", [2022, 2023], [2024, 2025]),
    "Remal 2024 (Sundarbans)": ("sundarbans", None, "2024-05-26", [2022, 2023], [2024, 2025]),
    "control: Jamaica east (Melissa date)": ("jamaica", "east", "2025-10-28", [2023, 2024], [2025]),
}


def sample(region, zone, pre, post, per_bin, seed=0):
    df = load_index("data/temporal")
    g = df[df.region == region].reset_index(drop=True)
    d = load(region)
    pm = np.load(f"data/temporal/_report/{region}_pm.npz")["pm"]
    jp, jq = [YEARS.index(y) for y in pre], [YEARS.index(y) for y in post]
    rows = []
    for i, r in g.iterrows():
        if zone and not _jm_zone(r.lat, r.lon).startswith(zone):
            continue
        c = d["cls"][i]
        was = np.all(c[jp] == M, 0) & d["ok"][i]
        rr, cc = np.nonzero(was)
        if not len(rr):
            continue
        conf = pm[i][jp][:, rr, cc].astype(np.float32).mean(0)
        lost = np.all(c[jq][:, rr, cc] != M, 0)
        rows.append(pd.DataFrame(dict(row=rr, col=cc, conf=conf, lost=lost, crs=r.crs, x0=r.x0, y1=r.y1)))
    p = pd.concat(rows, ignore_index=True)
    p["bin"] = pd.cut(p.conf, BINS, labels=LABELS, right=False)
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(p.bin == b), min(per_bin, int((p.bin == b).sum())), replace=False)
            for b in LABELS if (p.bin == b).any()]
    p = p.iloc[np.concatenate(keep)]
    out = []
    for crs, q in p.groupby("crs"):
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(q.x0 + (q.col + 0.5) * 10, q.y1 - (q.row + 0.5) * 10)
        out.append(q.assign(lon=lon, lat=lat))
    return pd.concat(out).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=400)
    a = ap.parse_args()
    ee.Initialize(project=config.load("configs/default.yaml").project)
    rows, pixels = [], []
    for name, (region, zone, landfall, pre, post) in CASES.items():
        pts = sample(region, zone, pre, post, a.per_bin)
        now, base = window_values(pts, landfall), window_values(pts, landfall, shift_years=1)
        pts = pts.join(now.add_prefix("now_")).join(base.add_prefix("base_"))
        pts["dndvi"] = (pts.now_post_ndvi - pts.now_pre_ndvi) - (pts.base_post_ndvi - pts.base_pre_ndvi)
        pts["dndmi"] = (pts.now_post_ndmi - pts.now_pre_ndmi) - (pts.base_post_ndmi - pts.base_pre_ndmi)
        pts["pre_ndvi"] = pts.now_pre_ndvi
        pixels.append(pts.assign(case=name)[["case", "lon", "lat", "conf", "bin", "lost", "pre_ndvi", "dndvi", "dndmi"]])
        pd.concat(pixels).round(5).to_csv("results/change_report/storm_damage_pixels.csv", index=False)
        for b, q in pts.groupby("bin", observed=True):
            q = q.dropna(subset=["dndvi"])
            rows.append(dict(case=name, conf_bin=b, px=len(q), pre_ndvi=q.pre_ndvi.median(),
                             dndvi_median=q.dndvi.median(), damaged_pct=100 * (q.dndvi < -0.1).mean(),
                             dndmi_median=q.dndmi.median(), detector_lost_pct=100 * q.lost.mean()))
        print(pd.DataFrame([r for r in rows if r["case"] == name]).round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv("results/change_report/storm_damage_by_confidence.csv", index=False)


if __name__ == "__main__":
    main()
