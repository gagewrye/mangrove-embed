"""Precision of the confirmed-loss layer, checked against Sentinel-2.

Per region, sample pixels from three groups (change report definitions, all chips):
  confirmed   lost, in a patch >= 0.5 ha
  edge        lost, not in such a patch
  stable      mangrove in 2017, 2018, 2024 and 2025 (the control)
and take cloud-masked (Cloud Score+) Sentinel-2 median NDVI / NDMI over 2017-01..2018-12 and
2024-01..2025-12 (full years, so seasons balance). A lost pixel counts as "verified" when its NDVI
change is below the 5th percentile of the region's stable mangrove (~5% false alarms on stable).
Pixels with early NDVI < 0.3 are flagged: no real canopy in 2017-18, so "loss" was misclassification.
Mangrove converted to other vegetation may keep its NDVI; results are broken down by 2025 class.

usage: python scripts/loss_precision.py [--n 300]
Writes results/change_report/loss_precision.csv and loss_precision_pixels.csv
"""
import argparse
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from storm_check import s2_index  # noqa: E402

from mangrove_embed import config  # noqa: E402
from mangrove_embed.report import NICE, NOW, load  # noqa: E402
from mangrove_embed.seg.train import load_index  # noqa: E402

WINDOWS = {"early": ("2017-01-01", "2019-01-01"), "late": ("2024-01-01", "2026-01-01")}


def sample(region, g, d, n, rng):
    groups = {"confirmed": d["loss_conf"], "edge": d["loss"] & ~d["loss_conf"], "stable": d["before"] & d["after"]}
    out = []
    for name, mask in groups.items():
        i, r, c = np.nonzero(mask)
        if not len(i):
            continue
        k = rng.choice(len(i), min(n, len(i)), replace=False)
        out.append(pd.DataFrame(dict(group=name, chip=i[k], row=r[k], col=c[k],
                                     now=[NOW.get(int(v), "mangrove") for v in d["cls"][i[k], -1, r[k], c[k]]])))
    p = pd.concat(out, ignore_index=True)
    lon, lat = np.zeros(len(p)), np.zeros(len(p))
    for crs, q in p.groupby(g.crs.values[p.chip.values]):
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        x0, y1 = g.x0.values[q.chip.values], g.y1.values[q.chip.values]
        lo, la = tr.transform(x0 + (q.col.values + 0.5) * 10, y1 - (q.row.values + 0.5) * 10)
        lon[q.index], lat[q.index] = lo, la
    return p.assign(region=region, lon=lon, lat=lat)


def _chip_fc(r):
    """Composites over one chip's footprint (region-wide bounds exceed EE memory)."""
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"i": int(i)})
                               for i, lo, la in zip(r.index, r.lon, r.lat)])
    geom = fc.geometry().bounds()
    img = ee.Image.cat([s2_index(geom, a, b).select(["ndvi", "ndmi"]).rename([f"{w}_ndvi", f"{w}_ndmi"])
                        for w, (a, b) in WINDOWS.items()])
    return img.reduceRegions(fc, ee.Reducer.first(), scale=10)


def s2_values(p, chips_per_call=6):
    chips = p.chip.unique()
    vals = []
    for s in range(0, len(chips), chips_per_call):
        q = p[p.chip.isin(chips[s:s + chips_per_call])]
        fcs = ee.FeatureCollection([_chip_fc(r) for _, r in q.groupby("chip")]).flatten()
        for t in range(6):
            try:
                got = fcs.getInfo()["features"]
                break
            except ee.EEException as e:
                print(f"  retry {t}: {e}", flush=True)
                time.sleep(30 * (t + 1))
        else:
            raise RuntimeError("Earth Engine failed 6 times")
        vals.append(pd.DataFrame([f["properties"] for f in got]).set_index("i"))
    return p.join(pd.concat(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    ee.Initialize(project=config.load("configs/default.yaml").project)
    df = load_index("data/temporal")
    rng = np.random.default_rng(0)
    pix, rows = [], []
    for region, g in df.groupby("region"):
        g = g.reset_index(drop=True)
        p = s2_values(sample(region, g, load(region), a.n, rng))
        p["dndvi"], p["dndmi"] = p.late_ndvi - p.early_ndvi, p.late_ndmi - p.early_ndmi
        stable = p[p.group == "stable"].dropna(subset=["dndvi"])
        thr = float(stable.dndvi.quantile(0.05))
        p["verified"] = p.dndvi < thr
        p["no_canopy_2017"] = p.early_ndvi < 0.3
        pix.append(p)
        for grp, q in p.dropna(subset=["dndvi"]).groupby("group"):
            rows.append(dict(region=region, group=grp, px=len(q), threshold=thr,
                             early_ndvi=q.early_ndvi.median(), dndvi=q.dndvi.median(),
                             verified_pct=100 * q.verified.mean(),
                             no_canopy_2017_pct=100 * q.no_canopy_2017.mean()))
        print(pd.DataFrame([r for r in rows if r["region"] == region]).round(3).to_string(index=False), flush=True)
        pd.concat(pix).round(5).to_csv("results/change_report/loss_precision_pixels.csv", index=False)
    t = pd.DataFrame(rows)
    t.round(4).to_csv("results/change_report/loss_precision.csv", index=False)
    P = pd.concat(pix).dropna(subset=["dndvi"])
    lost = P[P.group != "stable"]
    print("\nverified % by what the pixel looks like in 2025")
    print(lost.pivot_table(index="now", columns="group", values="verified", aggfunc="mean").mul(100).round(0))
    print(lost.pivot_table(index="now", columns="group", values="verified", aggfunc="size"))
    print("\nsummary (verified % / no-canopy-in-2017 %)")
    s = t.pivot_table(index="region", columns="group", values="verified_pct").round(0)
    s.index = [NICE[r] for r in s.index]
    print(s.to_string())


if __name__ == "__main__":
    main()
