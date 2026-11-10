"""Did a storm cause lasting mangrove damage? An independent check of the annual detector with
Sentinel-2 at sub-annual resolution.

For each storm: sample pixels our classifier mapped as mangrove in the years before it, then take
cloud-masked (Cloud Score+) Sentinel-2 median NDVI / NDMI in three windows around the landfall date:
  pre     90 days before
  post    5-120 days after         (immediate canopy damage)
  later   300-420 days after       (lasting damage or recovery)
The same calendar windows one year earlier give a seasonal baseline; storm effect =
(change in storm year) - (change in baseline year). A pixel counts as damaged when its NDVI drop
exceeds 0.1 beyond its own baseline-year change. Reported alongside: the share of the same pixels
our annual detector maps as non-mangrove after the storm.

usage: python scripts/storm_check.py [--n 1500] [--storms irma beryl ...]
Writes results/change_report/storm_check.csv. Needs Earth Engine (`mangrove auth`) and the
report cache (`mangrove change-report`).
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

from mangrove_embed import config
from mangrove_embed.report import M, _jm_zone, load
from mangrove_embed.seg.train import load_index

YEARS = list(range(2017, 2026))
# name: region, zone filter, landfall date, years mapped mangrove before, years checked after (detector)
STORMS = {
    "irma":    ("everglades", None, "2017-09-10", [2017], [2018, 2019]),
    "beryl":   ("jamaica", "south", "2024-07-03", [2022, 2023], [2024, 2025]),
    "melissa": ("jamaica", "sw", "2025-10-28", [2023, 2024], [2025]),
    "amphan":  ("sundarbans", None, "2020-05-20", [2018, 2019], [2020, 2021]),
    "remal":   ("sundarbans", None, "2024-05-26", [2022, 2023], [2024, 2025]),
    # controls: same method, no major storm on that date
    "control: Jamaica east (Beryl date)": ("jamaica", "east", "2024-07-03", [2022, 2023], [2024, 2025]),
    "control: Baja (no storm)": ("mexico_pacific", None, "2022-07-01", [2020, 2021], [2022, 2023]),
}
WINDOWS = {"pre": (-90, -1), "post": (5, 120), "later": (300, 420)}
TODAY = date(2026, 11, 10)


def zone_ok(z, lat, lon):
    if z is None:
        return True
    name = _jm_zone(lat, lon)
    return {"south": name.startswith(("SW", "S-central")), "sw": name.startswith("SW"),
            "east": name == "east"}[z]


def sample_points(region, zone, before, after, n, seed=0):
    """Random pixels mapped mangrove in every `before` year -> lon, lat, detector says non-mangrove."""
    df = load_index("data/temporal")
    g = df[df.region == region].reset_index(drop=True)
    d = load(region)
    rng = np.random.default_rng(seed)
    jb, ja = [YEARS.index(y) for y in before], [YEARS.index(y) for y in after]
    rows = []
    for i, r in g.iterrows():
        if not zone_ok(zone, r.lat, r.lon):
            continue
        c = d["cls"][i]
        m = np.all(c[jb] == M, 0) & d["ok"][i]
        rr, cc = np.nonzero(m)
        if not len(rr):
            continue
        lost = np.all(c[ja][:, rr, cc] != M, 0)
        rows.append(pd.DataFrame(dict(chip=i, row=rr, col=cc, lost=lost, crs=r.crs, x0=r.x0, y1=r.y1)))
    p = pd.concat(rows, ignore_index=True)
    p = p.iloc[rng.choice(len(p), min(n, len(p)), replace=False)].reset_index(drop=True)
    out = []
    for crs, q in p.groupby("crs"):
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(q.x0 + (q.col + 0.5) * 10, q.y1 - (q.row + 0.5) * 10)
        out.append(q.assign(lon=lon, lat=lat))
    return pd.concat(out).reset_index(drop=True)


def s2_index(geom, start, end):
    """Cloud-masked median NDVI and NDMI, Sentinel-2 (harmonized TOA), + count of clear looks."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(geom).filterDate(start, end)
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(start, end)
           .linkCollection(cs, ["cs_cdf"]))

    def prep(img):
        clear = img.select("cs_cdf").gte(0.6)
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndmi = img.normalizedDifference(["B8", "B11"]).rename("ndmi")
        return ndvi.addBands(ndmi).updateMask(clear).addBands(clear.selfMask().rename("n").toFloat())

    c = col.map(prep)
    return c.select(["ndvi", "ndmi"]).median().addBands(c.select("n").count().rename("n"))


def window_values(pts, landfall, shift_years=0):
    d0 = date.fromisoformat(landfall)
    d0 = d0.replace(year=d0.year - shift_years)
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"i": int(i)})
                               for i, (lo, la) in enumerate(zip(pts.lon, pts.lat))])
    geom = fc.geometry().bounds()
    imgs = []
    for w, (a, b) in WINDOWS.items():
        end = min(d0 + timedelta(days=b), TODAY - timedelta(days=3))
        start = d0 + timedelta(days=a)
        if end <= start + timedelta(days=20):
            continue
        imgs.append(s2_index(geom, start.isoformat(), end.isoformat())
                    .rename([f"{w}_ndvi", f"{w}_ndmi", f"{w}_n"]))
    img = ee.Image.cat(imgs)
    got = img.reduceRegions(fc, ee.Reducer.first(), scale=10).getInfo()["features"]
    v = pd.DataFrame([f["properties"] for f in got]).set_index("i").sort_index()
    return v


def summarize(name, pts, now, base):
    row = dict(storm=name, pixels=len(pts), detector_lost_pct=100 * pts.lost.mean())
    for w in ("post", "later"):
        if f"{w}_ndvi" not in now:
            row[f"{w}_dndvi"] = np.nan
            continue
        d_now = now[f"{w}_ndvi"] - now["pre_ndvi"]
        d_base = base[f"{w}_ndvi"] - base["pre_ndvi"]
        eff = (d_now - d_base).dropna()
        dm = ((now[f"{w}_ndmi"] - now["pre_ndmi"]) - (base[f"{w}_ndmi"] - base["pre_ndmi"])).dropna()
        row[f"{w}_dndvi"] = float(eff.median())
        row[f"{w}_dndmi"] = float(dm.median())
        row[f"{w}_damaged_pct"] = float(100 * (eff < -0.1).mean())
        row[f"{w}_valid"] = int(len(eff))
        row[f"{w}_clear_looks"] = float(now[f"{w}_n"].median())
    return row


PLOT = [  # storm row, label (Sundarbans omitted: repeated cyclones contaminate its baseline)
    ("melissa", "Melissa, Cat 5\nSW Jamaica, Oct 2025"),
    ("irma", "Irma, Cat 3-4\nEverglades, Sep 2017"),
    ("beryl", "Beryl, Cat 4\nS Jamaica, Jul 2024"),
    ("control: Jamaica east (Beryl date)", "control: E Jamaica\n(missed by Beryl)"),
    ("control: Baja (no storm)", "control: Baja\n(no storm)"),
]


def plot(path="results/change_report/storm_check.csv", out="results/change_report/damage_vs_loss.png"):
    import matplotlib.pyplot as plt
    t = pd.read_csv(path).set_index("storm")
    series = [("post_damaged_pct", "canopy damaged, 0-4 months after (Sentinel-2)", "#e34948"),
              ("later_damaged_pct", "still damaged, ~1 year after (Sentinel-2)", "#f6c4c3"),
              ("detector_lost_pct", "mapped as lost by our model (annual map)", "#3b3b3b")]
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    h = 0.26
    for k, (name, _) in enumerate(PLOT):
        for j, (col, lab, c) in enumerate(series):
            v = t.loc[name, col]
            y = k + (j - 1) * (h + 0.02)
            ax.barh(y, v, h, color=c, label=lab if k == 0 else None)
            ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8.5, color="#333")
    ax.set_yticks(range(len(PLOT)), [lab for _, lab in PLOT], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of sampled mangrove pixels", color="#555")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors="#555", length=0)
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Storms damage far more mangrove than they remove", x=0.02, ha="left", fontsize=13)
    fig.text(0.02, 0.905, "Damaged = NDVI drop > 0.1 beyond the same season a year earlier. Most damaged canopy "
             "recovers; the annual map flags only\nthe part that stops looking like mangrove.",
             fontsize=8.5, color="#555", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--storms", nargs="*")
    ap.add_argument("--plot-only", action="store_true", help="redraw damage_vs_loss.png from the CSV")
    a = ap.parse_args()
    if a.plot_only:
        return plot()
    ee.Initialize(project=config.load("configs/default.yaml").project)
    rows = []
    for name, (region, zone, landfall, before, after) in STORMS.items():
        if a.storms and not any(s in name for s in a.storms):
            continue
        pts = sample_points(region, zone, before, after, a.n)
        now = window_values(pts, landfall)
        base = window_values(pts, landfall, shift_years=1)
        rows.append(summarize(name, pts, now, base))
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    path = Path("results/change_report/storm_check.csv")
    if path.exists() and a.storms:  # merge with earlier runs
        out = pd.concat([pd.read_csv(path)[lambda t: ~t.storm.isin(out.storm)], out], ignore_index=True)
    out.round(4).to_csv(path, index=False)
    print(out.round(3).to_string(index=False))
    plot()


if __name__ == "__main__":
    main()
