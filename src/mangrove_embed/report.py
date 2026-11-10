"""
Change report for partners: observed mangrove change 2017-2025, major-loss hotspots, and the
(experimental) forecasts, as plain-language figures, tables and GIS files (`mangrove change-report`).

Runs the shipped bundle (classifier + 2-year forecaster) over every temporal chip:
  maps     classifier map per year; forecasts of 2025 (from 2023-24) and 2026 (from 2024-25)
           -> data/temporal/_report/<region>.npz (cache; delete to recompute)

Observed change uses stable endpoints: mangrove "before" = mangrove in both 2017 and 2018,
"after" = in both 2024 and 2025. Lost / gained pixels are then split by patch size:
  confirmed   part of a connected (8-neighbour) patch of >= 0.5 ha (50 px) of loss or gain
  boundary    the rest: mostly 1-2 px shifts along patch edges, below what 10 m maps resolve
Area rates use the "mangrove" and "random" chips only (the "change" chips were placed on GMW
change and would inflate rates); hotspots use every chip, deduplicated. Rates are shares of the
sampled area, not regional totals.

Forecasts: skill (2025) is scored on the bundle's test blocks only. Hectare forecasts are sums of
forecast probabilities and are labelled experimental: per region they were off by up to ~3x in 2025.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
M = 0           # mangrove class index
NONE = 255
HA = 0.01       # 10 m pixel
MIN_PATCH = 50  # px = 0.5 ha
FIELDS = ("cls", "f25_next", "f25_pchange", "f25_pm", "base25", "f26_next", "f26_pchange", "f26_pm")
NICE = {"everglades": "Everglades", "jamaica": "Jamaica", "mahakam": "Mahakam", "matang": "Matang",
        "mekong": "Mekong", "mexico_pacific": "Baja California Sur", "niger_delta": "Niger Delta",
        "papua": "Papua", "para": "Pará", "rufiji": "Rufiji", "sundarbans": "Sundarbans"}
BAJA = "mexico_pacific"
# what a lost pixel looks like in 2025 (class index -> group); a hint at cause, not a diagnosis
NOW = {1: "other vegetation", 2: "water / tidal flat", 3: "water / tidal flat", 4: "water / tidal flat",
       5: "built-up", 6: "bare ground", 7: "other vegetation", 8: "cropland", 9: "other vegetation",
       10: "other vegetation"}
NOW_ORDER = ["water / tidal flat", "bare ground", "built-up", "cropland", "other vegetation"]


# ------------------------------------------------------------------ inference
def chip_maps(bundle, path):
    """One chip -> dict of [H, W] (or [T, H, W]) maps; NONE / nan where any year lacks data."""
    import torch
    d = np.load(path)
    x, years = d["x"], [int(v) for v in d["years"]]
    T, _, H, W = x.shape
    ok = (np.abs(x.astype(np.float32)).sum(1) > 0).all(0).ravel()
    e = torch.from_numpy(x).to(bundle.device).float().permute(0, 2, 3, 1).reshape(T, H * W, 64)
    probs = bundle.classify(e.reshape(-1, 64)).reshape(T, H * W, -1)
    cls = probs.argmax(2).astype(np.uint8)
    cls[:, ~ok] = NONE
    j = {y: i for i, y in enumerate(years)}
    out = dict(cls=cls.reshape(T, H, W), base25=(1 - probs[j[2024]].max(1)))
    for tag, (a, b) in (("f25", (2023, 2024)), ("f26", (2024, 2025))):
        f = bundle.forecast(torch.stack([e[j[a]], e[j[b]]], 1))
        out[f"{tag}_next"] = f["next"].astype(np.uint8)
        out[f"{tag}_pchange"], out[f"{tag}_pm"] = f["p_change"], f["p_mangrove"]
    for k in FIELDS[1:]:
        v = out[k].astype(np.uint8 if k.endswith("next") else np.float16)
        v[~ok] = NONE if v.dtype == np.uint8 else np.nan
        out[k] = v.reshape(H, W)
    return out, years


def compute(bundle_path, temporal_dir="data/temporal", device=None):
    """Run the bundle over every chip; one cached npz per region. -> (index df, years)."""
    from .deploy import Bundle
    from .seg.train import add_blocks, load_index, split_blocks
    df = add_blocks(load_index(temporal_dir), 0.1)
    b = Bundle.load(bundle_path, device)
    df["test"] = split_blocks(df, 0.2, b.meta["config"]["seed"] + 1)  # = change.run's test blocks
    cache = Path(temporal_dir) / "_report"
    cache.mkdir(parents=True, exist_ok=True)
    years = None
    for region, g in df.groupby("region"):
        f = cache / f"{region}.npz"
        if f.exists() and list(np.load(f, allow_pickle=True)["chips"]) == list(g.chip):
            years = list(np.load(f, allow_pickle=True)["years"])
            continue
        log.info("%s: %d chips", region, len(g))
        maps = {k: [] for k in FIELDS}
        for p in g.path:
            m, years = chip_maps(b, p)
            for k in FIELDS:
                maps[k].append(m[k])
        np.savez(f, chips=np.array(g.chip, dtype=str), years=np.array(years),
                 **{k: np.stack(v) for k, v in maps.items()})
    return df, [int(y) for y in years]


def load(region, temporal_dir="data/temporal"):
    d = dict(np.load(Path(temporal_dir) / "_report" / f"{region}.npz", allow_pickle=True))
    d.update(change_maps(d["cls"]))
    return d


# ------------------------------------------------------------------ observed change
def change_maps(cls):
    """[n, T, H, W] -> stable endpoints, loss / gain split into confirmed (patch >= 0.5 ha) and
    boundary, the year since which each lost pixel has been non-mangrove, and patch labels."""
    from scipy import ndimage as ndi
    ok = (cls != NONE).all(1)
    m = cls == M
    before, after = m[:, 0] & m[:, 1] & ok, m[:, -2] & m[:, -1] & ok
    nb, na = ~m[:, 0] & ~m[:, 1] & ok, ~m[:, -2] & ~m[:, -1] & ok
    loss, gain = before & na, nb & after
    out = dict(ok=ok, before=before, after=after, loss=loss, gain=gain,
               loss_lab=np.zeros(cls.shape[:1] + cls.shape[2:], np.int32))
    eight = np.ones((3, 3), bool)
    for k in ("loss", "gain"):
        conf = np.zeros_like(loss)
        for i in range(len(cls)):
            lab, n = ndi.label(out[k][i], eight)
            if not n:
                continue
            big = np.flatnonzero(np.bincount(lab.ravel()) >= MIN_PATCH)
            big = big[big > 0]
            conf[i] = np.isin(lab, big)
            if k == "loss":
                out["loss_lab"][i] = np.where(conf[i], lab, 0)
        out[f"{k}_conf"] = conf
    T = cls.shape[1]
    last = T - 1 - np.argmax(m[:, ::-1], axis=1)       # last year index that was mangrove
    out["loss_since"] = np.where(loss, last + 1, -1).astype(np.int8)  # index of first non-mangrove year
    return out


def capture(score, pos):
    """Share of positives caught when flagging the top-q share of pixels, at q = 0..1."""
    o = np.argsort(-score, kind="stable")
    hit = np.cumsum(pos[o]) / max(pos.sum(), 1)
    q = np.arange(1, len(o) + 1) / len(o)
    grid = np.linspace(0, 1, 201)
    return grid, np.interp(grid, np.r_[0, q], np.r_[0, hit])


def stats(df, years, temporal_dir="data/temporal"):
    """-> dict of tables (region rows + 'all') and the capture curves."""
    ext, gl, ann, cal, out, now, when = [], [], [], [], [], [], []
    pool = {k: [] for k in ("chg", "s_model", "s_base", "loss", "r_model", "r_base")}
    for region, g in df.groupby("region"):
        d = load(region, temporal_dir)
        area = g.kind.isin(["mangrove", "random"]).values
        test = g.test.values & area
        cls = d["cls"][area]
        ok = d["ok"][area]
        for t, y in enumerate(years):
            ext.append(dict(region=region, year=y, mangrove_ha=float((cls[:, t] == M)[ok].sum() * HA)))
        s = {k: float(d[k][area].sum() * HA) for k in ("ok", "before", "after", "loss", "gain", "loss_conf", "gain_conf")}
        gl.append(dict(region=region, sampled_ha=s["ok"], before_ha=s["before"], after_ha=s["after"],
                       loss_confirmed_ha=s["loss_conf"], loss_boundary_ha=s["loss"] - s["loss_conf"],
                       gain_confirmed_ha=s["gain_conf"], gain_boundary_ha=s["gain"] - s["gain_conf"]))
        lc = d["loss_conf"][area]
        c25 = cls[:, -1][lc]
        row = dict(region=region)
        for grp in NOW_ORDER:
            row[grp] = float(np.isin(c25, [k for k, v in NOW.items() if v == grp]).sum() * HA)
        now.append(row)
        since = d["loss_since"][area][lc]
        when.append(dict(region=region, **{str(y): float((since == j).sum() * HA) for j, y in enumerate(years)
                                           if 1 < j < len(years) - 1}))
        for t in range(len(years) - 1):  # year-to-year gross change (includes flicker)
            a, b = cls[:, t] == M, cls[:, t + 1] == M
            ann.append(dict(region=region, year=years[t + 1], from_ha=float(a[ok].sum() * HA),
                            gain_ha=float((~a & b)[ok].sum() * HA), loss_ha=float((a & ~b)[ok].sum() * HA)))

        # 2025 forecast on test blocks
        c = d["cls"][test]
        v = (c[:, 7] != NONE) & (c[:, 8] != NONE)
        prev, nxt = c[:, 7][v], c[:, 8][v]
        pc, base, pm = (d[k][test][v].astype(np.float32) for k in ("f25_pchange", "base25", "f25_pm"))
        pool["chg"].append(prev != nxt); pool["s_model"].append(pc); pool["s_base"].append(base)
        mg = prev == M
        pool["loss"].append(nxt[mg] != M); pool["r_model"].append(1 - pm[mg]); pool["r_base"].append(base[mg])
        cal.append(dict(region=region, mangrove_2024_ha=float(mg.sum() * HA),
                        loss_forecast_ha=float((1 - pm[mg]).sum() * HA),
                        loss_observed_ha=float((mg & (nxt != M)).sum() * HA),
                        gain_forecast_ha=float(pm[~mg].sum() * HA),
                        gain_observed_ha=float((~mg & (nxt == M)).sum() * HA)))

        # 2026 forecast, all sampled chips
        cur = cls[:, 8] == M
        pm6 = d["f26_pm"][area].astype(np.float32)
        okn = ~np.isnan(pm6)
        out.append(dict(region=region, mangrove_2025_ha=float(cur[okn].sum() * HA),
                        loss_forecast_ha=float((1 - pm6)[cur & okn].sum() * HA),
                        gain_forecast_ha=float(pm6[~cur & okn].sum() * HA)))

    gl, ann, cal, out, now, when = (pd.DataFrame(r) for r in (gl, ann, cal, out, now, when))
    for t in (gl, cal, out, now, when):
        t.loc[len(t)] = dict(region="all", **t.drop(columns="region").sum())
    for k in ("loss_confirmed", "loss_boundary", "gain_confirmed", "gain_boundary"):
        gl[f"{k}_pct"] = 100 * gl[f"{k}_ha"] / gl.before_ha
    gl["net_confirmed_pct"] = gl.gain_confirmed_pct - gl.loss_confirmed_pct
    gl["rank_confirmed_loss"] = gl.loss_confirmed_pct.where(gl.region != "all").rank(ascending=False)
    a = ann.groupby("region")[["from_ha", "gain_ha", "loss_ha"]].sum()
    a.loc["all"] = a.sum()
    a["loss_pct_per_yr"], a["gain_pct_per_yr"] = (100 * a[k] / a.from_ha for k in ("loss_ha", "gain_ha"))
    out = out.merge(a[["loss_pct_per_yr", "gain_pct_per_yr"]].reset_index(), on="region")
    out["loss_pct"] = 100 * out.loss_forecast_ha / out.mangrove_2025_ha
    out["gain_pct"] = 100 * out.gain_forecast_ha / out.mangrove_2025_ha
    cal["loss_ratio"] = cal.loss_forecast_ha / cal.loss_observed_ha
    cal["gain_ratio"] = cal.gain_forecast_ha / cal.gain_observed_ha

    cat = {k: np.concatenate(v) for k, v in pool.items()}
    curves, skill = {}, []
    for name, pos, s in (("any change", cat["chg"], {"forecast": cat["s_model"], "today's map only": cat["s_base"]}),
                         ("mangrove loss", cat["loss"], {"forecast": cat["r_model"], "today's map only": cat["r_base"]})):
        for meth, sc in s.items():
            q, h = capture(sc, pos)
            curves[(name, meth)] = (q, h)
            for top in (5, 10, 20):
                skill.append(dict(target=name, method=meth, flagged_pct=top, base_rate_pct=100 * pos.mean(),
                                  caught_pct=100 * float(np.interp(top / 100, q, h))))
    return dict(extent=pd.DataFrame(ext), change=gl, annual=a.reset_index(), skill=pd.DataFrame(skill),
                calibration=cal, forecast_2026=out, loss_now=now, loss_year=when, curves=curves)


def hotspots(df, years, temporal_dir="data/temporal"):
    """Every confirmed loss patch in every chip: size, location, year, what it looks like now.
    Patches seen in two overlapping chips are kept once (the larger)."""
    from pyproj import Transformer
    rows = []
    for region, g in df.groupby("region"):
        d = load(region, temporal_dir)
        for i, r in enumerate(g.itertuples()):
            lab = d["loss_lab"][i]
            ids = np.unique(lab[lab > 0])
            if not len(ids):
                continue
            tr = Transformer.from_crs(r.crs, "EPSG:4326", always_xy=True)
            for k in ids:
                rr, cc = np.nonzero(lab == k)
                c25 = d["cls"][i, -1][rr, cc]
                grp = pd.Series([NOW.get(int(v), "other vegetation") for v in c25]).value_counts()
                since = d["loss_since"][i][rr, cc]
                lon, lat = tr.transform(r.x0 + (cc.mean() + 0.5) * 10, r.y1 - (rr.mean() + 0.5) * 10)
                rows.append(dict(region=region, chip=r.chip, chip_index=i, patch=int(k), ha=len(rr) * HA,
                                 lat=lat, lon=lon, lost_since=years[int(np.bincount(since).argmax())],
                                 now=grp.index[0], now_share=float(grp.iloc[0] / len(rr)),
                                 row=float(rr.mean()), col=float(cc.mean())))
    h = pd.DataFrame(rows).sort_values("ha", ascending=False).reset_index(drop=True)
    keep, seen = [], []
    for r in h.itertuples():  # drop a patch whose centre is within 100 m of a kept one
        if any(abs(r.lat - a) < 0.0009 and abs(r.lon - b) < 0.0009 / np.cos(np.radians(a)) for a, b in seen):
            continue
        keep.append(r.Index)
        seen.append((r.lat, r.lon))
    return h.loc[keep].reset_index(drop=True)


def write_points(h, path):
    feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[round(r.lon, 6), round(r.lat, 6)]),
                  properties=dict(region=NICE[r.region], ha=round(r.ha, 2), lost_since=int(r.lost_since),
                                  now=r.now, chip=r.chip)) for r in h.itertuples()]
    Path(path).write_text(json.dumps(dict(type="FeatureCollection", features=feats)))


# ------------------------------------------------------------------ figures
def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import _style
    return plt, _style


from .plots import BLUE, GRID, INK, INK2, SURFACE  # noqa: E402
GAIN, LOSS = "#2a78d6", "#e34948"
GAIN_B, LOSS_B = "#b7d3f6", "#f6c4c3"
STABLE, OTHER, NODATA_C = "#a9d3a9", "#ecebe7", "#ffffff"
EXPERIMENTAL = "#eb6834"
NOW_COLORS = {"water / tidal flat": "#2a78d6", "bare ground": "#eda100", "built-up": "#6250d6",
              "cropland": "#e87ba4", "other vegetation": "#1baf7a"}


def _save(fig, out, tries=5):
    """savefig, retrying briefly: on Windows a viewer or scanner can hold the old file open."""
    import time
    for k in range(tries):
        try:
            return fig.savefig(out, dpi=150, facecolor=SURFACE)
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1.0)


def _name(r):
    return NICE.get(r, "All regions")


def _badge(fig, text="EXPERIMENTAL PROTOTYPE", x=0.98, y=0.975):
    fig.text(x, y, text, ha="right", va="top", fontsize=9.5, color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc=EXPERIMENTAL, ec="none"))


def _change_rgb(d, i):
    """Chip i -> RGB: unchanged mangrove, confirmed / boundary loss and gain, other, no data."""
    import matplotlib.colors as mc
    img = np.empty(d["cls"].shape[2:] + (3,))
    img[:] = mc.to_rgb(NODATA_C)
    img[d["ok"][i]] = mc.to_rgb(OTHER)
    img[d["before"][i]] = mc.to_rgb(STABLE)
    for k, c in (("gain", GAIN_B), ("loss", LOSS_B), ("gain_conf", GAIN), ("loss_conf", LOSS)):
        img[d[k][i]] = mc.to_rgb(c)
    return img


CHANGE_KEY = [("mangrove, unchanged", STABLE), ("confirmed loss", LOSS), ("edge loss", LOSS_B),
              ("confirmed gain", GAIN), ("edge gain", GAIN_B), ("other land / water", OTHER)]


def _legend(fig, items, y=0.01):
    import matplotlib.patches as mp
    fig.legend([mp.Patch(fc=c, ec=GRID) for _, c in items], [n for n, _ in items], loc="lower center",
               ncol=len(items), frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, y))


def _tile(ax, img, title, sub=None):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title + (f"\n{sub}" if sub else ""), fontsize=9, color=INK, loc="left", linespacing=1.4)


def fig_health(st, out):
    """Confirmed and edge loss / gain as % of 2017-18 mangrove, per region, Baja highlighted."""
    plt, style = _fig()
    t = st["change"].set_index("region")
    regions = [r for r in t.sort_values("loss_confirmed_pct", ascending=False).index if r != "all"] + ["all"]
    regions = regions[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(regions))
    T = t.loc[regions]
    h = 0.66
    ax.barh(y, -T.loss_confirmed_pct, h, color=LOSS, label="confirmed loss (patches ≥ 0.5 ha)")
    ax.barh(y, -T.loss_boundary_pct, h, left=-T.loss_confirmed_pct, color=LOSS_B, label="edge loss (uncertain)")
    ax.barh(y, T.gain_confirmed_pct, h, color=GAIN, label="confirmed gain")
    ax.barh(y, T.gain_boundary_pct, h, left=T.gain_confirmed_pct, color=GAIN_B, label="edge gain (uncertain)")
    ax.axvline(0, color=INK2, lw=1)
    lim = 1.05 * max((T.loss_confirmed_pct + T.loss_boundary_pct).max(), (T.gain_confirmed_pct + T.gain_boundary_pct).max())
    for i, r in enumerate(regions):
        tot = T.loc[r].loss_confirmed_pct + T.loc[r].loss_boundary_pct
        ax.text(-tot - lim * 0.015, i, f"{T.loc[r].loss_confirmed_pct:.1f}%", ha="right", va="center",
                fontsize=9, color=INK, fontweight="bold" if r in (BAJA, "all") else None)
    labels = [("▶ " if r == BAJA else "") + _name(r) for r in regions]
    ax.set_yticks(y, labels, fontsize=10)
    for lab, r in zip(ax.get_yticklabels(), regions):
        if r in (BAJA, "all"):
            lab.set_fontweight("bold")
    ax.set_xlim(-lim * 1.18, lim)
    ax.xaxis.set_major_formatter(lambda v, _: f"{abs(v):.0f}%")
    ax.set_xlabel("% of 2017-18 mangrove area      ◀ lost  |  gained ▶", color=INK2)
    ax.legend(loc="lower center", frameon=False, fontsize=9, ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("How stable are the mangroves? Change from 2017-18 to 2024-25, by region",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.915, "Sorted by confirmed loss (number at left). Edge loss and gain are 1-2 pixel shifts "
             "along patch edges, mostly map noise.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out)
    plt.close(fig)


def fig_extent(st, out):
    plt, style = _fig()
    e = st["extent"]
    regions = [r for r in st["change"].sort_values("before_ha", ascending=False).region if r != "all"]
    fig, axs = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True, facecolor=SURFACE)
    tot = e.groupby("year").mangrove_ha.sum()
    series = [("all", tot)] + [(r, e[e.region == r].set_index("year").mangrove_ha) for r in regions]
    for ax, (r, s) in zip(axs.flat, series):
        style(ax)
        idx = 100 * s / s.iloc[0]
        ax.plot(idx.index, idx.values, color=BLUE, lw=2)
        ax.plot(idx.index[-1], idx.values[-1], "o", ms=6, color=BLUE, mec=SURFACE, mew=2)
        ax.axhline(100, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.set_title(("▶ " if r == BAJA else "") + _name(r), fontsize=10.5, color=INK, loc="left",
                     fontweight="bold" if r in ("all", BAJA) else None)
        ax.text(idx.index[-1], idx.values[-1], f"  {idx.values[-1] - 100:+.0f}%", va="center", fontsize=9, color=INK2)
        lo, hi = idx.min(), idx.max()
        pad = max(3, (hi - lo) * 0.25)
        ax.set_ylim(min(lo, 100) - pad, max(hi, 100) + pad)
        ax.set_xlim(2016.6, 2026.4)
        ax.set_xticks([2017, 2021, 2025])
    fig.suptitle("Mangrove extent each year in the sampled areas (2017 = 100)", x=0.06, ha="left", fontsize=13, color=INK)
    fig.text(0.06, 0.935, "Year-to-year wiggles are partly map noise; read the trend.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0.03, 0, 1, 0.93))
    _save(fig, out)
    plt.close(fig)


def fig_loss_context(st, out):
    """For confirmed loss: what the area looks like in 2025, and since when it has been lost."""
    plt, style = _fig()
    now = st["loss_now"].set_index("region")
    chg = st["change"].set_index("region")
    regions = [r for r in chg.sort_values("loss_confirmed_ha").index if r != "all"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.1, 1]))
    style(a1)
    a1.grid(axis="y", visible=False)
    y = np.arange(len(regions))
    left = np.zeros(len(regions))
    for grp in NOW_ORDER:
        v = now.loc[regions, grp].values
        a1.barh(y, v, 0.66, left=left, color=NOW_COLORS[grp], label=grp, ec=SURFACE, lw=1)
        left += v
    for i, r in enumerate(regions):
        a1.text(left[i], i, f"  {left[i]:,.0f} ha", va="center", fontsize=9, color=INK2)
    a1.set_yticks(y, [("▶ " if r == BAJA else "") + _name(r) for r in regions], fontsize=10)
    a1.set_xlabel("confirmed mangrove loss in the sample (ha)", color=INK2)
    a1.set_xlim(0, left.max() * 1.18)
    a1.legend(frameon=False, fontsize=9, loc="lower right", title="what it looks like in 2025", title_fontsize=9)
    a1.set_title("What replaced the lost mangrove", loc="left", fontsize=11.5, color=INK)

    w = st["loss_year"].set_index("region").loc[regions[::-1]]
    share = w.div(w.sum(1).replace(0, np.nan), axis=0) * 100
    import matplotlib.colors as mc
    cmap = mc.LinearSegmentedColormap.from_list("seq", ["#f7f6f3", "#f6c4c3", "#e34948", "#8f1f1e"])
    a2.imshow(share.values, cmap=cmap, vmin=0, vmax=60, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            if v >= 10:
                a2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 35 else INK)
    a2.set_xticks(range(share.shape[1]), share.columns, fontsize=9.5)
    a2.set_yticks(range(share.shape[0]), [("▶ " if r == BAJA else "") + _name(r) for r in share.index], fontsize=10)
    a2.tick_params(colors=INK2, length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title("Year the loss happened (% of each region's confirmed loss)", loc="left", fontsize=11.5, color=INK)
    fig.suptitle("Clues to the causes of confirmed mangrove loss, 2017-18 to 2024-25",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.905, "Land cover after loss is a hint, not a diagnosis: aquaculture ponds and erosion both "
             "read as water; dieback can read as bare ground. Year = first year it stayed non-mangrove.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, out)
    plt.close(fig)


def fig_hotspots(h, df, out, temporal_dir="data/temporal", n=12, regions=None, title=None):
    """The n largest confirmed loss patches, one tile each, the patch circled."""
    plt, _ = _fig()
    sel = h if regions is None else h[h.region.isin(regions)]
    sel = sel.drop_duplicates("chip").head(n)
    cols = 4
    rows = int(np.ceil(len(sel) / cols)) or 1
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.1 * rows + 1), facecolor=SURFACE, squeeze=False)
    cache = {}
    for ax, r in zip(axs.flat, sel.itertuples()):
        d = cache.setdefault(r.region, load(r.region, temporal_dir))
        _tile(ax, _change_rgb(d, r.chip_index), f"{_name(r.region)}  {r.lat:.3f}, {r.lon:.3f}",
              f"{r.ha:.1f} ha lost since {r.lost_since}\nnow looks like {r.now}")
        ax.contour(d["loss_lab"][r.chip_index] == r.patch, levels=[0.5], colors=INK, linewidths=1.1)
    for ax in list(axs.flat)[len(sel):]:
        ax.axis("off")
    fig.suptitle(title or "Largest confirmed mangrove losses, 2017-18 to 2024-25 (tiles 1.28 km across; "
                 "black outline = the patch)", x=0.02, ha="left", fontsize=13, color=INK)
    _legend(fig, CHANGE_KEY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.2)
    _save(fig, out)
    plt.close(fig)


def fig_skill(st, out):
    plt, style = _fig()
    sk, cal = st["skill"], st["calibration"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for ax, target in zip(axs[:2], ("any change", "mangrove loss")):
        style(ax)
        for meth, color, lw in (("forecast", BLUE, 2.4), ("today's map only", INK2, 1.6)):
            q, h = st["curves"][(target, meth)]
            ax.plot(100 * q, 100 * h, color=color, lw=lw, label=meth)
        ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls=(0, (3, 3)), label="random")
        r = sk[(sk.target == target) & (sk.flagged_pct == 10)].set_index("method").caught_pct
        ax.plot(10, r["forecast"], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
        base = r["today's map only"]
        ax.annotate(f"top 10% flagged → {r['forecast']:.0f}% caught\n(today's map alone: {base:.0f}%)",
                    (10, r["forecast"]), xytext=(15, r["forecast"] - 22), fontsize=9.5, color=INK)
        ax.set_xlim(0, 50); ax.set_ylim(0, 100)
        ax.set_xlabel(("% of mangrove flagged" if target == "mangrove loss" else "% of area flagged")
                      + ", highest risk first", color=INK2)
        ax.set_ylabel(f"% of 2025 {target} caught", color=INK2)
        ax.set_title("Where will land cover change?" if target == "any change" else "Where will mangrove be lost?",
                     loc="left", fontsize=11.5, color=INK)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax = axs[2]
    style(ax)
    c = cal[cal.region != "all"]
    for col, color, name in (("loss", LOSS, "loss"), ("gain", GAIN, "gain")):
        ax.scatter(c[f"{col}_observed_ha"], c[f"{col}_forecast_ha"], s=60, color=color, ec=SURFACE,
                   lw=1.5, label=f"mangrove {name}", zorder=3)
    b = c[c.region == BAJA].iloc[0]
    ax.annotate("Baja C.S.", (b.loss_observed_ha, b.loss_forecast_ha), xytext=(8, 6), textcoords="offset points",
                fontsize=8.5, color=INK2)
    hi = 1.1 * float(c[["loss_forecast_ha", "loss_observed_ha", "gain_forecast_ha", "gain_observed_ha"]].values.max())
    ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(hi * 0.97, hi * 0.9, "forecast = observed", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("observed in 2025 (ha)", color=INK2)
    ax.set_ylabel("forecast from 2023-24 (ha)", color=INK2)
    ax.set_title("Hectare forecast per region (experimental)", loc="left", fontsize=11.5, color=EXPERIMENTAL)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Testing the 2025 forecast on held-out areas (never used in training)",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.905, "Left, middle: ranking where change happens. For mangrove loss the forecast is no better "
             "than today's map; right: totals are off by up to ~3x per region.", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, out)
    plt.close(fig)


def _risk_cmap():
    import matplotlib.colors as mc
    return mc.LinearSegmentedColormap.from_list("risk", ["#fbe3d9", "#f4a582", "#eb6834", "#b8431a", "#7a2a0d"])


def _risk_img(cls_now, pm):
    import matplotlib.colors as mc
    img = np.empty(cls_now.shape + (4,))
    img[:] = mc.to_rgba(NODATA_C)
    img[cls_now != NONE] = mc.to_rgba(OTHER)
    m = cls_now == M
    img[m] = _risk_cmap()(np.clip((1 - pm.astype(np.float32)) / 0.5, 0, 1)[m])
    return img


def _risk_bar(fig, axes, label):
    plt, _ = _fig()
    sm = plt.cm.ScalarMappable(cmap=_risk_cmap(), norm=plt.Normalize(0, 50))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(label, color=INK2, fontsize=9)
    cb.set_ticks([0, 10, 20, 30, 40, 50], labels=["0", "10", "20", "30", "40", "50+"])


def fig_forecast_2026(st, out):
    plt, style = _fig()
    t = st["forecast_2026"].set_index("region")
    cal = st["calibration"].set_index("region")
    regions = [r for r in t.sort_values("loss_pct").index if r != "all"] + ["all"]
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(regions))
    ax.barh(y + 0.19, t.loc[regions].loss_pct, 0.36, color=LOSS, label="forecast loss, 2026")
    ax.barh(y - 0.19, t.loc[regions].gain_pct, 0.36, color=GAIN, label="forecast gain, 2026")
    for i, r in enumerate(regions):
        ratio = cal.loc[r].loss_ratio
        ax.text(max(t.loc[r].loss_pct, t.loc[r].gain_pct) + 0.15, i,
                f"{t.loc[r].loss_forecast_ha:,.0f} ha lost · 2025 test: {ratio:.1f}× observed",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [("▶ " if r == BAJA else "") + _name(r) for r in regions], fontsize=10)
    ax.set_xlabel("% of 2025 mangrove area in the sample", color=INK2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlim(0, t.loc[regions][["loss_pct", "gain_pct"]].values.max() * 1.75)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Forecast mangrove change in 2026, from 2024-25", x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.905, "Includes year-to-year map flicker, so it overstates lasting change. The ratio at right "
             "is how the same forecast compared with what was observed in 2025.", fontsize=9.5, color=INK2)
    _badge(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, out)
    plt.close(fig)


def fig_forecast_maps(df, out, temporal_dir="data/temporal", n=4):
    """Held-out chips with the most observed 2025 mangrove loss: risk forecast vs what happened."""
    import matplotlib.colors as mc
    plt, _ = _fig()
    picks = []
    for region, g in df.groupby("region"):
        d = load(region, temporal_dir)
        idx = np.flatnonzero(g.test.values)
        if not len(idx):
            continue
        c = d["cls"][idx]
        lost = ((c[:, 7] == M) & (c[:, 8] != M) & (c[:, 8] != NONE)).sum((1, 2))
        i = idx[int(np.argmax(lost))]
        picks.append((lost.max(), region, d["cls"][i], d["f25_pm"][i], g.iloc[i]))
    picks = sorted(picks, key=lambda p: -p[0])[:n]
    fig, axs = plt.subplots(2, n, figsize=(3.3 * n + 1, 7.6), facecolor=SURFACE)
    for k, (_, region, cls, pm, row) in enumerate(picks):
        _tile(axs[0, k], _risk_img(cls[7], pm), f"{_name(region)}  {row.lat:.3f}, {row.lon:.3f}")
        img = np.empty(cls.shape[1:] + (3,))
        img[:] = mc.to_rgb(NODATA_C)
        img[cls[8] != NONE] = mc.to_rgb(OTHER)
        img[(cls[7] == M) & (cls[8] == M)] = mc.to_rgb(STABLE)
        img[(cls[7] == M) & (cls[8] != M) & (cls[8] != NONE)] = mc.to_rgb(LOSS)
        img[(cls[7] != M) & (cls[7] != NONE) & (cls[8] == M)] = mc.to_rgb(GAIN)
        _tile(axs[1, k], img, "")
    axs[0, 0].set_ylabel("forecast risk of loss\n(made from 2023-24)", fontsize=10, color=INK)
    axs[1, 0].set_ylabel("what happened in 2025", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.87, bottom=0.08, wspace=0.05, hspace=0.15)
    _risk_bar(fig, axs[0, :], "chance mangrove is lost (%)")
    fig.suptitle("2025 forecast on held-out areas: predicted risk (top) vs observed change (bottom)",
                 x=0.02, ha="left", fontsize=13, color=INK)
    _legend(fig, [("mangrove kept", STABLE), ("mangrove lost", LOSS), ("mangrove gained", GAIN),
                  ("other land / water", OTHER)], y=0.005)
    _save(fig, out)
    plt.close(fig)


# ------------------------------------------------------------------ Baja California Sur
PLACES = {"Bahía Magdalena": (-112.05, 24.6), "Puerto San Carlos": (-112.11, 24.79),
          "Bahía Almejas": (-111.75, 24.4), "La Paz": (-110.31, 24.14)}


def baja_tiles(df, temporal_dir="data/temporal"):
    g = df[df.region == BAJA].reset_index(drop=True)
    d = load(BAJA, temporal_dir)
    now = d["cls"][:, 8] == M
    pm6 = d["f26_pm"].astype(np.float32)
    t = g[["chip", "kind", "lat", "lon", "x0", "y1", "crs", "test"]].copy()
    s = lambda k: d[k].sum((1, 2)) * HA  # noqa: E731
    t["mangrove_2017_18_ha"], t["mangrove_2024_25_ha"] = s("before"), s("after")
    t["loss_confirmed_ha"], t["loss_edge_ha"] = s("loss_conf"), s("loss") - s("loss_conf")
    t["gain_confirmed_ha"], t["gain_edge_ha"] = s("gain_conf"), s("gain") - s("gain_conf")
    t["forecast_loss_2026_ha_experimental"] = np.nansum(np.where(now, 1 - pm6, 0), (1, 2)) * HA
    return t, d


def write_gis(t, d, out, px=128):
    """One GeoTIFF per tile and a GeoJSON of tile footprints with the tile's numbers."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin
    out.mkdir(parents=True, exist_ok=True)
    feats = []
    for i, r in t.iterrows():
        code = np.full((px, px), np.nan, np.float32)
        code[d["ok"][i]] = 1
        code[d["before"][i]] = 2
        code[d["gain"][i]] = 3
        code[d["loss"][i]] = 4
        code[d["gain_conf"][i]] = 5
        code[d["loss_conf"][i]] = 6
        risk = np.where(d["cls"][i, 8] == M, 1 - d["f26_pm"][i].astype(np.float32), np.nan).astype(np.float32)
        with rasterio.open(out / f"{r.chip}.tif", "w", driver="GTiff", width=px, height=px, count=2,
                           dtype="float32", crs=r.crs, transform=from_origin(r.x0, r.y1, 10, 10),
                           nodata=np.nan, compress="deflate") as f:
            f.write(code, 1)
            f.write(risk, 2)
            f.set_band_description(1, "2017-18 to 2024-25: 1 other, 2 mangrove kept, 3 edge gain, 4 edge loss, "
                                      "5 confirmed gain, 6 confirmed loss")
            f.set_band_description(2, "EXPERIMENTAL: chance mangrove is lost in 2026 (mangrove pixels only)")
        tr = Transformer.from_crs(r.crs, "EPSG:4326", always_xy=True)
        s = px * 10
        ring = [list(tr.transform(x, y)) for x, y in ((r.x0, r.y1), (r.x0 + s, r.y1), (r.x0 + s, r.y1 - s),
                                                      (r.x0, r.y1 - s), (r.x0, r.y1))]
        props = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else
                     bool(v) if isinstance(v, (bool, np.bool_)) else v)
                 for k, v in r.drop(["x0", "y1", "crs"]).items()}
        props["geotiff"] = f"tiles/{r.chip}.tif"
        feats.append(dict(type="Feature", properties=props, geometry=dict(type="Polygon", coordinates=[ring])))
    (out.parent / "baja_tiles.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats)))


def fig_baja_overview(t, st, out):
    import matplotlib.colors as mc
    import matplotlib.patches as mp
    plt, style = _fig()
    fig = plt.figure(figsize=(14, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    style(ax)
    tot = t.loss_confirmed_ha + t.loss_edge_ha
    vmax = max(1.0, float(np.percentile(tot[tot > 0], 95))) if (tot > 0).any() else 1.0
    cmap = mc.LinearSegmentedColormap.from_list("loss", ["#f6c4c3", LOSS, "#8f1f1e"])
    dlat = 1.28 / 111
    for (_, r), lt in zip(t.iterrows(), tot):
        dlon = dlat / np.cos(np.radians(r.lat))
        has = r.mangrove_2017_18_ha + r.gain_confirmed_ha + r.gain_edge_ha > 0
        fc = "#ffffff" if not has else (STABLE if lt == 0 else cmap(min(lt / vmax, 1)))
        ax.add_patch(mp.Rectangle((r.lon - dlon / 2, r.lat - dlat / 2), dlon, dlat, fc=fc,
                                  ec=INK2 if has else GRID, lw=0.5 if has else 0.4, zorder=3 if has else 2))
        if r.loss_confirmed_ha > 0:
            ax.plot(r.lon, r.lat, "o", ms=6 + 4 * np.sqrt(r.loss_confirmed_ha), mfc="none", mec=INK, mew=1.3,
                    zorder=5)
    for name, (lo, la) in PLACES.items():
        ax.plot(lo, la, "s", ms=4, color=INK2, zorder=4)
        ax.text(lo + 0.03, la - 0.035, name, fontsize=9, color=INK2, zorder=4, style="italic")
    ax.set_xlim(t.lon.min() - 0.1, t.lon.max() + 0.25)
    ax.set_ylim(t.lat.min() - 0.08, t.lat.max() + 0.08)
    ax.set_aspect(1 / np.cos(np.radians(t.lat.mean())))
    ax.set_xlabel("longitude", color=INK2); ax.set_ylabel("latitude", color=INK2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mangrove lost per tile, 2017-18 to 2024-25 (ha, mostly edge)", color=INK2, fontsize=9)
    ax.set_title("Sampled 1.28 km tiles · green: no loss · white: no mangrove · ○ confirmed loss patch (size = ha)",
                 loc="left", fontsize=10.5, color=INK)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    c = st["change"].set_index("region")
    b = c.loc[BAJA]
    allr = c.loc["all"]
    n = int((c.index != "all").sum())
    fc = st["forecast_2026"].set_index("region").loc[BAJA]
    rows = [("Mangrove mapped in sample, 2017-18", f"{b.before_ha:,.0f} ha"),
            ("Confirmed loss (patches ≥ 0.5 ha)", f"{b.loss_confirmed_ha:,.0f} ha ({b.loss_confirmed_pct:.2f}%)"),
            ("", f"rank {int(b.rank_confirmed_loss)} of {n} regions · all regions: {allr.loss_confirmed_pct:.1f}%"),
            ("Confirmed gain", f"{b.gain_confirmed_ha:,.0f} ha ({b.gain_confirmed_pct:.2f}%)"),
            ("Edge shifts (uncertain)", f"−{b.loss_boundary_ha:,.0f} / +{b.gain_boundary_ha:,.0f} ha"),
            None,
            ("2026 forecast loss (experimental)", f"{fc.loss_forecast_ha:,.0f} ha ({fc.loss_pct:.1f}%)"),
            ("", f"same method in 2025: {st['calibration'].set_index('region').loc[BAJA].loss_ratio:.1f}× the "
                 "observed loss")]
    y = 0.96
    for row in rows:
        if row is None:
            y -= 0.05
            continue
        if row[0]:
            ax2.text(0, y, row[0], fontsize=10, color=EXPERIMENTAL if "experimental" in row[0] else INK2,
                     transform=ax2.transAxes)
            ax2.text(0, y - 0.05, row[1], fontsize=14, color=INK, transform=ax2.transAxes, fontweight="bold")
            y -= 0.125
        else:
            ax2.text(0, y + 0.03, row[1], fontsize=9, color=INK2, transform=ax2.transAxes)
            y -= 0.05
    ax2.text(0, 0.0, f"Sample: {int((t.kind != 'change').sum())} area tiles (used for rates) + "
             f"{int((t.kind == 'change').sum())} tiles placed on\nGlobal Mangrove Watch change. Edges shift "
             "1-2 pixels (10-20 m)\nbetween years; this is mostly map noise.",
             fontsize=8.5, color=INK2, transform=ax2.transAxes, va="bottom")
    fig.suptitle("Baja California Sur, Bahía Magdalena to La Paz: no large-scale mangrove loss, 2017-2025",
                 x=0.03, ha="left", fontsize=14, color=INK)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.08)
    _save(fig, out)
    plt.close(fig)


def fig_baja_tiles(t, d, out, n=8):
    """Tiles with confirmed loss first, then the most edge loss."""
    plt, _ = _fig()
    top = t.assign(k=t.loss_confirmed_ha * 1000 + t.loss_edge_ha).sort_values("k", ascending=False).head(n)
    fig, axs = plt.subplots(2, n // 2, figsize=(3.4 * n // 2, 9), facecolor=SURFACE)
    for ax, (i, r) in zip(axs.flat, top.iterrows()):
        _tile(ax, _change_rgb(d, i), f"{r.lat:.3f}, {r.lon:.3f}",
              f"confirmed −{r.loss_confirmed_ha:.1f} ha · edge −{r.loss_edge_ha:.1f} ha")
    fig.suptitle("Baja California Sur: the tiles with the most mangrove loss (all small; several are strips "
                 "along the seaward edge)", x=0.02, ha="left", fontsize=13, color=INK)
    _legend(fig, CHANGE_KEY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=1.5)
    _save(fig, out)
    plt.close(fig)


def fig_baja_2026(t, d, out, n=8):
    plt, _ = _fig()
    top = t.sort_values("forecast_loss_2026_ha_experimental", ascending=False).head(n)
    fig, axs = plt.subplots(2, n // 2, figsize=(3.4 * n // 2 + 1, 8), facecolor=SURFACE)
    for ax, (i, r) in zip(axs.flat, top.iterrows()):
        _tile(ax, _risk_img(d["cls"][i, 8], d["f26_pm"][i]), f"{r.lat:.3f}, {r.lon:.3f}",
              f"~{r.forecast_loss_2026_ha_experimental:.1f} ha forecast lost")
    fig.subplots_adjust(left=0.02, right=0.9, top=0.86, bottom=0.03, wspace=0.06, hspace=0.22)
    _risk_bar(fig, list(axs.flat), "chance mangrove is lost in 2026 (%)")
    fig.suptitle("Baja California Sur: highest forecast mangrove loss in 2026", x=0.02, ha="left", fontsize=13, color=INK)
    _badge(fig)
    _save(fig, out)
    plt.close(fig)


# ------------------------------------------------------------------ hurricanes
JM_ZONES = ["SW coast (Black River, Westmoreland)", "S-central (Manchester, Clarendon)",
            "SE (Portland Bight, Kingston)", "east", "north coast"]


def _jm_zone(lat, lon):
    if lat > 18.25:
        return JM_ZONES[4]
    if lon < -77.6:
        return JM_ZONES[0]
    if lon < -77.0:
        return JM_ZONES[1]
    if lon < -76.7:
        return JM_ZONES[2]
    return JM_ZONES[3]


def hurricane_tables(df, years, temporal_dir="data/temporal", max_chips=60):
    """Jamaica mangrove extent per coastal zone and year, and year-to-year embedding similarity of
    always-mangrove pixels per zone (Jamaica zones, Everglades, Baja)."""
    g = df[df.region == "jamaica"].reset_index(drop=True)
    d = load("jamaica", temporal_dir)
    g["zone"] = [_jm_zone(a, b) for a, b in zip(g.lat, g.lon)]
    ext = []
    for z, gg in g.groupby("zone"):
        i = gg.index.values
        m, ok = d["cls"][i] == M, d["ok"][i]
        ha = [float(m[:, t][ok].sum() * HA) for t in range(len(years))]
        ref = np.mean(ha[4:7])  # 2021-2023
        ext += [dict(zone=z, year=y, mangrove_ha=v, index=100 * v / ref) for y, v in zip(years, ha)]
    sim = []
    groups = [(z, "jamaica", gg.index.values) for z, gg in g.groupby("zone")]
    for reg, name in (("everglades", "Everglades"), (BAJA, "Baja California Sur")):
        groups.append((name, reg, np.arange((df.region == reg).sum())))
    for name, reg, idx in groups:
        dd = d if reg == "jamaica" else load(reg, temporal_dir)
        paths = df[df.region == reg].reset_index(drop=True).path
        acc = {t: [] for t in range(1, len(years))}
        n = 0
        for i in idx:
            mm = (dd["cls"][i] == M).all(0)
            if mm.sum() < 20:
                continue
            x = np.load(paths[i])["x"][:, :, mm].astype(np.float32)
            nrm = np.linalg.norm(x, axis=1)
            for t in acc:
                acc[t].append(float(((x[t - 1] * x[t]).sum(0) / np.maximum(nrm[t - 1] * nrm[t], 1e-6)).mean()))
            n += 1
            if n >= max_chips:
                break
        sim.append(dict(area=name, **{f"{years[t - 1]}-{years[t]}": float(np.mean(v)) for t, v in acc.items()}))
    return pd.DataFrame(ext), pd.DataFrame(sim), g, d


def fig_hurricanes(ext, sim, g, d, out, n=4):
    import matplotlib.colors as mc
    plt, style = _fig()
    fig = plt.figure(figsize=(15, 10.2), facecolor=SURFACE)
    top = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.84, bottom=0.5, wspace=0.3, width_ratios=[1.25, 1])
    ax = fig.add_subplot(top[0])
    style(ax)
    ax.grid(axis="x", visible=False)
    for y0 in range(2017, 2027):  # year boundaries: each point summarises one calendar year
        ax.axvline(y0, color=GRID, lw=0.6, zorder=0)
    for z in JM_ZONES[::-1]:
        e = ext[ext.zone == z]
        sw = z == JM_ZONES[0]
        ax.plot(e.year + 0.5, e["index"], color=LOSS if sw else "#b5b4ae", lw=2.6 if sw else 1.4,
                marker="o", ms=5 if sw else 0, zorder=3 if sw else 2,
                label="SW coast (Black River)" if sw else ("other Jamaican coasts" if z == JM_ZONES[1] else None))
    for x, lab in ((2024 + 184 / 366, "Beryl\nCat 4\n3 Jul 2024"), (2025 + 300 / 365, "Melissa\nCat 5\n28 Oct 2025")):
        ax.axvline(x, color=INK, lw=1.1, ls=(0, (3, 3)))
        right = lab.startswith("Melissa")
        ax.text(x + (0.06 if right else -0.05), 76, lab, fontsize=8.5, color=INK, ha="left" if right else "right",
                va="bottom")
    ax.axhline(100, color=INK2, lw=0.8)
    ax.set_xlim(2017, 2027)
    ax.set_ylim(75, 112)
    ax.set_xticks(range(2017, 2027), [str(y) for y in range(2017, 2027)])
    ax.tick_params(axis="x", length=4, color=INK2)
    ax.set_xlabel("each point = one full-year composite, plotted mid-year between 1 Jan ticks", color=INK2, fontsize=9)
    ax.set_ylabel("mangrove extent (2021-23 = 100)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Jamaica: mangrove extent by coast", loc="left", fontsize=11.5, color=INK)

    ax = fig.add_subplot(top[1])
    s = sim.set_index("area")
    cmap = mc.LinearSegmentedColormap.from_list("sim", ["#8f1f1e", "#e34948", "#f6c4c3", "#f7f6f3"])
    ax.imshow(s.values, cmap=cmap, vmin=0.88, vmax=0.99, aspect="auto")
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            v = s.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v < 0.92 else INK)
    ax.set_xticks(range(s.shape[1]), ["'" + c[2:4] + "-'" + c[7:9] for c in s.columns], fontsize=8.5)
    ax.set_yticks(range(s.shape[0]), [("Jamaica " + a.split(" (")[0]) if a in JM_ZONES else a for a in s.index],
                  fontsize=9)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Year-to-year similarity of intact mangrove (1 = no change)", loc="left", fontsize=11.5, color=INK)

    sw = np.flatnonzero(g.zone.values == JM_ZONES[0])
    m = d["cls"][sw] == M
    base = m[:, 4] & m[:, 5] & m[:, 6]
    lost24 = base & ~m[:, 7] & ~m[:, 8] & d["ok"][sw]
    lost25 = base & m[:, 7] & ~m[:, 8] & d["ok"][sw]
    order = np.argsort(-(lost24 | lost25).sum((1, 2)))[:n]
    bot = fig.add_gridspec(1, n, left=0.06, right=0.97, top=0.36, bottom=0.07, wspace=0.08)
    for k, j in enumerate(order):
        a = fig.add_subplot(bot[k])
        img = np.empty(m.shape[2:] + (3,))
        img[:] = mc.to_rgb(NODATA_C)
        img[d["ok"][sw[j]]] = mc.to_rgb(OTHER)
        img[base[j]] = mc.to_rgb(STABLE)
        img[lost24[j]] = mc.to_rgb("#8f1f1e")
        img[lost25[j]] = mc.to_rgb(LOSS)
        r = g.iloc[sw[j]]
        _tile(a, img, f"{r.lat:.3f}, {r.lon:.3f}",
              f"−{lost24[j].sum() * HA:.0f} ha in 2024 · −{lost25[j].sum() * HA:.0f} ha in 2025")
    fig.text(0.06, 0.41, "SW Jamaica, Black River area: tiles with the most mangrove mapped as lost "
             "(Melissa made landfall on this coast, at New Hope)", fontsize=11.5, color=INK)
    _legend(fig, [("mangrove 2021-23, still mapped 2025", STABLE), ("not mangrove since 2024", "#8f1f1e"),
                  ("not mangrove in 2025", LOSS), ("other land / water", OTHER)], y=0.005)
    fig.suptitle("Hurricanes: southwest Jamaica's mangroves changed sharply in 2024-25",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.935, "Right: embedding similarity of pixels mapped as mangrove every year. The two darkest cells "
             "are SW Jamaica 2024-25 (Beryl, Melissa) and the Everglades\n2017-18 (Hurricane Irma, Sept 2017). "
             "The 2025 data hold only ~2 months after Melissa, so its full effect will show in 2026. "
             "Lost pixels now map as\ncoastal forest or herbaceous wetland: damaged canopy, not clearing.",
             fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    _save(fig, out)
    plt.close(fig)


# ------------------------------------------------------------------ run
def run(bundle="models/change_logreg_k2.pt", temporal_dir="data/temporal", out_dir="results/change_report",
        device=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, years = compute(bundle, temporal_dir, device)
    st = stats(df, years, temporal_dir)
    for k in ("extent", "change", "annual", "loss_now", "loss_year", "skill", "calibration", "forecast_2026"):
        st[k].round(4).to_csv(out / f"{k}.csv", index=False)
    h = hotspots(df, years, temporal_dir)
    h.drop(columns=["chip_index", "patch", "row", "col"]).round(5).to_csv(out / "loss_hotspots.csv", index=False)
    write_points(h, out / "loss_hotspots.geojson")
    fig_health(st, out / "regions_compared.png")
    fig_extent(st, out / "extent_trend.png")
    fig_loss_context(st, out / "loss_causes.png")
    fig_hotspots(h, df, out / "loss_hotspots.png", temporal_dir)
    fig_skill(st, out / "forecast_skill.png")
    fig_forecast_maps(df, out / "forecast_maps.png", temporal_dir)
    fig_forecast_2026(st, out / "forecast_2026.png")
    ext, sim, g, d = hurricane_tables(df, years, temporal_dir)
    ext.round(3).to_csv(out / "hurricane_jamaica_extent.csv", index=False)
    sim.round(4).to_csv(out / "hurricane_similarity.csv", index=False)
    fig_hurricanes(ext, sim, g, d, out / "hurricanes.png")

    b = out / "baja"
    b.mkdir(parents=True, exist_ok=True)
    t, d = baja_tiles(df, temporal_dir)
    t.drop(columns=["x0", "y1"]).round(3).to_csv(b / "baja_tiles.csv", index=False)
    write_gis(t, d, b / "tiles")
    fig_baja_overview(t, st, b / "baja_overview.png")
    fig_baja_tiles(t, d, b / "baja_change_tiles.png")
    fig_baja_2026(t, d, b / "baja_2026.png")
    if (h.region == BAJA).any():
        fig_hotspots(h, df, b / "baja_loss_patches.png", temporal_dir, n=8, regions=[BAJA],
                     title="Baja California Sur: every confirmed mangrove loss patch, 2017-18 to 2024-25")
    return st, h
