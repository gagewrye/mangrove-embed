"""Summary figures for the results folders.

  model_comparison/model_comparison.png  held-out mangrove IoU per model (mean over regions,
                                         one dot per seed) and per region
  receptive_field/receptive_field.png    FCN receptive-field sweep (Mekong) and 10 m vs 630 m
                                         on all held-out regions (rf_runs.csv, from W&B)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reference palette (dataviz skill), light mode
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, BLUE_DARK = "#2a78d6", "#eb6834", "#184f95"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# run folder (seed suffix stripped) -> (label, group); unknown folders are labeled by name
VARIANTS = {
    "pixel_mlp": ("Pixel MLP 128-64 (sklearn)", "per-pixel, 10 m"),
    "loro_fcn": ("FCN, 630 m context", "spatial context"),
    "loro_fcn_pc": ("FCN, 630 m context", "spatial context"),  # seed-42 rerun of loro_fcn
    "loro_fcn_rf1": ("FCN, no context (10 m)", "per-pixel, 10 m"),
    "loro": ("SegFormer-B0", "spatial context"),
    "pixel_logreg": ("Logistic regression", "per-pixel, 10 m"),
    "pixel_knn": ("kNN (k=5, cosine)", "per-pixel, 10 m"),
    "loro_mlp": ("Chip MLP 64-64-11 (4.9k params)", "per-pixel, 10 m"),
}
REGIONS = ["everglades", "jamaica", "mekong", "sundarbans"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load_runs(model_dir):
    rows = []
    for f in sorted(Path(model_dir).glob("*/summary.csv")):
        s = pd.read_csv(f)
        s = s[s.protocol == "loro"] if "protocol" in s else s
        s["run"] = f.parent.name
        rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    df["variant"] = df.run.str.replace(r"_s\d+$", "", regex=True)
    df["label"] = [VARIANTS.get(v, (v, ""))[0] for v in df.variant]
    return df


def model_comparison(model_dir, out):
    df = load_runs(model_dir)
    df = df[df.region.isin(REGIONS)]
    # only runs that cover every region, so the means are like-for-like
    full = df.groupby("run").region.nunique()
    df = df[df.run.isin(full[full == len(REGIONS)].index)]
    per_run = df.groupby(["label", "run"]).mangrove_iou.mean().reset_index()
    order = per_run.groupby("label").mangrove_iou.mean().sort_values().index
    per_region = df.groupby(["label", "region"]).mangrove_iou.mean().unstack()[REGIONS]
    per_region = per_region.loc[order[::-1]]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 0.55 * len(order) + 2.2),
                               gridspec_kw=dict(width_ratios=[1.15, 1]), facecolor=SURFACE)
    _style(a)
    a.grid(axis="y", visible=False)
    means = per_run.groupby("label").mangrove_iou.mean().loc[order]
    y = np.arange(len(order))
    a.barh(y, means.values, height=0.55, color=BLUE, alpha=0.35, edgecolor=SURFACE, lw=2)
    for i, lab in enumerate(order):
        runs = per_run[per_run.label == lab].mangrove_iou.values
        a.scatter(runs, np.full(len(runs), i), s=34, color=BLUE_DARK, edgecolor=SURFACE,
                  lw=1.2, zorder=3)
        n = f"  {len(runs)} runs" if len(runs) > 1 else ""
        sd = f", sd {runs.std(ddof=1):.3f}" if len(runs) > 1 else ""
        a.text(max(runs.max(), means[lab]) + 0.006, i, f"{means[lab]:.3f}{n}{sd}",
               va="center", fontsize=9, color=INK)
    a.set_yticks(y, order, color=INK, fontsize=10)
    lo = np.floor((per_run.mangrove_iou.min() - 0.03) * 20) / 20
    a.set_xlim(lo, 0.93)
    a.set_xlabel("held-out mangrove IoU (mean of 4 regions)", color=INK2, fontsize=9)
    a.set_title("Model comparison, leave-one-region-out\nbar = mean, dot = one run (seed)",
                loc="left", color=INK, fontsize=11)

    vals = per_region.values
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    b.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            t = (vals[i, j] - vmin) / (vmax - vmin)
            b.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=9,
                   color="white" if t > 0.55 else INK)
    b.set_xticks(range(len(REGIONS)), [r.capitalize() for r in REGIONS], color=INK, fontsize=10)
    b.set_yticks(range(len(per_region)), per_region.index, color=INK, fontsize=10)
    b.tick_params(length=0)
    for s in b.spines.values():
        s.set_visible(False)
    b.set_xticks(np.arange(-0.5, len(REGIONS)), minor=True)
    b.set_yticks(np.arange(-0.5, len(per_region)), minor=True)
    b.grid(which="minor", color=SURFACE, lw=2)
    b.tick_params(which="minor", length=0)
    b.set_title("By held-out region (mean over seeds)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def receptive_field(rf_csv, model_dir, out):
    rf = pd.read_csv(rf_csv)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)

    sw = rf[rf.group == "rf-sweep"].sort_values("rf_m")
    _style(a)
    a.plot(sw.rf_m, sw.mangrove_iou, color=BLUE, lw=2, marker="o", ms=8,
           markeredgecolor=SURFACE, markeredgewidth=1.5)
    for x, v in zip(sw.rf_m, sw.mangrove_iou):
        a.annotate(f"{v:.3f}", (x, v), xytext=(0, 9), textcoords="offset points",
                   ha="center", fontsize=9, color=INK)
    a.set_xscale("log")
    a.set_xticks(sw.rf_m, [f"{m:g} m" if m < 1000 else f"{m / 1000:g} km" for m in sw.rf_m])
    a.minorticks_off()
    a.set_ylim(0.70, 0.86)
    a.set_xlabel("FCN receptive field (fixed depth and ~190k params)", color=INK2, fontsize=9)
    a.set_ylabel("Mekong held-out mangrove IoU", color=INK2, fontsize=9)
    n = int(sw.train_chips.iloc[0] + sw.val_chips.iloc[0])
    a.set_title(f"Sweep: Mekong held out, trained on 9 other regions ({n} chips), 1 seed",
                loc="left", color=INK, fontsize=11)
    # seed-to-seed spread of the same 630 m FCN on Mekong (4-region setup), for scale
    seeds = _fcn630_mekong(model_dir, rf)
    if len(seeds) > 1:
        a.text(0.02, 0.04, f"For scale: 630 m FCN on Mekong across {len(seeds)} seeds "
               f"(4-region setup) ranges {seeds.min():.3f}-{seeds.max():.3f}",
               transform=a.transAxes, fontsize=8.5, color=INK2)

    # 10 m vs 630 m, same splits and seed (42), all 4 held-out regions
    _style(b)
    b.grid(axis="x", visible=False)
    p10 = rf[rf.group == "loro_fcn_rf1_all"].set_index("holdout").mangrove_iou
    p630 = rf[rf.group == "loro_fcn"].set_index("holdout").mangrove_iou
    other = _fcn630_runs(model_dir)
    x = np.arange(len(REGIONS))
    for i, r in enumerate(REGIONS):
        b.plot([i, i], [p630[r], p10[r]], color=GRID, lw=2, zorder=1)
        if r in other:
            b.scatter(np.full(len(other[r]), i + 0.18), other[r], s=22, facecolor="none",
                      edgecolor=BLUE, lw=1.2, zorder=2)
    b.scatter(x, p630[REGIONS], s=70, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3,
              label=f"630 m (seed 42), mean {p630[REGIONS].mean():.3f}")
    b.scatter(x, p10[REGIONS], s=70, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=3,
              label=f"10 m, no context (seed 42), mean {p10[REGIONS].mean():.3f}")
    if other:
        m = np.mean([other[r].mean() for r in REGIONS if r in other])
        n = min(len(v) for v in other.values())
        b.scatter([], [], s=22, facecolor="none", edgecolor=BLUE, lw=1.2,
                  label=f"630 m, {n} other runs (seeds 0-2, seed-42 rerun), mean {m:.3f}")
    b.set_xticks(x, [r.capitalize() for r in REGIONS], color=INK, fontsize=10)
    b.set_ylabel("held-out mangrove IoU", color=INK2, fontsize=9)
    b.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    b.set_ylim(0.6, 1.0)
    b.set_title("10 m vs 630 m, all 4 held-out regions, same splits", loc="left", color=INK,
                fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _fcn630_runs(model_dir):
    """Held-out IoU per region for every local default-FCN run (loro_fcn*, no rf suffix)."""
    df = load_runs(model_dir)
    df = df[df.variant.isin(["loro_fcn", "loro_fcn_pc"])]
    return {r: g.mangrove_iou.values for r, g in df.groupby("region")}


def _fcn630_mekong(model_dir, rf):
    local = _fcn630_runs(model_dir).get("mekong", np.array([]))
    wb = rf[(rf.group == "loro_fcn") & (rf.holdout == "mekong")].mangrove_iou.values
    return np.concatenate([local, wb])


def run(results_dir="results"):
    r = Path(results_dir)
    outs = []
    md = r / "model_comparison"
    if any(md.glob("*/summary.csv")):
        model_comparison(md, md / "model_comparison.png")
        outs.append(md / "model_comparison.png")
    rf = r / "receptive_field" / "rf_runs.csv"
    if rf.exists():
        receptive_field(rf, md, rf.parent / "receptive_field.png")
        outs.append(rf.parent / "receptive_field.png")
    for o in outs:
        print(f"wrote {o}")
    return outs
