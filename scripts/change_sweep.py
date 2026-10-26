"""Training variants for the change forecasters: loss, change weighting, target masking.

Logistic-regression maps as targets, 2 prior years, tidal flat and water merged (the robustness
setup), forecast years 2023-2025 x seeds 0-2. Every variant trains next_class and
next_class_hazard; classify_current (persistence ranked by classifier uncertainty) is the
baseline to beat.

  base          CE                                      (this sweep's reference)
  cw5           CE, changing pairs weigh 5
  jaccard       0.5 CE + 0.5 (1 - batch-pooled soft IoU over classes + change)
  conf          drop training pairs with classifier top probability < --conf in either year
  noflicker     drop training pairs whose change reverts the next year
  clean         conf + noflicker
  jaccard_clean jaccard + conf + noflicker

Runs are tagged "sweep" (change_logreg_k2_y<year>[_<variant>]_sweep_s<seed>), so they never
overwrite the benchmark runs. Resumes: runs whose change.json exists are skipped.
Writes results/change/variants.csv (one row per run x method) and prints mean / std over runs
and the paired difference to base (same seed and forecast year = same split and test pixels).

usage: python scripts/change_sweep.py [--seeds 0 1 2] [--years 2023 2024 2025] [--conf 0.6]
                                      [--variants base jaccard ...]
"""
import argparse
from pathlib import Path

import pandas as pd

from mangrove_embed.change import ChangeConfig, variant
from mangrove_embed.cli import main

OUT = Path("results/change")
METHODS = ["classify_current", "next_class", "next_class_hazard"]
METRICS = ["change_ap", "durable_ap", "durable_gain_ap", "durable_loss_ap", "notide_ap", "mangrove_iou"]


def variants(conf):
    """name -> ChangeConfig fields."""
    clean = dict(mask_uncertain=conf, mask_flicker=True)
    return {"base": {}, "cw5": dict(change_weight=5.0), "jaccard": dict(loss="jaccard"),
            "conf": dict(mask_uncertain=conf), "noflicker": dict(mask_flicker=True),
            "clean": clean, "jaccard_clean": dict(loss="jaccard", **clean)}


def flags(kw):
    """ChangeConfig fields -> `mangrove change` flags."""
    out = []
    for k, v in kw.items():
        out += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return out


def folder(kw, year, seed):
    """Run folder `mangrove change` writes for this variant (same naming as change.run)."""
    v = variant(ChangeConfig(**kw))
    return OUT / "_".join(p for p in ("change_logreg_k2", f"y{year}", v, "sweep", f"s{seed}") if p)


def sweep():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--years", nargs="*", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--conf", type=float, default=0.6, help="mask_uncertain threshold")
    ap.add_argument("--variants", nargs="*", help="default: all")
    ap.add_argument("--epochs", type=int, help="override (default: mangrove change default)")
    a = ap.parse_args()
    todo = {k: v for k, v in variants(a.conf).items() if not a.variants or k in a.variants}

    rows = []
    for name, kw in todo.items():
        for year in a.years:
            for seed in a.seeds:
                d = folder(kw, year, seed)
                if not (d / "change.json").exists():
                    argv = ["change", "--labels", "logreg", "--history", "2", "--test-year", str(year),
                            "--seed", str(seed), "--tag", "sweep", "--methods", *METHODS, *flags(kw)]
                    if a.epochs:
                        argv += ["--epochs", str(a.epochs)]
                    print(">>", " ".join(argv), flush=True)
                    main(argv)
                else:
                    print(f"skip {d.name} (done)")
                s = pd.read_csv(d / "summary.csv")
                s = s[s.region == "all"]
                rows += [dict(variant=name, test_year=year, seed=seed, method=r.method,
                              **{m: getattr(r, m, float("nan")) for m in METRICS}) for r in s.itertuples()]

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "variants.csv", index=False)
    cols = [m for m in METRICS if df[m].notna().any()]
    print("\nmean / std over seeds x forecast years:\n",
          df.groupby(["variant", "method"])[cols].agg(["mean", "std"]).round(3).to_string())
    if "base" in todo:  # paired: same seed and year as base
        key = ["test_year", "seed", "method"]
        b = df[df.variant == "base"].set_index(key)[cols]
        d = df[df.variant != "base"].set_index(key)
        delta = (d[cols] - b.reindex(d.index)).assign(variant=d.variant).reset_index()
        print("\nminus base, paired by seed and forecast year (mean, min, max):\n",
              delta[delta.method != "classify_current"].groupby(["variant", "method"])[cols]
              .agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    sweep()
