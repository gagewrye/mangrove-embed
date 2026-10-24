"""Spatial-context test for the regularized per-pixel MLP.

Same MLP (128-64, BatchNorm, dropout 0.1), same pooled 10-region splits as the *_no_mexpac runs
(mexico_pacific held out), seeds 0-2. Each variant concatenates masked mean-pooled neighbourhoods
of radius r to every pixel's embedding (--mlp-context). Then scores every run on Baja north.

  ctx ()  -> existing pooled_mlp_h128-64_bn_do0.1_no_mexpac_s*   (per-pixel baseline, 10 m)
  ctx 1   -> 3x3    (30 m)      ctx 3 -> 7x7 (70 m)     ctx 7 -> 15x15 (150 m)
  ctx 15  -> 31x31  (310 m)     ctx 1,3,7 -> multi-scale

Resumes: runs whose pooled.json already exists are skipped.

usage: python scripts/context_sweep.py [--seeds 0 1 2] [--contexts 1 3 7 15 1,3,7] [--wandb]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mangrove_embed.cli import main

REGIONS = ["everglades", "jamaica", "mahakam", "matang", "mekong", "niger_delta", "papua",
           "para", "rufiji", "sundarbans"]
OUT = Path("results/model_comparison")
BASE = "pooled_mlp_h128-64_bn_do0.1"

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
ap.add_argument("--contexts", nargs="*", default=["1", "3", "7", "15", "1,3,7"])
ap.add_argument("--test-chips", default="data/test/chips")
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--epochs", type=int, help="override (default: train-seg default)")
a = ap.parse_args()


def folder(ctx, seed):
    c = f"_ctx{ctx.replace(',', '-')}" if ctx else ""
    return f"{BASE}{c}_no_mexpac_s{seed}"


runs = []
for ctx in [""] + a.contexts:
    for seed in a.seeds:
        tag = folder(ctx, seed)
        runs.append((ctx, seed, tag))
        if (OUT / tag / "pooled.json").exists():
            print(f"skip {tag} (done)")
            continue
        argv = ["train-seg", "--protocol", "pooled", "--model", "mlp", "--mlp-hidden", "128,64",
                "--mlp-norm", "--mlp-dropout", "0.1", "--seed", str(seed), "--regions", *REGIONS,
                "--tag", tag]
        if ctx:
            argv += ["--mlp-context", ctx]
        if a.epochs:
            argv += ["--epochs", str(a.epochs)]
        if a.wandb:
            argv += ["--wandb", "--wandb-name", f"MLP ctx{ctx or 0} s{seed}",
                     "--wandb-group", "mlp-context"]
        print(">>", " ".join(argv), flush=True)
        main(argv)

# unseen region: score every run's checkpoint on Baja north
if Path(a.test_chips).exists():
    main(["eval-seg", "--chips", a.test_chips, "--runs", *[str(OUT / t) for _, _, t in runs],
          "--out", "results/generalization/context_baja_north.csv"])

rows = []
for ctx, seed, tag in runs:
    s = pd.read_csv(OUT / tag / "summary.csv")
    info = json.loads((OUT / tag / "pooled.json").read_text())["model_info"]
    rows.append(dict(context=ctx or "none", rf_m=info.get("receptive_field_m", 10), seed=seed,
                     params=info["params"], pooled_iou=s.mangrove_iou.mean(),
                     pooled_ap=s.mangrove_ap.mean(), jamaica=s.set_index("region").mangrove_iou.get("jamaica")))
df = pd.DataFrame(rows)
g = Path("results/generalization/context_baja_north.csv")
if g.exists():
    b = pd.read_csv(g)
    b["seed"] = b.run.str.extract(r"_s(\d+)$").astype(int)
    b["context"] = b.run.str.extract(r"_ctx([\d-]+)_")[0].str.replace("-", ",").fillna("none")
    df = df.merge(b[["context", "seed", "mangrove_iou", "mangrove_ap"]]
                  .rename(columns={"mangrove_iou": "baja_iou", "mangrove_ap": "baja_ap"}),
                  on=["context", "seed"], how="left")
Path("results/context").mkdir(parents=True, exist_ok=True)
df.to_csv("results/context/mlp_context.csv", index=False)
cols = [c for c in ["pooled_iou", "pooled_ap", "jamaica", "baja_iou", "baja_ap"] if c in df]
agg = df.groupby(["context", "rf_m", "params"])[cols].agg(["mean", "std"]).sort_index(level="rf_m")
print("\nper seed:\n", df.round(3).to_string(index=False))
print("\nmean / std over seeds:\n", agg.round(3).to_string())
# paired vs per-pixel baseline (same seed = same split)
base = df[df.context == "none"].set_index("seed")
for c in cols:
    d = df[df.context != "none"].assign(delta=lambda x: x[c] - x.seed.map(base[c]))
    print(f"\n{c} minus per-pixel, paired by seed:\n",
          d.pivot_table(index="context", columns="seed", values="delta").round(3).to_string())
