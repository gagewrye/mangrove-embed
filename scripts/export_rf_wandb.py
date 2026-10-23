"""Export the FCN receptive-field runs from W&B to results/receptive_field/rf_runs.csv.
These runs exist only in W&B (trained on another machine), so this keeps a versioned copy.
One row per (run, held-out region); `train_chips` shows which training set a run used
(~800 = the 3 other original regions, ~2400 = 9 other regions).

usage: python scripts/export_rf_wandb.py [entity/project]
"""
import sys
from pathlib import Path

import pandas as pd
import wandb

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "gagepersonal/mangrove-embed"
# rf-sweep: 10 m - 1.27 km on Mekong; loro_fcn_rf1_all: 10 m on all 4; loro_fcn: 630 m on all 4
GROUPS = ["rf-sweep", "loro_fcn_rf1_all", "loro_fcn"]
METRICS = ["mangrove_iou", "mangrove_f1", "mangrove_ap", "miou"]

rows = []
for r in wandb.Api(timeout=60).runs(PROJECT, filters={"group": {"$in": GROUPS}}):
    if r.state != "finished":
        continue
    c, s = r.config, r.summary
    for region in c.get("test_regions") or []:
        rows.append(dict(group=r.group, run=r.name, run_id=r.id, holdout=region,
                         rf_m=c.get("receptive_field_m") or 630,  # default dilations: not logged
                         fcn_dilations="-".join(map(str, c.get("fcn_dilations") or (1, 2, 4, 8, 16))),
                         fcn_width=c.get("fcn_width"), params=c.get("params"), seed=c.get("seed"),
                         train_chips=c.get("train_chips"), val_chips=c.get("val_chips"),
                         **{m: s.get(f"test/{region}/{m}") for m in METRICS},
                         created=r.created_at))

df = pd.DataFrame(rows).sort_values(["group", "rf_m", "holdout"])
out = Path("results/receptive_field/rf_runs.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(df.drop(columns=["run_id", "created"]).round(3).to_string(index=False))
