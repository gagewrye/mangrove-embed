"""Sample a fraction of valid pixels per chip, plus per-chip embedding sums (for region means).
Feeds scripts/calibration_probe.py.

usage: python scripts/sample_pixels.py data/chips data/pxsample <region> 0.04
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root, out, r, p = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
idx = pd.read_csv(root / r / "index.csv")
X, Y, C, sums, cnts = [], [], [], [], []
t0 = time.time()
for i, row in idx.iterrows():
    f = root / r / f"{row.chip}.npz"
    if not f.exists():
        sums.append(np.zeros(64)); cnts.append(0); continue
    d = np.load(f)
    x, y = d["x"].reshape(64, -1), d["y"].reshape(-1)
    emb = np.abs(x).sum(0) > 0
    sums.append(x[:, emb].astype(np.float64).sum(1)); cnts.append(int(emb.sum()))
    ok = np.flatnonzero((y != 255) & emb)
    j = rng.choice(ok, rng.binomial(len(ok), p), replace=False)
    X.append(x[:, j].T); Y.append(y[j]); C.append(np.full(len(j), i, np.int32))
np.savez(out / f"{r}.npz", x=np.concatenate(X).astype(np.float16), y=np.concatenate(Y),
         chip=np.concatenate(C), chip_sum=np.array(sums), chip_cnt=np.array(cnts),
         lon=idx.lon.values, lat=idx.lat.values, kind=idx.kind.values.astype(str))
print(r, len(np.concatenate(Y)), round(time.time() - t0, 1), "s")
