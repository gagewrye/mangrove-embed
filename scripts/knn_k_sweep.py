"""kNN k sweep, leave-one-region-out, on the pixel sample from scripts/sample_pixels.py.
Neighbours are found once at max k (cosine); every smaller k reuses them.

usage: python scripts/knn_k_sweep.py [data/pxsample]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

D = sys.argv[1] if len(sys.argv) > 1 else "data/pxsample"
R = ["everglades", "jamaica", "mekong", "sundarbans"]
KS = [1, 3, 5, 10, 20, 50, 100, 200]
N_CLASSES = 11
rng = np.random.default_rng(0)
data = {r: np.load(f"{D}/{r}.npz", allow_pickle=True) for r in R}


def sub(n, size):
    return rng.choice(size, min(n, size), replace=False)


def scores(y, p_m, pred):
    t, f = y == 1, pred == 1
    tp, fp, fn = (t & f).sum(), (~t & f).sum(), (t & ~f).sum()
    ths = np.linspace(0.02, 0.98, 49)
    return dict(iou=tp / (tp + fp + fn), prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1),
                ap=average_precision_score(t, p_m),
                iou_oracle=max((t & (p_m > h)).sum() / max((t | (p_m > h)).sum(), 1) for h in ths))


rows = []
for ho in R:
    Xs, Ys = [], []
    for r in R:
        if r != ho:
            i = sub(30000, len(data[r]["y"]))
            Xs.append(data[r]["x"][i].astype(np.float32)); Ys.append(data[r]["y"][i])
    Xtr, ytr = np.concatenate(Xs), np.concatenate(Ys)
    it = sub(100000, len(data[ho]["y"]))
    Xte, yte = data[ho]["x"][it].astype(np.float32), data[ho]["y"][it]
    nn = NearestNeighbors(n_neighbors=max(KS), metric="cosine", n_jobs=-1).fit(Xtr)
    dist, ind = nn.kneighbors(Xte)
    lab = ytr[ind].astype(int) - 1                       # [n, kmax] class index
    for weight in ["uniform", "distance"]:
        w = np.ones_like(dist) if weight == "uniform" else 1 / np.maximum(dist, 1e-6)
        for k in KS:
            votes = np.zeros((len(yte), N_CLASSES))
            np.add.at(votes, (np.arange(len(yte))[:, None], lab[:, :k]), w[:, :k])
            votes /= votes.sum(1, keepdims=True)
            res = scores(yte, votes[:, 0], votes.argmax(1) + 1)
            rows.append(dict(holdout=ho, weight=weight, k=k, **res))
            print(ho, weight, k, {a: round(float(b), 3) for a, b in res.items()}, flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/seg_calibration/knn_k_sweep.csv", index=False)
for m in ["iou", "ap", "iou_oracle"]:
    t = df.pivot_table(index=["weight", "k"], columns="holdout", values=m)
    t["mean"] = t.mean(axis=1)
    print(f"\n{m}\n", t.round(3).to_string())
