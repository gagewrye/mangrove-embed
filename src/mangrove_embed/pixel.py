"""
Per-pixel classifiers on chip pixels, evaluated exactly like train-seg so results sit in the same
compare-seg table (results/seg/pixel_<model>/).

For each held-out region (leave-one-region-out):
  train   pixels sampled from the other regions' chips (equal count per region, proportional to
          each chip's valid pixels), minus validation blocks
  val     pixels from held-out spatial blocks of the training regions -> mangrove threshold
  test    uniform pixel sample of the held-out region's chips (same population train-seg scores)

Models: knn (k=5 cosine, as in the v3 Code Editor script), logreg, mlp (128-64, sklearn).
Reports argmax metrics (11-class), AP, and IoU/F1 at the validation-tuned mangrove threshold.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASS_NAMES, MANGROVE
from .thresholds import tuned_metrics

log = logging.getLogger(__name__)
IGNORE = 255
CLASS_IDS = np.arange(1, len(CLASS_NAMES) + 1)


def make(name, seed=0):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
    if name == "logreg":
        return LogisticRegression(max_iter=3000)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=200,
            random_state=seed))
    raise ValueError(name)


def n_params(model):
    m = model[-1] if hasattr(model, "steps") else model
    if hasattr(m, "coefs_"):
        return int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))
    if hasattr(m, "coef_"):
        return int(m.coef_.size + m.intercept_.size)
    return None  # knn: non-parametric


# ------------------------------------------------------------------ data
def load_index(chips_dir, block_deg):
    rows = []
    for idx in sorted(Path(chips_dir).glob("*/index.csv")):
        df = pd.read_csv(idx)
        df["path"] = [str(idx.parent / f"{c}.npz") for c in df.chip]
        rows.append(df[[Path(p).exists() for p in df.path]])
    if not rows:
        raise SystemExit(f"No chips under {chips_dir}. Run `mangrove chips` first.")
    df = pd.concat(rows, ignore_index=True)
    bx = np.floor(df.lon / block_deg).astype(int).astype(str)
    by = np.floor(df.lat / block_deg).astype(int).astype(str)
    df["block"] = df.region + "_" + bx + "_" + by
    return df


def sample_pixels(paths, n_total, seed):
    """~n_total valid pixels from these chips, proportional to each chip's valid-pixel count
    (two passes; the label pass is cheap), so the sample is uniform over pixels."""
    rng = np.random.default_rng(seed)
    valid = np.array([(np.load(p)["y"] != IGNORE).sum() for p in paths], float)
    if valid.sum() == 0:
        return np.zeros((0, 64), np.float32), np.zeros(0, np.uint8)
    take = np.minimum(np.round(n_total * valid / valid.sum()).astype(int), valid.astype(int))
    X, Y = [], []
    for p, k in zip(paths, take):
        if k == 0:
            continue
        d = np.load(p)
        y = d["y"].reshape(-1)
        ok = np.flatnonzero(y != IGNORE)
        i = rng.choice(ok, k, replace=False)
        X.append(d["x"].reshape(d["x"].shape[0], -1)[:, i].T.astype(np.float32))
        Y.append(y[i])
    return np.concatenate(X), np.concatenate(Y)


def split_val_blocks(df, frac, seed):
    rng = np.random.default_rng(seed)
    val = np.zeros(len(df), bool)
    for r, g in df.groupby("region"):
        blocks = np.array(sorted(g.block.unique()), dtype=object)
        rng.shuffle(blocks)
        n = max(1, int(round(frac * len(blocks)))) if len(blocks) > 1 else 0
        val |= (df.region == r).values & df.block.isin(set(blocks[:n])).values
    return val


# ------------------------------------------------------------------ metrics
def metrics(y, proba, classes, val_t=None, val_p=None):
    classes = list(classes)
    pred = np.asarray(classes)[proba.argmax(1)]
    p_m = proba[:, classes.index(MANGROVE)] if MANGROVE in classes else np.zeros(len(y))
    cm = np.zeros((len(CLASS_IDS), len(CLASS_IDS)), np.int64)
    np.add.at(cm, (y.astype(int) - 1, pred.astype(int) - 1), 1)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    present = cm.sum(1) > 0
    k = MANGROVE - 1
    prec, rec = tp[k] / max(tp[k] + fp[k], 1), tp[k] / max(tp[k] + fn[k], 1)
    t = y == MANGROVE
    out = dict(mangrove_iou=float(iou[k]), mangrove_precision=float(prec), mangrove_recall=float(rec),
               mangrove_f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
               mangrove_ap=float(average_precision_score(t, p_m)) if t.any() else float("nan"),
               miou=float(iou[present].mean()), overall_acc=float(tp.sum() / max(cm.sum(), 1)),
               mangrove_px=int(t.sum()), valid_px=int(len(y)),
               per_class_iou={CLASS_NAMES[i]: float(iou[i]) for i in range(len(CLASS_IDS)) if present[i]})
    if val_t is not None:
        out.update(tuned_metrics(val_t, val_p, t, p_m))
    return out


# ------------------------------------------------------------------ run
def run(chips_dir="data/chips", out_dir="results/seg", models=("knn", "logreg", "mlp"),
        holdout="all", n_train_per_region=30_000, n_val=30_000, n_test=100_000,
        block_deg=0.1, val_frac=0.15, seed=0):
    df = load_index(chips_dir, block_deg)
    regions = sorted(df.region.unique())
    if len(regions) < 2:
        raise SystemExit("compare-pixel needs chips from at least 2 regions.")
    targets = regions if holdout == "all" else [holdout]
    rows = {m: [] for m in models}
    for r in targets:
        tr = df[df.region != r]
        val = split_val_blocks(tr, val_frac, seed)
        Xs, Ys = [], []
        for g in sorted(tr.region.unique()):
            sel = (tr.region == g).values & ~val
            X, Y = sample_pixels(list(tr.path[sel]), n_train_per_region, seed)
            Xs.append(X); Ys.append(Y)
        Xtr, ytr = np.concatenate(Xs), np.concatenate(Ys)
        Xva, yva = sample_pixels(list(tr.path[val]), n_val, seed + 1)
        Xte, yte = sample_pixels(list(df.path[df.region == r]), n_test, seed + 2)
        log.info("holdout %s: train %d px, val %d px, test %d px (mangrove %.2f)", r, len(ytr),
                 len(yva), len(yte), (yte == MANGROVE).mean())
        for name in models:
            t0 = time.time()
            m = make(name, seed).fit(Xtr, ytr)
            pv = m.predict_proba(Xva)
            cl = list(m.classes_)
            vt = yva == MANGROVE
            vp = pv[:, cl.index(MANGROVE)] if MANGROVE in cl else np.zeros(len(yva))
            res = metrics(yte, m.predict_proba(Xte), cl, vt, vp)
            res["fit_seconds"] = round(time.time() - t0, 1)
            run_dir = Path(out_dir) / f"pixel_{name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            info = dict(model=name, params=n_params(m), receptive_field_m=10,
                        train_px=int(len(ytr)), val_px=int(len(yva)), test_px=int(len(yte)))
            (run_dir / f"holdout_{r}.json").write_text(json.dumps(
                dict(tag=f"holdout_{r}", model_info=info, test={r: res}), indent=2))
            row = dict(protocol="loro", region=r,
                       **{k: v for k, v in res.items() if not isinstance(v, dict)})
            rows[name].append(row)
            print(f"[pixel_{name}] {r}: IoU {res['mangrove_iou']:.3f} F1 {res['mangrove_f1']:.3f} "
                  f"AP {res['mangrove_ap']:.3f} | tuned@{res['threshold']:.2f}: IoU "
                  f"{res['mangrove_iou_tuned']:.3f} ({res['fit_seconds']}s)")
    for name, rr in rows.items():
        path = Path(out_dir) / f"pixel_{name}" / "summary.csv"
        new = pd.DataFrame(rr)
        if path.exists() and holdout != "all":
            prev = pd.read_csv(path)
            new = pd.concat([prev[~prev.region.isin(new.region)], new], ignore_index=True)
        new.to_csv(path, index=False)
    return {n: pd.DataFrame(r) for n, r in rows.items()}
