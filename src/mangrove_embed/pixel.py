"""
Per-pixel classifiers on chip pixels, evaluated exactly like train-seg so results sit in the same
compare-seg table (results/model_comparison/pixel_<model>/).

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
from .wb import log_run, metric_summary

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
def run(chips_dir="data/chips", out_dir="results/model_comparison",
        models=("knn", "logreg", "mlp"),
        holdout="all", n_train_per_region=30_000, n_val=30_000, n_test=100_000,
        block_deg=0.1, val_frac=0.15, seed=0, regions=None, wandb=None):
    """wandb: None, or dict(project=, entity=) to log one W&B run per (model, held-out region)."""
    df = load_index(chips_dir, block_deg)
    if regions:
        df = df[df.region.isin(regions)].reset_index(drop=True)
    regions = sorted(df.region.unique())
    if len(regions) < 2:
        raise SystemExit("compare-pixel needs chips from at least 2 regions.")
    targets = regions if holdout == "all" else [holdout]
    rows = {m: [] for m in models}
    # seed 0 keeps the unsuffixed folder, so compare-seg counts it as one seed of the variant
    folder = {m: f"pixel_{m}" + (f"_s{seed}" if seed else "") for m in models}
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
            run_dir = Path(out_dir) / folder[name]
            run_dir.mkdir(parents=True, exist_ok=True)
            info = dict(model=name, params=n_params(m), receptive_field_m=10,
                        train_px=int(len(ytr)), val_px=int(len(yva)), test_px=int(len(yte)))
            (run_dir / f"holdout_{r}.json").write_text(json.dumps(
                dict(tag=f"holdout_{r}", model_info=info, test={r: res}), indent=2))
            row = dict(protocol="loro", region=r,
                       **{k: v for k, v in res.items() if not isinstance(v, dict)})
            rows[name].append(row)
            if wandb:
                summ = metric_summary({r: res})
                # region-agnostic keys too, as train-seg logs for a single held-out region
                summ.update({f"test/{k}": v for k, v in res.items() if not isinstance(v, dict)})
                log_run(wandb["project"], wandb.get("entity"), folder[name],
                        f"{folder[name]}/holdout_{r}", "loro",
                        dict(info, holdout=r, seed=seed, n_train_per_region=n_train_per_region,
                             block_deg=block_deg, val_frac=val_frac), summ, tags=["pixel", name])
            print(f"[{folder[name]}] {r}: IoU {res['mangrove_iou']:.3f} F1 {res['mangrove_f1']:.3f} "
                  f"AP {res['mangrove_ap']:.3f} | tuned@{res['threshold']:.2f}: IoU "
                  f"{res['mangrove_iou_tuned']:.3f} ({res['fit_seconds']}s)")
    for name, rr in rows.items():
        path = Path(out_dir) / folder[name] / "summary.csv"
        new = pd.DataFrame(rr)
        if path.exists() and holdout != "all":
            prev = pd.read_csv(path)
            new = pd.concat([prev[~prev.region.isin(new.region)], new], ignore_index=True)
        new.to_csv(path, index=False)
    return {n: pd.DataFrame(r) for n, r in rows.items()}


def run_external(chips_dir, test_chips_dir, out_dir="results/generalization",
                 models=("logreg", "mlp"), n_train_per_region=30_000, n_val=30_000,
                 n_test=100_000, block_deg=0.1, val_frac=0.15, seed=0, regions=None, wandb=None):
    """Train on every region in chips_dir (minus validation blocks), test on each region in
    test_chips_dir. Writes <out_dir>/<test regions>_pixel.csv, one row per (model, seed, test
    region); rerunning replaces matching rows, so seeds accumulate. wandb: None, or
    dict(project=, entity=) to log one W&B run per (model, seed)."""
    df = load_index(chips_dir, block_deg)
    if regions:
        df = df[df.region.isin(regions)].reset_index(drop=True)
    te = load_index(test_chips_dir, block_deg)
    train_regions = sorted(df.region.unique())
    val = split_val_blocks(df, val_frac, seed)
    Xs, Ys = [], []
    for g in train_regions:
        X, Y = sample_pixels(list(df.path[(df.region == g).values & ~val]), n_train_per_region, seed)
        Xs.append(X); Ys.append(Y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(Ys)
    Xva, yva = sample_pixels(list(df.path[val]), n_val, seed + 1)
    tests = {r: sample_pixels(list(g.path), n_test, seed + 2) for r, g in te.groupby("region")}
    log.info("train %d px from %d regions, val %d px, test %s", len(ytr), len(train_regions),
             len(yva), {r: len(t[1]) for r, t in tests.items()})
    rows = []
    for name in models:
        t0 = time.time()
        m = make(name, seed).fit(Xtr, ytr)
        cl = list(m.classes_)
        pv = m.predict_proba(Xva)
        vt = yva == MANGROVE
        vp = pv[:, cl.index(MANGROVE)] if MANGROVE in cl else np.zeros(len(yva))
        scored = {}
        for r, (Xte, yte) in tests.items():
            res = metrics(yte, m.predict_proba(Xte), cl, vt, vp)
            scored[r] = res
            rows.append(dict(run=f"pixel_{name}", split=f"all_{len(train_regions)}", model=name,
                             seed=seed, train_regions=" ".join(train_regions), test_region=r,
                             params=n_params(m), fit_seconds=round(time.time() - t0, 1),
                             **{k: v for k, v in res.items() if not isinstance(v, dict)}))
            print(f"[pixel_{name} s{seed}, {len(train_regions)} regions] -> {r}: IoU "
                  f"{res['mangrove_iou']:.3f} AP {res['mangrove_ap']:.3f} | tuned@"
                  f"{res['threshold']:.2f}: IoU {res['mangrove_iou_tuned']:.3f}")
        if wandb:
            split = f"all_{len(train_regions)}"
            log_run(wandb["project"], wandb.get("entity"), f"pixel_{name}_{split}",
                    f"pixel_{name} {split} s{seed}", "generalization",
                    dict(model=name, seed=seed, params=n_params(m), train_regions=train_regions,
                         train_px=int(len(ytr)), val_px=int(len(yva)),
                         n_train_per_region=n_train_per_region, block_deg=block_deg,
                         val_frac=val_frac, receptive_field_m=10),
                    metric_summary(scored), tags=["pixel", name, "generalization"])
    new = pd.DataFrame(rows)
    path = Path(out_dir) / f"{'_'.join(sorted(tests))}_pixel.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prev = pd.read_csv(path)
        key = ["run", "split", "seed", "test_region"]
        prev = prev[~prev.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([prev, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"-> {path}")
    return new


def run_pooled(chips_dir="data/chips", out_dir="results/model_comparison", models=("logreg",),
               n_train_per_region=30_000, n_val=30_000, n_test=100_000, seed=0, regions=None,
               tag=None, wandb=None):
    """Pooled protocol with exactly train-seg's chip splits (same block size, fractions and
    seeds), so scores line up with `train-seg --protocol pooled --seed N` region by region:
    test = 20% of each region's blocks, val = 10% of the remaining blocks (threshold tuning).
    Writes <out_dir>/pooled_pixel_<model>[_<tag>]_s<seed>/summary.csv for compare-seg."""
    from .seg.train import TrainConfig, add_blocks, load_index as seg_index, split_blocks
    tc = TrainConfig(seed=seed)
    df = add_blocks(seg_index(chips_dir, regions), tc.block_deg)
    test = split_blocks(df, 0.2, seed + 1)
    tr = df[~test]
    val = split_blocks(tr, tc.val_frac, seed)
    all_regions = sorted(df.region.unique())
    Xs, Ys = [], []
    for g in all_regions:
        X, Y = sample_pixels(list(tr.path[(tr.region == g).values & ~val]), n_train_per_region, seed)
        Xs.append(X); Ys.append(Y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(Ys)
    Xva, yva = sample_pixels(list(tr.path[val]), n_val, seed + 1)
    tests = {r: sample_pixels(list(df.path[test & (df.region == r).values]), n_test, seed + 2)
             for r in all_regions}
    tests = {r: t for r, t in tests.items() if len(t[1])}
    log.info("pooled: train %d px from %d regions, val %d px, test %d regions", len(ytr),
             len(all_regions), len(yva), len(tests))
    out = {}
    for name in models:
        t0 = time.time()
        m = make(name, seed).fit(Xtr, ytr)
        cl = list(m.classes_)
        pv = m.predict_proba(Xva)
        vt = yva == MANGROVE
        vp = pv[:, cl.index(MANGROVE)] if MANGROVE in cl else np.zeros(len(yva))
        folder = "_".join(x for x in ("pooled_pixel", name, tag, f"s{seed}") if x)
        run_dir = Path(out_dir) / folder
        run_dir.mkdir(parents=True, exist_ok=True)
        res, rows = {}, []
        for r, (Xte, yte) in tests.items():
            res[r] = metrics(yte, m.predict_proba(Xte), cl, vt, vp)
            rows.append(dict(protocol="pooled", region=r,
                             **{k: v for k, v in res[r].items() if not isinstance(v, dict)}))
            print(f"[{folder}] {r}: IoU {res[r]['mangrove_iou']:.3f} AP "
                  f"{res[r]['mangrove_ap']:.3f} | tuned@{res[r]['threshold']:.2f}: IoU "
                  f"{res[r]['mangrove_iou_tuned']:.3f}")
        info = dict(model=name, params=n_params(m), receptive_field_m=10, train_px=int(len(ytr)),
                    val_px=int(len(yva)), train_regions=all_regions,
                    fit_seconds=round(time.time() - t0, 1))
        (run_dir / "pooled.json").write_text(json.dumps(
            dict(tag="pooled", model_info=info, test=res), indent=2))
        summary = pd.DataFrame(rows)
        summary.to_csv(run_dir / "summary.csv", index=False)
        print(f"[{folder}] mean IoU {summary.mangrove_iou.mean():.3f} -> {run_dir}")
        if wandb:
            log_run(wandb["project"], wandb.get("entity"), folder, f"{folder}/pooled", "pooled",
                    dict(info, seed=seed, n_train_per_region=n_train_per_region),
                    metric_summary(res), tags=["pixel", name, "pooled"])
        out[name] = summary
    return out
