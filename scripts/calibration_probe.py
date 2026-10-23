import sys, json, numpy as np, pandas as pd

from mangrove_embed import CLASS_NAMES
from mangrove_embed.pixel import make
from mangrove_embed.thresholds import tuned_metrics
from sklearn.metrics import average_precision_score
D = sys.argv[1] if len(sys.argv) > 1 else "data/pxsample"
R = ["everglades", "jamaica", "mekong", "sundarbans"]
rng = np.random.default_rng(0)
data = {}
for r in R:
    d = np.load(f"{D}/{r}.npz", allow_pickle=True)
    blk = (np.floor(d["lon"] / 0.1).astype(int) * 10000 + np.floor(d["lat"] / 0.1).astype(int))[d["chip"]]
    cnt = d["chip_cnt"]; s = d["chip_sum"]
    rand = d["kind"] == "random"
    data[r] = dict(x=d["x"].astype(np.float32), y=d["y"], blk=blk,
                   mu_all=s.sum(0) / cnt.sum(), mu_rand=s[rand].sum(0) / cnt[rand].sum())

def sub(n, m):
    i = np.flatnonzero(m); return rng.choice(i, min(n, len(i)), replace=False)

def em_prior(p, prior_tr, iters=100):
    """Saerens et al. 2002: re-estimate target class priors from unlabeled posteriors."""
    pr = prior_tr.copy()
    for _ in range(iters):
        q = p * (pr / prior_tr); q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pr).max() < 1e-6: break
        pr = new
    return q

def score(y, P, cl, vt=None, vp=None):
    k = cl.index(1); pred = np.asarray(cl)[P.argmax(1)]
    t, pm = y == 1, P[:, k]; f = pred == 1
    tp, fp, fn = (t & f).sum(), (~t & f).sum(), (t & ~f).sum()
    out = dict(iou=tp / (tp + fp + fn), prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1),
               ap=average_precision_score(t, pm))
    # oracle threshold (upper bound for any calibration fix)
    ths = np.linspace(0.02, 0.98, 49)
    out["iou_oracle"] = max(((t & (pm > h)).sum() / max((t | (pm > h)).sum(), 1)) for h in ths)
    fpcls = pd.Series(y[f & ~t]).map(lambda c: CLASS_NAMES[c - 1]).value_counts(normalize=True)
    out["fp_top"] = ", ".join(f"{c} {v:.0%}" for c, v in fpcls.head(3).items())
    out["fp_share_of_pred"] = fp / max(f.sum(), 1)
    return out

rows = []
for model in ["logreg", "mlp"]:
    for ho in R:
        tr = [r for r in R if r != ho]
        for variant in ["raw", "center_all", "center_rand", "center_rand_norm", "em_prior"]:
            def prep(r, x):
                if variant.startswith("center"):
                    mu = data[r]["mu_rand" if "rand" in variant else "mu_all"]
                    x = x - mu
                    if variant.endswith("norm"):
                        x = x / np.linalg.norm(x, axis=1, keepdims=True)
                return x
            Xs, Ys, Xv, Yv = [], [], [], []
            for r in tr:
                d = data[r]; ub = np.unique(d["blk"]); r2 = np.random.default_rng(1)
                vb = set(r2.choice(ub, max(1, round(0.15 * len(ub))), replace=False))
                isv = np.isin(d["blk"], list(vb))
                i = sub(30000, ~isv); j = sub(10000, isv)
                Xs.append(prep(r, d["x"][i])); Ys.append(d["y"][i])
                Xv.append(prep(r, d["x"][j])); Yv.append(d["y"][j])
            Xtr, ytr = np.concatenate(Xs), np.concatenate(Ys)
            Xva, yva = np.concatenate(Xv), np.concatenate(Yv)
            d = data[ho]; it = sub(100000, np.ones(len(d["y"]), bool))
            Xte, yte = prep(ho, d["x"][it]), d["y"][it]
            m = make(model, 0).fit(Xtr, ytr); cl = list(m.classes_)
            P = m.predict_proba(Xte)
            if variant == "em_prior":
                prior = np.array([(ytr == c).mean() for c in cl])
                P = em_prior(P, prior)
            res = score(yte, P, cl)
            rows.append(dict(model=model, holdout=ho, variant=variant, **res))
            print(model, ho, variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
pd.DataFrame(rows).to_csv("results/seg_calibration/calibration.csv", index=False)
