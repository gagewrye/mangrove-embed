"""
Cross-region diagnostics for mangrove classification on AlphaEarth embeddings.

Models are trained on all 11 classes (as in the v3 kNN) and scored on mangrove
(precision / recall / F1 / IoU from argmax, AP from p(mangrove)) plus overall accuracy.

  Q1 Geometry  - Is a region's mangrove centroid closer to other regions' mangroves than to its
                 most confusable local class?
  Q2 Model     - kNN vs logistic regression vs MLP under leave-one-region-out (LORO).
                 Also: which true classes become false "mangrove" calls.
  Q3 Shift     - Domain gap = within-region spatially blocked CV minus LORO;
                 single-anchor training (e.g. Jamaica only) -> every other region.
  Q4 Spatial   - Pixel vs 3x3 / 7x7 / 15x15 mean-pooled context, within-region and LORO.
                 Q4-edge repeats it on boundary pixels (*_edge.parquet, test-only), where the
                 two-stage training sample is thin and context should matter most.
  +  Centering - Subtract a label-free per-region mean (coastal-zone pixels) before LORO.

Run via:  mangrove diagnose [--models knn logreg mlp] [--skip-within]
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import BANDS, CLASS_NAMES, MANGROVE

log = logging.getLogger(__name__)

FEATURE_SETS = {
    "pixel":       [0],
    "ctx3":        [1],
    "ctx7":        [3],
    "ctx15":       [7],
    "pixel+ctx7":  [0, 3],
    "pixel+multi": [0, 1, 3, 7],
}
NAME = dict(enumerate(CLASS_NAMES, start=1))
METRIC = "mangrove_f1"


# ---------------------------------------------------------------- data
def _read(files):
    return pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                      for f in files], ignore_index=True) if files else None


def load(data_dir, block_deg=0.05):
    d = Path(data_dir)
    lab = _read(sorted(d.glob("*_labeled.parquet")) or sorted(d.glob("*_labeled.csv")))
    if lab is None:
        raise SystemExit(f"No *_labeled.parquet/csv in {d}. Run `mangrove extract` first.")
    unl = _read(sorted(d.glob("*_unlabeled.parquet")) or sorted(d.glob("*_unlabeled.csv")))
    edge = _read(sorted(d.glob("*_edge.parquet")))
    lab = _prep(lab, block_deg)
    if edge is not None and len(edge):
        edge = _prep(edge, block_deg)
    else:
        edge = None
    return lab, unl, edge


def _prep(df, block_deg):
    df = df.dropna().reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    bx = np.floor(df["longitude"].values / block_deg).astype(int)
    by = np.floor(df["latitude"].values / block_deg).astype(int)
    df["block"] = df["region"] + "_" + pd.Series(bx).astype(str) + "_" + pd.Series(by).astype(str)
    return df


def unit(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)


def present_scales(df):
    return {int(c.split("_")[0][1:]) for c in df.columns if c.startswith("s") and c.endswith("_A00")}


def features(df, fset, centers=None):
    """Concat per-scale blocks, each re-normalized to unit length (means of unit vectors aren't)."""
    blocks = []
    for s in FEATURE_SETS[fset]:
        v = df[[f"s{s}_{b}" for b in BANDS]].values.astype(np.float32)
        if centers is not None:
            v = v - np.stack([centers[(r, s)] for r in df["region"].values])
        blocks.append(unit(v))
    return np.hstack(blocks) / np.sqrt(len(blocks))


def region_centers(unl, lab):
    """Label-free per-region mean from random coastal-zone pixels (not the class-balanced sample)."""
    src = unl if unl is not None else lab
    if unl is None:
        log.warning("no unlabeled tables; centering on balanced sample leaks class proportions")
    return {(r, s): g[[f"s{s}_{b}" for b in BANDS]].values.mean(0)
            for r, g in src.groupby("region") for s in present_scales(src)}


# ---------------------------------------------------------------- models
def make_model(name):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)  # matches v3 JS k
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=3000)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True,
            max_iter=300, random_state=0))
    raise ValueError(name)


def score(y, pred, p_m):
    t, p = y == MANGROVE, pred == MANGROVE
    tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(overall_acc=accuracy_score(y, pred),
                mangrove_precision=prec, mangrove_recall=rec,
                mangrove_f1=2 * prec * rec / max(prec + rec, 1e-9),
                mangrove_iou=tp / max(tp + fp + fn, 1),
                mangrove_ap=average_precision_score(t, p_m) if t.any() else np.nan)


def fit_predict(model, Xtr, ytr, Xte):
    return predict(make_model(model).fit(Xtr, ytr), Xte)


def predict(m, Xte):
    proba = m.predict_proba(Xte)
    classes = list(m.classes_)
    pred = np.asarray(classes)[proba.argmax(1)]
    p_m = proba[:, classes.index(MANGROVE)] if MANGROVE in classes else np.zeros(len(Xte))
    return pred, p_m


# ---------------------------------------------------------------- Q1 geometry
def geometry(lab, out):
    regions = sorted(lab.region.unique())
    X = features(lab, "pixel")
    cent = {}
    for (r, c), idx in lab.groupby(["region", "class"]).groups.items():
        cent[(r, c)] = unit(X[idx].mean(0, keepdims=True))[0]

    rows = []
    for r in regions:
        if (r, MANGROVE) not in cent:
            continue
        m = cent[(r, MANGROVE)]
        others = [cent[(o, MANGROVE)] for o in regions if o != r and (o, MANGROVE) in cent]
        m_other = unit(np.mean(others, 0, keepdims=True))[0]
        local = {c: float(m @ v) for (rr, c), v in cent.items() if rr == r and c != MANGROVE}
        nearest = max(local, key=local.get)
        rows.append(dict(region=r, cos_to_other_regions_mangrove=float(m @ m_other),
                         nearest_local_class=NAME[nearest], cos_to_nearest_local=local[nearest],
                         margin=float(m @ m_other) - local[nearest]))
    g = pd.DataFrame(rows)
    g.to_csv(out / "q1_geometry.csv", index=False)

    # region x region mangrove-centroid similarity
    rs = [r for r in regions if (r, MANGROVE) in cent]
    M = np.stack([cent[(r, MANGROVE)] for r in rs])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M @ M.T, cmap="viridis")
    ax.set_xticks(range(len(rs)), rs, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rs)), rs, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Mangrove centroid cosine across regions (pixel)")
    fig.tight_layout(); fig.savefig(out / "q1_mangrove_centroids.png", dpi=150); plt.close(fig)

    reg_acc = LogisticRegression(max_iter=3000).fit(X, lab.region).score(X, lab.region)
    print("\n=== Q1 geometry (pixel) ===")
    print(g.round(3).to_string(index=False))
    print(f"region identifiable from embedding (train acc, {len(regions)} regions): {reg_acc:.3f}")
    print("margin > 0: a region's mangroves sit closer to other regions' mangroves than to their "
          "most confusable local class.")
    return g


# ---------------------------------------------------------------- protocols
def run_protocols(lab, models, fsets, centers, skip_within, anchor, edge=None):
    """edge: boundary test set (never trained on). Adds loro_edge / within_edge protocols."""
    regions = sorted(lab.region.unique())
    y = lab["class"].values
    ye = edge["class"].values if edge is not None else None
    rows, false_src = [], []
    variants = [(fs, False) for fs in fsets] + ([("pixel", True)] if centers else [])

    def edge_eval(model, mask, tag, protocol, r):
        if edge is None or not mask.any() or not (ye[mask] == MANGROVE).any():
            return None
        pred, p_m = predict(model, Xe[mask])
        return dict(**tag, protocol=protocol, region=r, **score(ye[mask], pred, p_m))

    for fs, centered in variants:
        X = features(lab, fs, centers if centered else None)
        use_edge = edge is not None and not centered
        Xe = features(edge, fs) if use_edge else None
        for mdl in models:
            tag = dict(features=fs, model=mdl, centered=centered)
            for r in regions:  # LORO
                te = (lab.region == r).values
                model = make_model(mdl).fit(X[~te], y[~te])
                pred, p_m = predict(model, X[te])
                rows.append(dict(**tag, protocol="loro", region=r, **score(y[te], pred, p_m)))
                if use_edge:
                    row = edge_eval(model, (edge.region == r).values, tag, "loro_edge", r)
                    if row:
                        rows.append(row)
                if fs == "pixel" and not centered:
                    fm = (pred == MANGROVE) & (y[te] != MANGROVE)
                    for c, n in pd.Series(y[te][fm]).value_counts().items():
                        false_src.append(dict(model=mdl, region=r, true_class=NAME[c], n=n))
            if anchor and anchor in regions and not centered:
                tr = (lab.region == anchor).values
                for r in regions:
                    if r == anchor:
                        continue
                    te = (lab.region == r).values
                    pred, p_m = fit_predict(mdl, X[tr], y[tr], X[te])
                    rows.append(dict(**tag, protocol=f"{anchor}_only", region=r,
                                     **score(y[te], pred, p_m)))
            if not skip_within and not centered:
                for r in regions:
                    idx = np.where((lab.region == r).values)[0]
                    groups = lab.block.values[idx]
                    k = min(5, len(np.unique(groups)))
                    if k < 2:
                        continue
                    folds, efolds = [], []
                    for tr_i, te_i in GroupKFold(k).split(idx, groups=groups):
                        a, b = idx[tr_i], idx[te_i]
                        if not (y[b] == MANGROVE).any() or not (y[a] == MANGROVE).any():
                            continue
                        model = make_model(mdl).fit(X[a], y[a])
                        pred, p_m = predict(model, X[b])
                        folds.append(score(y[b], pred, p_m))
                        if use_edge:  # edge points in this fold's held-out blocks
                            emask = ((edge.region == r) & edge.block.isin(set(groups[te_i]))).values
                            row = edge_eval(model, emask, tag, "within_edge", r)
                            if row:
                                efolds.append({k_: v for k_, v in row.items() if k_ in folds[0]})
                    if folds:
                        rows.append(dict(**tag, protocol="within_blocked", region=r,
                                         **pd.DataFrame(folds).mean().to_dict()))
                    if efolds:
                        rows.append(dict(**tag, protocol="within_edge", region=r,
                                         **pd.DataFrame(efolds).mean().to_dict()))
            log.info("done: %-12s %-6s centered=%s", fs, mdl, centered)
    return pd.DataFrame(rows), pd.DataFrame(false_src)


def summarize(res, false_src, out, anchor):
    res.to_csv(out / "results_long.csv", index=False)
    metrics = [c for c in res.columns if c.startswith(("mangrove_", "overall_"))]
    agg = (res.groupby(["protocol", "centered", "features", "model"])[metrics]
           .mean().round(3).reset_index())
    agg.to_csv(out / "results_summary.csv", index=False)

    def m(protocol, fs, mdl, centered=False):
        q = agg[(agg.protocol == protocol) & (agg.features == fs)
                & (agg.model == mdl) & (agg.centered == centered)]
        return float(q[METRIC].iloc[0]) if len(q) else np.nan

    models = sorted(res.model.unique())
    fsets = list(dict.fromkeys(res.features))
    print(f"\n=== mean {METRIC}, uncentered ===")
    piv = agg[~agg.centered].pivot_table(index=["features", "model"], columns="protocol",
                                         values=METRIC)
    print(piv.round(3).to_string())

    if len(false_src):
        fs_tab = (false_src.groupby(["model", "true_class"]).n.sum().unstack(0).fillna(0)
                  .astype(int).sort_values(models[0], ascending=False))
        fs_tab.to_csv(out / "q2_false_mangrove_sources.csv")
        print("\n=== false 'mangrove' calls by true class (LORO, pixel) ===")
        print(fs_tab.to_string())

    print("\n=== readout ===")
    if {"knn", "logreg"} <= set(models):
        d = m("loro", "pixel", "logreg") - m("loro", "pixel", "knn")
        print(f"Q2 logreg - kNN, LORO, pixel:        {d:+.3f}  "
              f"({'parametric transfers better' if d > 0.02 else 'no clear model effect'})")
    if {"mlp", "logreg"} <= set(models):
        d = m("loro", "pixel", "mlp") - m("loro", "pixel", "logreg")
        print(f"Q2 MLP - logreg, LORO, pixel:        {d:+.3f}  "
              f"({'nonlinearity helps' if d > 0.02 else 'linear is enough'})")
    best = max(models, key=lambda md: np.nan_to_num(m("loro", "pixel", md), nan=-1))
    print(f"reference model (best LORO, pixel): {best}")
    print(f"Q3 domain gap (within - LORO):       {m('within_blocked', 'pixel', best) - m('loro', 'pixel', best):+.3f}")
    if anchor and f"{anchor}_only" in set(res.protocol):
        print(f"Q3 {anchor}-only -> others:          {m(f'{anchor}_only', 'pixel', best):.3f}"
              f"  vs LORO {m('loro', 'pixel', best):.3f}  (gap = value of multi-region labels)")
    for fs in fsets:
        if fs != "pixel":
            dw = m("within_blocked", fs, best) - m("within_blocked", "pixel", best)
            dl = m("loro", fs, best) - m("loro", "pixel", best)
            print(f"Q4 {fs:12s} gain over pixel:     within {dw:+.3f} | LORO {dl:+.3f}")
    if res.centered.any():
        print(f"   region centering, LORO, pixel:   {m('loro', 'pixel', best, True) - m('loro', 'pixel', best):+.3f}")
    if "loro_edge" in set(res.protocol):
        print(f"Q4-edge: boundary pixels (another class within the edge radius), never trained on")
        print(f"   edge penalty, pixel (edge - all):  within {m('within_edge', 'pixel', best) - m('within_blocked', 'pixel', best):+.3f}"
              f" | LORO {m('loro_edge', 'pixel', best) - m('loro', 'pixel', best):+.3f}")
        for fs in fsets:
            if fs != "pixel":
                dw = m("within_edge", fs, best) - m("within_edge", "pixel", best)
                dl = m("loro_edge", fs, best) - m("loro_edge", "pixel", best)
                print(f"Q4-edge {fs:12s} gain over pixel: within {dw:+.3f} | LORO {dl:+.3f}")

    sub = res[(res.protocol == "loro") & (~res.centered) & (res.features == "pixel")]
    p = sub.pivot_table(index="region", columns="model", values=METRIC)
    ax = p.plot.bar(figsize=(10, 4)); ax.set_ylabel(f"{METRIC} (held-out region)")
    ax.set_title("Leave-one-region-out, pixel features"); plt.tight_layout()
    plt.savefig(out / "q2_loro_by_region.png", dpi=150); plt.close()

    sub = res[res.protocol.isin(["loro", "within_blocked", "loro_edge", "within_edge"])
              & (~res.centered) & (res.model == best)]
    p = sub.groupby(["features", "protocol"])[METRIC].mean().unstack().reindex(fsets)
    ax = p.plot.bar(figsize=(8, 4)); ax.set_ylabel(f"mean {METRIC}")
    ax.set_title(f"Spatial context ablation ({best})"); plt.tight_layout()
    plt.savefig(out / "q4_spatial_ablation.png", dpi=150); plt.close()


def run(data_dir, out, models=("knn", "logreg", "mlp"), fsets=None, anchor=None,
        skip_within=False, centering=True, block_deg=0.05):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lab, unl, edge = load(data_dir, block_deg)
    avail = present_scales(lab)
    fsets = [f for f in (fsets or FEATURE_SETS) if set(FEATURE_SETS[f]) <= avail]
    print(lab.groupby(["region", "class"]).size().unstack(fill_value=0)
          .rename(columns=NAME).to_string())
    geometry(lab, out)
    centers = region_centers(unl, lab) if centering else None
    if edge is not None:
        print(f"edge set: {len(edge)} boundary points, "
              f"{int((edge['class'] == MANGROVE).sum())} mangrove")
    res, false_src = run_protocols(lab, list(models), fsets, centers, skip_within, anchor, edge)
    summarize(res, false_src, out, anchor)
    print(f"\nWrote tables and plots to {out.resolve()}")
