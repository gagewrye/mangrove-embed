"""Binary mangrove metrics at a probability threshold, and threshold tuning (numpy only)."""
import numpy as np

GRID = np.round(np.linspace(0.02, 0.98, 97), 2)


def binary_scores(is_mangrove, p_mangrove, thr):
    t = np.asarray(is_mangrove, bool)
    pred = np.asarray(p_mangrove) >= thr
    tp, fp, fn = (t & pred).sum(), (~t & pred).sum(), (t & ~pred).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(iou=float(tp / max(tp + fp + fn, 1)), f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                precision=float(prec), recall=float(rec))


def tune_threshold(is_mangrove, p_mangrove, grid=GRID):
    """Threshold maximizing mangrove IoU. Ties -> the one closest to 0.5."""
    scores = np.array([binary_scores(is_mangrove, p_mangrove, g)["iou"] for g in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    i = best[np.argmin(np.abs(grid[best] - 0.5))]
    return float(grid[i]), float(scores[i])


def tuned_metrics(val_t, val_p, test_t, test_p):
    """Tune on validation pixels, report on test pixels. Keys suffixed _tuned."""
    thr, val_iou = tune_threshold(val_t, val_p)
    s = binary_scores(test_t, test_p, thr)
    return {"threshold": thr, "val_mangrove_iou_tuned": val_iou,
            "mangrove_iou_tuned": s["iou"], "mangrove_f1_tuned": s["f1"],
            "mangrove_precision_tuned": s["precision"], "mangrove_recall_tuned": s["recall"]}
