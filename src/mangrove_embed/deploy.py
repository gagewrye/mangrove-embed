"""
Deployable change models: the frozen classifier and the next_class forecaster as one file.

The forecaster's targets are the classifier's maps, so the two only make sense together; the
bundle ships both. It is written by `mangrove change --labels logreg --save-bundle PATH` (the
weights scored in that run's change.json, nothing retrained) and read by the two commands below.

Bundle (torch.save, loadable with weights_only=True):
  classifier   frozen logistic regression E_t -> C_t (nn.Linear 64 -> 11)
  forecaster   next_class MLP on the last `history` embeddings -> C_{t+1}; with merge_tide, any
               probability it leaves on water is added to tidal flat at prediction time
  merge_tide   water folded into tidal flat in both (code 3 = tidal_flat_water, code 4 unused)
  check        reference inputs and logits from training; load() refuses a bundle that no
               longer reproduces them (e.g. a changed model definition)
  meta         training config, years, regions, test scores

Inputs are 64-band AlphaEarth GeoTIFFs, one per year, on the same grid. Class codes are 1-11 as
CLASS_NAMES; 255 / NaN = no data (a pixel is no data unless every input year has a finite,
nonzero embedding there).

change-predict (forecast; `history` years in, oldest first):
  <out>_classes.tif  uint8: class_current (latest year), class_next (forecast)
  <out>_probs.tif    float32: p_change = 1 - P(next = current class), p_mangrove = P(mangrove next
                     year), then P(class) per class with --probs

change-detect (what changed between two observed years, classifier-driven):
  <out>_classes.tif  uint8: class_before, class_after
  <out>_probs.tif    float32: p_change = 1 - sum_c P_before(c) P_after(c) (the classes differ,
                     treating the two years' posteriors as independent), p_mangrove_gain =
                     (1 - P_before(m)) P_after(m), p_mangrove_loss = P_before(m) (1 - P_after(m)),
                     cosine = E_before . E_after (the embedding-only similarity, for reference)
  <out>_transitions.csv  from -> to area (ha) where the argmax class changes and p_change >= min_p;
                         <out>_classes.csv  per-class area before / after, gain, loss, net (ha)
A dot product says *whether* a pixel moved; the class pair says *what* it became.
"""
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import BANDS, CLASS_IDS
from .change import MANGROVE, N_CLASSES, TIDAL_FLAT, WATER, Forecaster, TideMerged, class_names

log = logging.getLogger(__name__)
FORMAT, VERSION = "mangrove-change-bundle", 1
NODATA = 255


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


def save_bundle(path, classifier, forecaster, history, hidden, dropout, merge_tide, check_x, meta):
    """classifier: nn.Linear(64, 11) (before tide merging); forecaster: trained next_class
    Forecaster; check_x: [n, history, 64] reference inputs (test pixels)."""
    clf = TideMerged(classifier).eval() if merge_tide else classifier.eval()
    forecaster.eval()
    x = check_x.detach().float().cpu()
    with torch.no_grad():
        dev = next(forecaster.parameters()).device
        c_logits = clf(x[:, -1].to(dev)).cpu()
        f_logits = forecaster(x.to(dev))[0].cpu()
    bundle = dict(
        format=FORMAT, version=VERSION, class_names=class_names(merge_tide), class_codes=list(CLASS_IDS),
        merge_tide=bool(merge_tide),
        classifier=dict(kind="linear", state={k: v.cpu() for k, v in classifier.state_dict().items()}),
        forecaster=dict(kind="next_class", history=int(history), hidden=[int(h) for h in hidden],
                        dropout=float(dropout), state={k: v.cpu() for k, v in forecaster.state_dict().items()}),
        check=dict(x=x, classifier_logits=c_logits, forecaster_logits=f_logits),
        meta=json.loads(json.dumps(dict(meta, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                        commit=_git_commit(), torch=torch.__version__), default=str)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    log.info("bundle -> %s", path)
    return path


class Bundle:
    """A loaded bundle: classify(e) on [N, 64], forecast(x) on [N, history, 64]."""

    def __init__(self, d, device="cpu"):
        if d.get("format") != FORMAT or d.get("version") != VERSION:
            raise ValueError(f"not a {FORMAT} v{VERSION} file")
        self.device = torch.device(device)
        self.names, self.merge_tide, self.meta = d["class_names"], d["merge_tide"], d["meta"]
        f = d["forecaster"]
        self.history = f["history"]
        lin = nn.Linear(64, N_CLASSES)
        lin.load_state_dict(d["classifier"]["state"])
        self.classifier = (TideMerged(lin) if self.merge_tide else lin).to(self.device).eval()
        self.forecaster = Forecaster(self.history, tuple(f["hidden"]), f["dropout"]).to(self.device)
        self.forecaster.load_state_dict(f["state"])
        self.forecaster.eval()
        self._check(d["check"])

    @classmethod
    def load(cls, path, device=None):
        return cls(torch.load(path, map_location="cpu", weights_only=True), device or _auto_device())

    @torch.no_grad()
    def _check(self, chk):
        x = chk["x"].to(self.device)
        for name, got, want in (("classifier", self.classifier(x[:, -1]), chk["classifier_logits"]),
                                ("forecaster", self.forecaster(x)[0], chk["forecaster_logits"])):
            err = float((got.cpu() - want).abs().max()) if len(x) else 0.0
            if err > 1e-3:
                raise ValueError(f"bundle self-check failed: {name} logits differ by {err:.2e}")

    @torch.no_grad()
    def classify(self, e, batch=262_144):
        """e [N, 64] -> class probabilities [N, 11] (numpy)."""
        e = torch.as_tensor(e, dtype=torch.float32)
        out = [self.classifier(b.to(self.device)).float().softmax(1).cpu().numpy() for b in e.split(batch)]
        return np.concatenate(out) if out else np.zeros((0, N_CLASSES), np.float32)

    @torch.no_grad()
    def forecast(self, x, batch=262_144):
        """x [N, history, 64] (oldest year first) -> dict of numpy arrays: current, next
        (class index 0-10), p_change, p_mangrove [N], probs [N, 11]."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.history, 64):
            raise ValueError(f"expected [N, {self.history}, 64], got {tuple(x.shape)}")
        out = {k: [] for k in ("current", "next", "p_change", "p_mangrove", "probs")}
        for b in torch.as_tensor(x, dtype=torch.float32).split(batch):
            b = b.to(self.device)
            cur = self.classifier(b[:, -1]).argmax(1)
            prob = self.forecaster(b)[0].float().softmax(1)
            if self.merge_tide:  # targets never contain water; fold any residual mass into tidal flat
                prob[:, TIDAL_FLAT] += prob[:, WATER]
                prob[:, WATER] = 0
            for k, v in (("current", cur), ("next", prob.argmax(1)),
                         ("p_change", 1 - prob.gather(1, cur[:, None])[:, 0]),
                         ("p_mangrove", prob[:, MANGROVE]), ("probs", prob)):
                out[k].append(v.cpu().numpy())
        empty = dict(probs=np.zeros((0, N_CLASSES), np.float32))
        return {k: np.concatenate(v) if v else empty.get(k, np.zeros(0)) for k, v in out.items()}

    def detect(self, before, after):
        """Observed change between two years. before, after [N, 64] -> dict of numpy arrays:
        before / after (class index), p_change, p_mangrove_gain, p_mangrove_loss, cosine [N]."""
        p0, p1 = self.classify(before), self.classify(after)
        b, a = np.asarray(before, np.float32), np.asarray(after, np.float32)
        cos = (b * a).sum(1) / np.maximum(np.linalg.norm(b, axis=1) * np.linalg.norm(a, axis=1), 1e-12)
        m0, m1 = p0[:, MANGROVE], p1[:, MANGROVE]
        return dict(before=p0.argmax(1), after=p1.argmax(1), p_change=1 - (p0 * p1).sum(1),
                    p_mangrove_gain=(1 - m0) * m1, p_mangrove_loss=m0 * (1 - m1), cosine=cos)


# ------------------------------------------------------------------ GeoTIFFs
def _auto_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _band_order(ds):
    """Indices (1-based) of A00..A63: by band description if set, else the first 64 bands."""
    desc = list(ds.descriptions)
    if all(b in desc for b in BANDS):
        return [desc.index(b) + 1 for b in BANDS]
    if ds.count != 64:
        raise ValueError(f"{ds.name}: {ds.count} bands without A00..A63 descriptions; need 64")
    return list(range(1, 65))


def _tiles(inputs, out, cls_names, prob_names, tags, block, fn):
    """Read the input GeoTIFFs window by window ([k, 64, h, w] + valid mask), call
    fn(x [n, k, 64]) -> (classes [2, n] uint8, probs [len(prob_names), n] float32) on valid
    pixels, and write <out>_classes.tif / <out>_probs.tif on the input grid."""
    import rasterio
    from rasterio.windows import Window
    srcs = [rasterio.open(p) for p in inputs]
    try:
        ref = srcs[0]
        for s in srcs[1:]:
            if (s.crs, s.transform, s.width, s.height) != (ref.crs, ref.transform, ref.width, ref.height):
                raise ValueError(f"{s.name} is not on the same grid as {ref.name}")
        order = [_band_order(s) for s in srcs]
        base = dict(driver="GTiff", width=ref.width, height=ref.height, crs=ref.crs, transform=ref.transform,
                    tiled=True, blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        p_cls, p_prob = out.with_name(out.name + "_classes.tif"), out.with_name(out.name + "_probs.tif")
        tags = dict(tags, inputs=json.dumps([str(p) for p in inputs]))
        with rasterio.open(p_cls, "w", count=len(cls_names), dtype="uint8", nodata=NODATA, **base) as oc, \
             rasterio.open(p_prob, "w", count=len(prob_names), dtype="float32", nodata=np.nan, **base) as op:
            oc.descriptions, op.descriptions = tuple(cls_names), tuple(prob_names)
            oc.update_tags(**tags)
            op.update_tags(**tags)
            for r in range(0, ref.height, block):
                for c in range(0, ref.width, block):
                    win = Window(c, r, min(block, ref.width - c), min(block, ref.height - r))
                    x = np.stack([s.read(o, window=win).astype(np.float32) for s, o in zip(srcs, order)])
                    ok = np.isfinite(x).all((0, 1)) & (np.abs(np.nan_to_num(x)).sum(1) > 0).all(0)
                    for s in srcs:
                        ok &= s.dataset_mask(window=win) > 0
                    cls = np.full((len(cls_names), *ok.shape), NODATA, np.uint8)
                    pr = np.full((len(prob_names), *ok.shape), np.nan, np.float32)
                    if ok.any():
                        cls[:, ok], pr[:, ok] = fn(np.ascontiguousarray(x[:, :, ok].transpose(2, 0, 1)))
                    oc.write(cls, window=win)
                    op.write(pr, window=win)
        pixel_ha = abs(ref.transform.a * ref.transform.e) / 1e4 if ref.crs and ref.crs.is_projected else None
    finally:
        for s in srcs:
            s.close()
    log.info("-> %s, %s", p_cls, p_prob)
    return p_cls, p_prob, pixel_ha


def _tags(b):
    return dict(classes=json.dumps(dict(zip(CLASS_IDS, b.names))),
                bundle=json.dumps({k: b.meta.get(k) for k in ("created", "commit", "years", "test_year")}))


def predict_rasters(bundle, inputs, out, block=512, probs=False, device=None):
    """Forecast next year's map from one embedding GeoTIFF per year (oldest first, same grid).
    Returns (classes path, probs path)."""
    b = bundle if isinstance(bundle, Bundle) else Bundle.load(bundle, device)
    if len(inputs) != b.history:
        raise ValueError(f"bundle needs {b.history} years of embeddings (oldest first), got {len(inputs)}")
    names = ["p_change", "p_mangrove"] + ([f"p_{n}" for n in b.names] if probs else [])

    def fn(x):
        f = b.forecast(x)
        p = [f["p_change"], f["p_mangrove"]] + (list(f["probs"].T) if probs else [])
        return np.stack([f["current"] + 1, f["next"] + 1]), np.stack(p)

    p_cls, p_prob, _ = _tiles(inputs, out, ("class_current", "class_next"), names, _tags(b), block, fn)
    return p_cls, p_prob


def detect_rasters(bundle, before, after, out, block=512, min_p=0.5, device=None):
    """Classifier-driven change between two observed years (same grid). Writes the two rasters
    plus transition / per-class area tables; returns (classes, probs, transitions_csv, classes_csv)."""
    b = bundle if isinstance(bundle, Bundle) else Bundle.load(bundle, device)
    n = N_CLASSES
    counts = np.zeros((n, n), np.int64)  # from -> to, argmax changed and p_change >= min_p
    totals = np.zeros((2, n), np.int64)  # per-class pixels before / after (all valid pixels)
    names = ("p_change", "p_mangrove_gain", "p_mangrove_loss", "cosine")

    def fn(x):
        d = b.detect(x[:, 0], x[:, 1])
        totals[0] += np.bincount(d["before"], minlength=n)
        totals[1] += np.bincount(d["after"], minlength=n)
        sel = (d["before"] != d["after"]) & (d["p_change"] >= min_p)
        counts[:] += np.bincount(d["before"][sel] * n + d["after"][sel], minlength=n * n).reshape(n, n)
        return np.stack([d["before"] + 1, d["after"] + 1]), np.stack([d[k] for k in names])

    p_cls, p_prob, ha = _tiles([before, after], out, ("class_before", "class_after"), names,
                               dict(_tags(b), min_p=str(min_p)), block, fn)
    ha = ha or float("nan")  # geographic CRS: areas unknown, pixel counts still reported
    out = Path(out)
    p_tr, p_cl = out.with_name(out.name + "_transitions.csv"), out.with_name(out.name + "_classes.csv")
    rows = [dict(src=b.names[i], dst=b.names[j], src_code=CLASS_IDS[i], dst_code=CLASS_IDS[j],
                 px=int(counts[i, j]), ha=counts[i, j] * ha)
            for i in range(n) for j in range(n) if i != j and counts[i, j]]
    pd.DataFrame(rows, columns=["src", "dst", "src_code", "dst_code", "px", "ha"]) \
        .sort_values("px", ascending=False).round(4).to_csv(p_tr, index=False)
    gain, loss = counts.sum(0), counts.sum(1)
    pd.DataFrame(dict(cls=b.names, code=CLASS_IDS, before_ha=totals[0] * ha, after_ha=totals[1] * ha,
                      gain_ha=gain * ha, loss_ha=loss * ha, net_ha=(gain - loss) * ha,
                      before_px=totals[0], after_px=totals[1], gain_px=gain, loss_px=loss)) \
        .query("before_px > 0 or after_px > 0").round(4).to_csv(p_cl, index=False)
    log.info("-> %s, %s", p_tr, p_cl)
    return p_cls, p_prob, p_tr, p_cl
