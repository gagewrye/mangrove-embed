"""
Change prediction: next year's 11-class map from prior years' AlphaEarth embeddings.

Data: data/temporal (`mangrove temporal-chips`). Pixels are sampled per chip as full time series
(x [T, 64], y [T]) and split by train-seg's spatial blocks (test = 20% of each region's blocks,
val = 10% of the rest). Embeddings 2017-2025; GMW labels 2017-2020.

Protocol (forecast `test_year`, default 2020; its labels are never used for training):
  readout    per-pixel MLP E_t -> C_t on train blocks, GMW years before test_year. Makes
             pseudo-labels for the years after GMW (train blocks only) and classifies predicted
             embeddings.
  train      pairs (E_{t-k+1..t} -> C_{t+1}, E_{t+1}) on train blocks; targets are GMW years
             before test_year plus pseudo-labeled years after the last GMW year.
  val        GMW targets before test_year on val blocks (best epoch by mangrove IoU).
  test       target = test_year on test blocks, GMW labels.

Methods (one call scores all of them on the same pixels):
 baselines
  persistence             next year's map = this year's map (needs this year's true map)
  classify_current        classify this year's embedding (the baseline to beat)
  classify_true_next      classify the true next-year embedding: ceiling for the embedding route
                          (exact by construction with labels="classifier")
  transition_table        P(next class | this year's class) from training transitions (per region,
                          shrunk to the global table)
  extrapolate_embedding   classify normalize(2 E_t - E_{t-1}): linear trend in embedding space
 learned forecasters (per-pixel MLP on the last k embeddings)
  next_class              -> next year's class
  next_class_emb_aux      -> next year's class, plus an auxiliary head predicting next year's
                          embedding (1 - cosine, weight aux_weight)
  next_embedding          -> next year's embedding (residual on E_t, unit norm), then classify it
  next_class_given_map    -> next year's class, also given this year's map (one-hot input + a
                          learned persistence bias on the logits): learns deviations from no change
  next_class_given_probs  as next_class_given_map, with this year's class probabilities from the
                          classifier in place of the map (needs no labels at prediction time)
On train blocks, label years >= test_year are replaced by readout pseudo-labels, so the test
year's labels never enter training, not even as C_t for the following year.

labels="classifier": the targets are the readout's own maps. The readout is trained on all GMW
years (train blocks) and labels every sampled pixel in every year 2017-2025, test blocks
included. Training / val targets are the years before test_year (default 2025), test = test_year
on test blocks. GMW/GWL label noise drops out; persistence = classify_current, and
classify_true_next is exact by construction (kept as a check). `pseudo` is ignored.
labels="model": as "classifier", but the readout is a saved train-seg per-pixel MLP (label_run),
e.g. the regularized MLP, frozen. labels="logreg": as "classifier", with the benchmarked logistic
regression (fit on GMW train-block pixels) as the frozen readout.

merge_tide (default on): water is folded into tidal flat ("tidal_flat_water") in the labels and
the readout, because flats and open water swap with tide stage between annual composites.

Per-class outputs (classes.csv, transitions.csv): for every class c, enter_ap ranks the pixels
not c this year by P(c next year) and exit_ap ranks the pixels in c by 1 - P(c); transitions
counts every from -> to change in the truth and in each method's prediction.

Change metrics compare against the true C_t: changed_acc (accuracy where C_t != C_{t+1}),
change_iou (predicted change = prediction != C_t), change_ap (ranking by P(change)), and mangrove
gain / loss IoU and AP (gain_rate / loss_rate = the AP of a random ranking).
"""
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from . import CLASS_NAMES
from .temporal import LABEL_YEARS

log = logging.getLogger(__name__)
IGNORE = 255
N_CLASSES = len(CLASS_NAMES)
MANGROVE = 0  # class 1 -> index 0
TIDAL_FLAT, WATER = 2, 3  # merged for tide-free change: flats flip with tide stage
METHODS = ("persistence", "classify_current", "classify_true_next", "transition_table", "extrapolate_embedding", "next_class", "next_class_emb_aux", "next_embedding",
           "next_class_given_map", "next_class_given_probs")
TRAINED = ("next_class", "next_class_emb_aux", "next_embedding", "next_class_given_map", "next_class_given_probs")
PRIOR = ("next_class_given_map", "next_class_given_probs")


@dataclass
class ChangeConfig:
    temporal_dir: str = "data/temporal"
    out_dir: str = "results/change"
    regions: tuple | None = None
    methods: tuple = METHODS
    history: int = 1              # k prior years of embeddings as input
    test_year: int | None = None  # default 2020 (gmw) / 2025 (classifier)
    px_per_chip: int = 1000
    hidden: tuple = (128, 64)
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 8192
    lr: float = 2e-3
    weight_decay: float = 1e-4
    aux_weight: float = 1.0
    pseudo: bool = True           # train on pseudo-labeled targets after GMW (else GMW pairs only)
    labels: str = "gmw"           # gmw | classifier (readout maps are the targets) | model
    label_run: str | None = None  # labels="model": a train-seg run dir whose per-pixel MLP makes
                                  # the targets and is the readout (nothing is retrained)
    block_deg: float = 0.1        # = train-seg's, so splits match
    val_frac: float = 0.1
    seed: int = 0
    device: str | None = None
    tag: str | None = None
    select: str = "change_ap"     # forecaster epoch selection on val: change_ap | mangrove_iou
    merge_tide: bool = True       # water -> tidal flat everywhere (labels and readout), so tide-stage
                                  # flips between the two are not change


# ------------------------------------------------------------------ data
def sample_series(paths, n, seed):
    """Up to n random pixels per chip with an embedding in every year and a label in some year.
    -> x float16 [N, T, 64], y uint8 [N, T] (0..10, 255 = none), chip index [N], years [T]."""
    rng = np.random.default_rng(seed)
    xs, ys, cs, years = [], [], [], None
    for i, p in enumerate(paths):
        d = np.load(p)
        x, y = d["x"], d["y"]
        years = d["years"] if years is None else years
        ok = (np.abs(x).sum(1) > 0).all(0) & (y != IGNORE).any(0)
        flat = np.flatnonzero(ok)
        if not len(flat):
            continue
        h, w = np.unravel_index(rng.choice(flat, min(n, len(flat)), replace=False), ok.shape)
        xs.append(x[:, :, h, w].transpose(2, 0, 1))
        ys.append(np.where(y[:, h, w] == IGNORE, IGNORE, y[:, h, w] - 1).T.astype(np.uint8))
        cs.append(np.full(len(h), i, np.int32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs), np.asarray(years)


def load_series(cc: ChangeConfig, df):
    """sample_series, cached under <temporal_dir>/_cache (keyed on the chip list)."""
    cache = Path(cc.temporal_dir) / "_cache" / f"series_n{cc.px_per_chip}_s{cc.seed}_{len(df)}.npz"
    paths = list(df.path)
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["paths"]) == paths:
            return d["x"], d["y"], d["chip"], d["years"]
    t0 = time.time()
    x, y, chip, years = sample_series(paths, cc.px_per_chip, cc.seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y, chip=chip, years=years, paths=np.array(paths))
    log.info("sampled %d px from %d chips in %.0fs -> %s", len(y), len(paths), time.time() - t0, cache)
    return x, y, chip, years


def pairs(y, rows, target_cols, k):
    """(pixel, target column) pairs with a label and k prior years of input."""
    out = [np.c_[rows[y[rows, j] != IGNORE], np.full((y[rows, j] != IGNORE).sum(), j)]
           for j in target_cols if j >= k]
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


# ------------------------------------------------------------------ models
def trunk(c_in, hidden, dropout):
    layers, c = [], c_in
    for h in hidden:
        layers += [nn.Linear(c, h, bias=False), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        c = h
    return nn.Sequential(*layers), c


class Readout(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.body, c = trunk(64, hidden, dropout)
        self.head = nn.Linear(c, N_CLASSES)

    def forward(self, e):
        return self.head(self.body(e))


class PixelReadout(nn.Module):
    """A saved train-seg per-pixel model (1x1 convs) applied to embeddings [B, 64]."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, e):
        return self.model(e[:, :, None, None]).flatten(1)


def fit_logreg_readout(x, y, pairs_tr, cc, device, n=330_000):
    """The benchmarked logistic regression (pixel.make("logreg"): multinomial, raw embeddings)
    fit on a random sample of GMW-labeled train-block pixel-years, as a frozen nn.Linear."""
    from .pixel import make
    rng = np.random.default_rng(cc.seed)
    pick = pairs_tr[rng.choice(len(pairs_tr), min(n, len(pairs_tr)), replace=False)]
    t0 = time.time()
    lr = make("logreg", cc.seed).fit(x[pick[:, 0], pick[:, 1]].astype(np.float32),
                                     y[pick[:, 0], pick[:, 1]])
    lin = nn.Linear(64, N_CLASSES)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.fill_(-1e4)                      # classes absent from the sample: never chosen
        for row, c in enumerate(lr.classes_):
            lin.weight[int(c)] = torch.from_numpy(lr.coef_[row]).float()
            lin.bias[int(c)] = float(lr.intercept_[row])
    lin = lin.to(device).eval()
    for prm in lin.parameters():
        prm.requires_grad_(False)
    log.info("logreg labels: %d px, %d classes, %.0fs", len(pick), len(lr.classes_), time.time() - t0)
    return lin, dict(label_model="logreg", fit_px=int(len(pick)), classes=[int(c) for c in lr.classes_],
                     fit_seconds=round(time.time() - t0, 1))


class Forecaster(nn.Module):
    """E_{t-k+1..t} [B, k, 64] -> (class logits | None, next embedding | None). The embedding
    head is a residual on E_t, renormalized (AlphaEarth embeddings are unit vectors). With
    prior=True a this-year class vector [B, 11] (one-hot map or probabilities) is also an input
    and is added to the logits with a learned weight (starts at 3: favor no change)."""
    def __init__(self, k, hidden, dropout, cls=True, emb=False, prior=False):
        super().__init__()
        self.body, c = trunk(64 * k + (N_CLASSES if prior else 0), hidden, dropout)
        self.cls = nn.Linear(c, N_CLASSES) if cls else None
        self.emb = nn.Linear(c, 64) if emb else None
        self.persist = nn.Parameter(torch.tensor(3.0)) if prior else None
        if self.emb is not None:
            nn.init.zeros_(self.emb.weight); nn.init.zeros_(self.emb.bias)

    def forward(self, xh, prior=None):
        z = self.body(xh.flatten(1) if prior is None else torch.cat([xh.flatten(1), prior], 1))
        logits = self.cls(z) if self.cls is not None else None
        if logits is not None and prior is not None:
            logits = logits + self.persist * prior
        e = F.normalize(xh[:, -1] + self.emb(z), dim=-1) if self.emb is not None else None
        return logits, e


def gather(X, P, k):
    """X [N, T, 64] (device), P [B, 2] -> history [B, k, 64] float, next embedding [B, 64]."""
    i, j = P[:, 0], P[:, 1]
    cols = j[:, None] - k + torch.arange(k, device=X.device)
    return X[i[:, None], cols].float(), X[i, j].float()


def _batches(n, bs, shuffle, gen=None, device=None):
    idx = torch.randperm(n, generator=gen, device=device) if shuffle else torch.arange(n, device=device)
    return idx.split(bs)


@torch.no_grad()
def predict(fn, n, bs):
    outs = [fn(b) for b in torch.arange(n).split(bs)]
    return tuple(torch.cat([o[i] for o in outs]) if outs[0][i] is not None else None
                 for i in range(len(outs[0])))


def fit_model(model, train_n, loss_fn, val_fn, cc, device, name, metric="mangrove IoU"):
    """AdamW, warmup + cosine, best epoch by val_fn() (higher is better)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cc.lr, weight_decay=cc.weight_decay)
    steps = cc.epochs * max(1, -(-train_n // cc.batch_size))
    warm = max(1, steps // 20)  # linear warmup, then cosine (as train-seg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else
                                              0.5 * (1 + np.cos(np.pi * (t - warm) / max(steps - warm, 1))))
    gen = torch.Generator(device=device).manual_seed(cc.seed)
    best, best_state, hist, t0 = -1.0, None, [], time.time()
    for ep in range(cc.epochs):
        model.train()
        tot, n = 0.0, 0
        for b in _batches(train_n, cc.batch_size, True, gen, device):
            loss = loss_fn(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(b); n += len(b)
        model.eval()
        v = val_fn()
        hist.append(dict(epoch=ep, loss=tot / max(n, 1), val=v))
        log.info("%s ep %02d loss %.4f val %s %.3f (%.0fs)", name, ep, tot / max(n, 1), metric, v,
                 time.time() - t0)
        if v > best or best_state is None:  # NaN val score: keep the latest epoch
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return dict(selected_on=metric, best_val=best, history=hist)


# ------------------------------------------------------------------ metrics
def _iou(t, p):
    return float((t & p).sum() / max((t | p).sum(), 1))


def _merge_tide(a):
    a = np.asarray(a).copy()
    a[a == WATER] = TIDAL_FLAT
    return a


def _ap(pos, score):
    return float(average_precision_score(pos, score)) if pos.any() and (~pos).any() else float("nan")


def score(pred, tgt, prev, p_mangrove=None, p_change=None, probs=None, nxt=None):
    """Next-year map metrics + change metrics vs this year's true map (numpy, class indices),
    on pixels labeled in both years (so every method is scored on the same pixels).
    probs (all class probabilities) adds notide_* scores: tidal flat and water merged, so tidal
    flips are not change. nxt (the true map the year after) adds durable_* scores: positives are
    changes still present a year later, negatives are pixels that did not change; changes that
    revert are left out. *_rate = share of positives = the AP of a random ranking."""
    v = (tgt != IGNORE) & (prev != IGNORE)
    t, p = tgt[v].astype(int), pred[v].astype(int)
    cm = np.bincount(t * N_CLASSES + p, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    present = cm.sum(1) > 0
    out = dict(px=int(v.sum()), mangrove_iou=float(iou[MANGROVE]), miou=float(iou[present].mean()),
               acc=float(tp.sum() / max(cm.sum(), 1)))
    if p_mangrove is not None and (t == MANGROVE).any():
        out["mangrove_ap"] = float(average_precision_score(t == MANGROVE, p_mangrove[v]))
    c = v
    changed, stable = c & (prev != tgt), c & (prev == tgt)
    out["changed_px"] = int(changed.sum())
    out["changed_frac"] = float(changed.sum() / max(c.sum(), 1))
    out["changed_acc"] = float((pred[changed] == tgt[changed]).mean()) if changed.any() else float("nan")
    out["stable_acc"] = float((pred[stable] == tgt[stable]).mean()) if stable.any() else float("nan")
    out["change_iou"] = _iou(changed, c & (pred != prev))
    if p_change is not None and changed.any():  # threshold-free: does it rank changing pixels?
        out["change_ap"] = float(average_precision_score(changed[c], p_change[c]))
    was, now, will = prev == MANGROVE, tgt == MANGROVE, pred == MANGROVE
    out["gain_px"], out["loss_px"] = int((c & ~was & now).sum()), int((c & was & ~now).sum())
    out["gain_iou"] = _iou(c & ~was & now, c & ~was & will)
    out["loss_iou"] = _iou(c & was & ~now, c & was & ~will)
    if p_mangrove is not None:  # threshold-free mangrove change: rank by P(mangrove next year)
        g, l = c & ~was, c & was
        if now[g].any():
            out["gain_ap"] = float(average_precision_score(now[g], p_mangrove[g]))
        if (~now[l]).any():
            out["loss_ap"] = float(average_precision_score(~now[l], 1 - p_mangrove[l]))
        out["gain_rate"] = float(now[g].mean()) if g.any() else float("nan")   # = random AP
        out["loss_rate"] = float((~now[l]).mean()) if l.any() else float("nan")
    mp, mt = _merge_tide(prev), _merge_tide(tgt)
    ch_m = c & (mp != mt)
    out["notide_rate"] = float(ch_m.sum() / max(c.sum(), 1))
    if probs is not None:
        pr = probs.copy()
        pr[:, TIDAL_FLAT] += pr[:, WATER]
        p_change_m = 1 - pr[np.arange(len(pr)), np.minimum(mp, N_CLASSES - 1)]
        out["notide_ap"] = _ap(ch_m[c], p_change_m[c])
    if nxt is not None:
        d = c & (nxt != IGNORE)
        ev = d & (~changed | (nxt == tgt))                  # durable changes + no change
        dur = changed & (nxt == tgt)
        out["durable_rate"] = float(dur[ev].mean()) if ev.any() else float("nan")
        if p_change is not None:
            out["durable_ap"] = _ap(dur[ev], p_change[ev])
        mn = _merge_tide(nxt)
        ev_m = d & (~ch_m | (mn == mt))
        dur_m = ch_m & (mn == mt)
        out["durable_notide_rate"] = float(dur_m[ev_m].mean()) if ev_m.any() else float("nan")
        if probs is not None:
            out["durable_notide_ap"] = _ap(dur_m[ev_m], p_change_m[ev_m])
        stays = nxt == MANGROVE
        gd = d & ~was & (~now | stays)                      # durable gains + no gain
        ld = d & was & (now | ~stays)                       # durable losses + no loss
        out["durable_gain_rate"] = float(now[gd].mean()) if gd.any() else float("nan")
        out["durable_loss_rate"] = float((~now[ld]).mean()) if ld.any() else float("nan")
        if p_mangrove is not None:
            out["durable_gain_ap"] = _ap(now[gd], p_mangrove[gd])
            out["durable_loss_ap"] = _ap(~now[ld], 1 - p_mangrove[ld])
    return out


def class_names(merge_tide):
    names = list(CLASS_NAMES)
    if merge_tide:
        names[TIDAL_FLAT] = "tidal_flat_water"
    return names


class TideMerged(nn.Module):
    """Readout with water folded into tidal flat: P(tidal_flat_water) = P(tidal) + P(water)."""
    def __init__(self, readout):
        super().__init__()
        self.readout = readout

    def forward(self, e):
        z = self.readout(e)
        z = z.clone()
        z[:, TIDAL_FLAT] = torch.logsumexp(z[:, [TIDAL_FLAT, WATER]], 1)
        z[:, WATER] = -1e4
        return z


def class_scores(method, probs, tgt, prev, names=CLASS_NAMES):
    """Per class c: AP for ranking the pixels that become c (among those not c now) by P(c),
    and the pixels that stop being c (among those c now) by 1 - P(c). rate = random AP."""
    v = (tgt != IGNORE) & (prev != IGNORE)
    t, p0 = tgt[v], prev[v]
    rows = []
    for c, name in enumerate(names):
        if not ((p0 == c).any() or (t == c).any()):
            continue
        g, l = p0 != c, p0 == c
        row = dict(method=method, cls=name, now_px=int(l.sum()), enter_px=int((g & (t == c)).sum()),
                   exit_px=int((l & (t != c)).sum()))
        row["enter_rate"] = row["enter_px"] / max(g.sum(), 1)
        row["exit_rate"] = row["exit_px"] / max(l.sum(), 1)
        if probs is not None:
            pc = probs[v][:, c]
            if row["enter_px"]:
                row["enter_ap"] = float(average_precision_score(t[g] == c, pc[g]))
            if row["exit_px"] and row["exit_px"] < l.sum():
                row["exit_ap"] = float(average_precision_score(t[l] != c, 1 - pc[l]))
        rows.append(row)
    return rows


def transitions(method, pred, tgt, prev, names=CLASS_NAMES):
    """Every from -> to pair (from != to): true count, predicted count, correctly predicted."""
    v = (tgt != IGNORE) & (prev != IGNORE)
    p0, t, q = prev[v].astype(int), tgt[v].astype(int), pred[v].astype(int)
    n = N_CLASSES
    true = np.bincount(p0 * n + t, minlength=n * n).reshape(n, n)
    pr = np.bincount(p0 * n + q, minlength=n * n).reshape(n, n)
    hit = np.bincount((p0 * n + t)[q == t], minlength=n * n).reshape(n, n)
    return [dict(method=method, src=names[a], dst=names[b], true_px=int(true[a, b]),
                 pred_px=int(pr[a, b]), hit_px=int(hit[a, b]))
            for a in range(n) for b in range(n) if a != b and (true[a, b] or pr[a, b])]


# ------------------------------------------------------------------ run
def _device(cc):
    if cc.device:
        return torch.device(cc.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def run(cc: ChangeConfig):
    from .seg.train import add_blocks, load_index, split_blocks
    torch.manual_seed(cc.seed); np.random.seed(cc.seed)
    device, k = _device(cc), cc.history
    df = add_blocks(load_index(cc.temporal_dir, cc.regions), cc.block_deg)
    test_chip = split_blocks(df, 0.2, cc.seed + 1)
    val_chip = np.zeros(len(df), bool)
    val_chip[~test_chip] = split_blocks(df[~test_chip].reset_index(drop=True), cc.val_frac, cc.seed)
    x, y, chip, years = load_series(cc, df)
    years = [int(v) for v in years]
    split = np.where(test_chip[chip], 2, np.where(val_chip[chip], 1, 0))
    rows = {s: np.flatnonzero(split == i) for i, s in enumerate(("train", "val", "test"))}
    col = {yr: j for j, yr in enumerate(years)}
    clf = cc.labels in ("classifier", "model", "logreg")
    test_year = cc.test_year or (years[-1] if clf else 2020)
    jt = col[test_year]
    gmw = [col[yr] for yr in LABEL_YEARS if yr in col and (clf or yr < test_year)]
    pseudo = [col[yr] for yr in years if yr > max(LABEL_YEARS)] if cc.pseudo and not clf else []
    log.info("%d chips (train/val/test %d/%d/%d px), years %d-%d, k=%d, device %s", len(df),
             *(len(rows[s]) for s in ("train", "val", "test")), years[0], years[-1], k, device)

    X = torch.from_numpy(x).to(device)
    Y = torch.from_numpy(y.astype(np.int64)).to(device)
    bs = cc.batch_size

    # readout E_t -> C_t on GMW years before test_year
    ro_tr = torch.from_numpy(pairs(y, rows["train"], gmw, 0)).to(device)
    ro_va = torch.from_numpy(pairs(y, rows["val"], gmw, 0)).to(device)
    if cc.labels == "model":
        from .seg.train import load_checkpoint
        js = sorted(q for q in Path(cc.label_run).glob("*.json")
                    if q.with_name(f"{q.stem}_best.pt").exists())
        if not js:
            raise SystemExit(f"no checkpoint (*_best.pt) in {cc.label_run}")
        model, tc = load_checkpoint(js[0], device=str(device))
        if tc.model != "mlp" or tc.mlp_context:
            raise SystemExit(f"{cc.label_run}: labels need a per-pixel mlp without context")
        readout = PixelReadout(model).eval()
        for prm in readout.parameters():
            prm.requires_grad_(False)
    elif cc.labels == "logreg":
        readout, lr_info = fit_logreg_readout(x, y, ro_tr.cpu().numpy(), cc, device)
    else:
        readout = Readout(cc.hidden, cc.dropout).to(device)

    def classify(P, model=None):  # model resolved at call time (readout may be wrapped later)
        return lambda b: ((model or readout)(X[P[b, 0], P[b, 1]].float()),)

    def val_iou(logits, tgt):
        return _iou((tgt == MANGROVE).cpu().numpy(), (logits.argmax(1) == MANGROVE).cpu().numpy())

    res = dict(config=asdict(cc), test_year=test_year, years=years, chips=int(len(df)),
               px={s: int(len(r)) for s, r in rows.items()}, methods={})
    if cc.labels in ("model", "logreg"):
        res["readout"] = dict(val_mangrove_iou_gmw=val_iou(predict(classify(ro_va), len(ro_va), bs)[0],
                                                           Y[ro_va[:, 0], ro_va[:, 1]]))
        res["readout"].update(dict(label_run=str(cc.label_run), checkpoint=js[0].name)
                              if cc.labels == "model" else lr_info)
    else:
        res["readout"] = fit_model(
            readout, len(ro_tr),
            lambda b: F.cross_entropy(readout(X[ro_tr[b, 0], ro_tr[b, 1]].float()), Y[ro_tr[b, 0], ro_tr[b, 1]]),
            lambda: val_iou(predict(classify(ro_va), len(ro_va), bs)[0], Y[ro_va[:, 0], ro_va[:, 1]]),
            cc, device, "readout")
    if cc.merge_tide:  # labels and readout: water folded into tidal flat
        readout = TideMerged(readout).eval()
        Y[Y == WATER] = TIDAL_FLAT
        y = np.where(y == WATER, TIDAL_FLAT, y).astype(y.dtype)

    if clf:  # the readout's maps become the labels, every pixel and year
        all_rows = torch.arange(len(y), device=device)
        for j in range(len(years)):
            P = torch.stack([all_rows, torch.full_like(all_rows, j)], 1)
            Y[:, j] = predict(classify(P), len(P), bs)[0].argmax(1)
        y = Y.cpu().numpy().astype(np.uint8)
        gmw = [j for j, yr in enumerate(years) if yr < test_year]   # target years from here on

    # pseudo-labels for years after GMW, and in place of labels from test_year on, train pixels only
    Yt = Y.clone()
    tr_rows = torch.from_numpy(rows["train"]).to(device)
    hidden_yrs = [] if clf else [col[yr] for yr in LABEL_YEARS if yr in col and yr >= test_year]
    for j in pseudo + hidden_yrs:
        P = torch.stack([tr_rows, torch.full_like(tr_rows, j)], 1)
        Yt[tr_rows, j] = predict(classify(P), len(P), bs)[0].argmax(1)
    y_t = Yt.cpu().numpy()
    tr = torch.from_numpy(pairs(y_t, rows["train"], gmw + pseudo, k)).to(device)
    va = torch.from_numpy(pairs(y, rows["val"], gmw, k)).to(device)
    te_np = pairs(y, rows["test"], [jt], k)
    te = torch.from_numpy(te_np).to(device)
    res["pairs"] = dict(train=int(len(tr)), val=int(len(va)), test=int(len(te)),
                        train_gmw_targets=[years[j] for j in gmw if j >= k],
                        train_pseudo_targets=[years[j] for j in pseudo if j >= k])
    log.info("pairs: train %d, val %d, test %d", len(tr), len(va), len(te))

    tgt = y[te_np[:, 0], te_np[:, 1]]
    prev = y[te_np[:, 0], te_np[:, 1] - 1]
    nxt = y[te_np[:, 0], jt + 1] if jt + 1 < len(years) else None   # for durable change
    region = df.region.values[chip[te_np[:, 0]]]
    e_now, e_next = (X[te[:, 0], te[:, 1] - 1].float(), X[te[:, 0], te[:, 1]].float())
    cos_persist = float((e_now * e_next).sum(1).mean())

    prev_idx = torch.from_numpy(np.minimum(prev, N_CLASSES - 1).astype(np.int64)).to(device)

    def from_logits(logits):
        """-> (class, P(mangrove), P(change) = 1 - P(this year's class))."""
        prob = logits.float().softmax(1)
        stay = prob.gather(1, prev_idx[:, None])[:, 0]
        return (prob.argmax(1).cpu().numpy(), prob[:, MANGROVE].cpu().numpy(),
                (1 - stay).cpu().numpy(), prob.cpu().numpy())

    def readout_of(E):
        return from_logits(predict(lambda b: (readout(E[b]),), len(E), bs)[0])

    def prior_of(m, P, labels):
        """This-year class vector for pairs P: one-hot of labels (zeros where unlabeled) for
        next_class_given_map, readout(E_t) probabilities for next_class_given_probs."""
        i, j = P[:, 0], P[:, 1] - 1
        if m == "next_class_given_probs":
            with torch.no_grad():
                return readout(X[i, j].float()).softmax(1)
        c = labels[i, j]
        return F.one_hot(c.clamp(max=N_CLASSES - 1), N_CLASSES).float() * (c != IGNORE)[:, None]

    def markov_probs():
        """Transition table from train-block GMW pairs; per region, shrunk to the global one."""
        P = pairs(y, rows["train"], [j for j in gmw if j >= 1], 1)
        a, b = y[P[:, 0], P[:, 1] - 1], y[P[:, 0], P[:, 1]]
        ok = a != IGNORE
        reg_tr = df.region.values[chip[P[:, 0]]][ok]
        a, b = a[ok].astype(int), b[ok].astype(int)
        glob = np.bincount(a * N_CLASSES + b, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES) + 1.0
        glob /= glob.sum(1, keepdims=True)
        out = np.tile(np.full(N_CLASSES, 1 / N_CLASSES), (len(prev), 1))
        for r in set(region):
            s = reg_tr == r
            t = np.bincount(a[s] * N_CLASSES + b[s], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
            t = t + 100 * glob                      # ~100 pseudo-counts of the global table per row
            t /= t.sum(1, keepdims=True)
            sel = (region == r) & (prev != IGNORE)
            out[sel] = t[prev[sel]]
        return out

    va_prev, va_tgt = Y[va[:, 0], va[:, 1] - 1], Y[va[:, 0], va[:, 1]]
    va_ok = ((va_prev != IGNORE) & (va_tgt != IGNORE)).cpu().numpy()
    va_changed = (va_prev != va_tgt).cpu().numpy()[va_ok]

    def val_change_ap(logits):
        """AP of P(change) = 1 - P(this year's class) for the val pixels that change class."""
        stay = logits.float().softmax(1).gather(1, va_prev.clamp(max=N_CLASSES - 1)[:, None])[:, 0]
        return _ap(va_changed, (1 - stay).cpu().numpy()[va_ok])

    preds = {}
    for m in cc.methods:
        info = {}
        if m == "persistence":
            preds[m] = (prev, None, None, None)
        elif m == "classify_current":
            preds[m] = readout_of(e_now)
        elif m == "classify_true_next":
            preds[m] = readout_of(e_next)
        elif m == "transition_table":
            p = markov_probs()
            preds[m] = (p.argmax(1), p[:, MANGROVE],
                        1 - p[np.arange(len(p)), np.minimum(prev, N_CLASSES - 1)], p)
        elif m == "extrapolate_embedding":
            if jt < 2:
                raise ValueError("trend needs two years before test_year")
            e_prev = X[te[:, 0], te[:, 1] - 2].float()
            preds[m] = readout_of(F.normalize(2 * e_now - e_prev, dim=-1))
        elif m in TRAINED:
            torch.manual_seed(cc.seed)
            net = Forecaster(k, cc.hidden, cc.dropout, cls=m != "next_embedding", emb=m in ("next_class_emb_aux", "next_embedding"),
                             prior=m in PRIOR).to(device)

            def fwd(P, net=net, m=m):
                return lambda b: net(gather(X, P[b], k)[0],
                                     prior_of(m, P[b], Y) if m in PRIOR else None)

            def loss_fn(b, net=net, m=m):
                xh, en = gather(X, tr[b], k)
                logits, e = net(xh, prior_of(m, tr[b], Yt) if m in PRIOR else None)
                loss = 0.0
                if logits is not None:
                    loss = F.cross_entropy(logits, Yt[tr[b, 0], tr[b, 1]], ignore_index=IGNORE)
                if e is not None:
                    w = 1.0 if m == "next_embedding" else cc.aux_weight
                    loss = loss + w * (1 - (e * en).sum(1)).mean()
                return loss

            def val_fn(net=net, m=m):
                logits, e = predict(fwd(va), len(va), bs)
                if m == "next_embedding":
                    logits = predict(lambda b: (readout(e[b]),), len(e), bs)[0]
                if cc.select == "mangrove_iou":
                    return val_iou(logits, Y[va[:, 0], va[:, 1]])
                return val_change_ap(logits)

            info = fit_model(net, len(tr), loss_fn, val_fn, cc, device, m,
                             "change AP" if cc.select == "change_ap" else "mangrove IoU")
            info["params"] = sum(p.numel() for p in net.parameters())
            logits, e = predict(fwd(te), len(te), bs)
            if e is not None:
                info["cos_next"] = float((e * e_next).sum(1).mean())
            preds[m] = readout_of(e) if m == "next_embedding" else from_logits(logits)
        else:
            raise ValueError(m)
        res["methods"][m] = info

    out_rows = []
    cls_rows, tr_rows_out = [], []
    for m, (pred, pm, pc, probs) in preds.items():
        cls_rows += class_scores(m, probs, tgt, prev, class_names(cc.merge_tide))
        tr_rows_out += transitions(m, pred, tgt, prev, class_names(cc.merge_tide))
        for r in ["all"] + sorted(set(region)):
            sel = slice(None) if r == "all" else region == r
            s = score(pred[sel], tgt[sel], prev[sel], None if pm is None else pm[sel],
                      None if pc is None else pc[sel], None if probs is None else probs[sel],
                      None if nxt is None else nxt[sel])
            out_rows.append(dict(method=m, region=r, **s))
        res["methods"][m]["test"] = out_rows[-1 - len(set(region))]
    summary = pd.DataFrame(out_rows)
    res["cos_persist"] = cos_persist
    mode = {"classifier": "clf", "model": "model", "logreg": "logreg"}.get(cc.labels)
    yr = f"y{test_year}" if cc.labels in ("model", "logreg") else None
    folder = "_".join(v for v in ("change", mode, f"k{k}", yr, cc.tag, f"s{cc.seed}") if v)
    run_dir = Path(cc.out_dir) / folder
    run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(run_dir / "classes.csv", index=False)
    pd.DataFrame(tr_rows_out).to_csv(run_dir / "transitions.csv", index=False)
    (run_dir / "change.json").write_text(json.dumps(res, indent=2, default=float))
    cols = ["method", "mangrove_iou", "miou", "acc", "changed_acc", "change_iou", "change_ap",
            "gain_iou", "loss_iou", "gain_ap", "loss_ap"]
    print(summary[summary.region == "all"][[c for c in cols if c in summary]].round(3).to_string(index=False))
    print(f"cos(E_t, E_t+1) = {cos_persist:.3f}" + "".join(
        f" | {m} cos {i['cos_next']:.3f}" for m, i in res["methods"].items() if "cos_next" in i))
    print(f"-> {run_dir}")
    return summary
