"""
Multi-region SegFormer training on embedding chips.

Protocols
  loro    For each held-out region: train on all other regions, test on the held-out one.
          Main metric: transfer to an unseen region.
  pooled  Train on 80% of spatial blocks from every region, test on the remaining 20% of each
          region. In-distribution reference; pooled - loro = segmentation-level domain gap.

Model selection uses a validation split carved from the *training* chips (10% of spatial blocks),
never the test region. The held-out score reported is at the best validation epoch.

Outputs (results/seg/<run>/): metrics.json per split, summary.csv, best checkpoint per split.
"""
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from .. import CLASS_NAMES
from . import JaccardLoss, SegFormer

log = logging.getLogger(__name__)
IGNORE = 255
N_CLASSES = len(CLASS_NAMES)
MANGROVE = 0  # class 1 -> index 0


@dataclass
class TrainConfig:
    chips_dir: str = "data/chips"
    out_dir: str = "results/seg"
    weights: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    stem_stride: int | None = 2
    epochs: int = 40
    batch_size: int = 16
    lr_backbone: float = 6e-5
    lr_new: float = 6e-4          # stem, decode head, classifier
    weight_decay: float = 0.01
    alpha: float = 0.5            # JaccardLoss CE weight
    block_deg: float = 0.1        # spatial blocks for val / pooled splits
    val_frac: float = 0.1
    workers: int = 4
    seed: int = 0
    max_chips_per_region: int | None = None
    device: str | None = None


# ------------------------------------------------------------------ data
def load_index(chips_dir, regions=None):
    rows = []
    for idx in sorted(Path(chips_dir).glob("*/index.csv")):
        df = pd.read_csv(idx)
        df["path"] = [str(idx.parent / f"{c}.npz") for c in df.chip]
        rows.append(df[[Path(p).exists() for p in df.path]])
    if not rows:
        raise SystemExit(f"No chips under {chips_dir}. Run `mangrove chips` first.")
    df = pd.concat(rows, ignore_index=True)
    if regions:
        df = df[df.region.isin(regions)]
    return df.reset_index(drop=True)


def add_blocks(df, block_deg):
    bx = np.floor(df.lon / block_deg).astype(int).astype(str)
    by = np.floor(df.lat / block_deg).astype(int).astype(str)
    df = df.copy()
    df["block"] = df.region + "_" + bx + "_" + by
    return df


def split_blocks(df, frac, seed):
    """Boolean mask selecting ~frac of chips, by whole spatial blocks, within each region."""
    rng = np.random.default_rng(seed)
    pick = np.zeros(len(df), bool)
    for _, g in df.groupby("region"):
        blocks = np.array(g.block.unique(), dtype=object)
        rng.shuffle(blocks)
        n = max(1, int(round(frac * len(blocks)))) if len(blocks) > 1 else 0
        pick |= df.block.isin(set(blocks[:n])).values & (df.region == g.region.iloc[0]).values
    return pick


class ChipDataset(Dataset):
    def __init__(self, paths, augment=False):
        self.paths, self.augment = list(paths), augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        x = torch.from_numpy(d["x"].astype(np.float32))
        y = d["y"].astype(np.int64)
        y = np.where(y == IGNORE, IGNORE, y - 1)
        y = torch.from_numpy(y)
        if self.augment:  # dihedral group: embeddings are orientation-free per pixel
            k = int(torch.randint(0, 4, ()))
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (0, 1))
            if torch.rand(()) < 0.5:
                x, y = x.flip(2), y.flip(1)
        return x.contiguous(), y.contiguous()


# ------------------------------------------------------------------ metrics
class Meter:
    def __init__(self, ap_sample=200_000, seed=0):
        self.cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
        self.p, self.t = [], []
        self.ap_sample, self.rng = ap_sample, np.random.default_rng(seed)

    def update(self, logits, y):
        valid = y != IGNORE
        if not valid.any():
            return
        prob = logits.float().softmax(1)
        pred = prob.argmax(1)[valid].cpu().numpy()
        true = y[valid].cpu().numpy()
        self.cm += np.bincount(true * N_CLASSES + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        pm = prob[:, MANGROVE][valid].cpu().numpy()
        keep = self.rng.random(len(pm)) < min(1.0, self.ap_sample / max(len(pm), 1) * 4)
        self.p.append(pm[keep]); self.t.append(true[keep] == MANGROVE)

    def result(self):
        cm = self.cm
        tp = np.diag(cm).astype(float)
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        present = cm.sum(1) > 0
        prec = tp[MANGROVE] / max(tp[MANGROVE] + fp[MANGROVE], 1)
        rec = tp[MANGROVE] / max(tp[MANGROVE] + fn[MANGROVE], 1)
        p = np.concatenate(self.p) if self.p else np.array([])
        t = np.concatenate(self.t) if self.t else np.array([], bool)
        if len(p) > self.ap_sample:
            idx = self.rng.choice(len(p), self.ap_sample, replace=False)
            p, t = p[idx], t[idx]
        return dict(
            mangrove_iou=float(iou[MANGROVE]), mangrove_precision=float(prec),
            mangrove_recall=float(rec), mangrove_f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
            mangrove_ap=float(average_precision_score(t, p)) if t.any() else float("nan"),
            miou=float(iou[present].mean()), overall_acc=float(tp.sum() / max(cm.sum(), 1)),
            mangrove_px=int(cm[MANGROVE].sum()), valid_px=int(cm.sum()),
            per_class_iou={CLASS_NAMES[i]: float(iou[i]) for i in range(N_CLASSES) if present[i]},
        )


# ------------------------------------------------------------------ training
def _device(tc):
    if tc.device:
        return torch.device(tc.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(paths, tc, train):
    return DataLoader(ChipDataset(paths, augment=train), batch_size=tc.batch_size, shuffle=train,
                      num_workers=tc.workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(paths) > tc.batch_size,
                      persistent_workers=tc.workers > 0)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    m = Meter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            m.update(model(x), y)
    return m.result()


def class_weights(paths, cap=10.0):
    """Inverse sqrt frequency over training chips, capped; passed to JaccardLoss."""
    counts = np.zeros(N_CLASSES)
    for p in paths:
        y = np.load(p)["y"]
        y = y[y != IGNORE].astype(int) - 1
        counts += np.bincount(y, minlength=N_CLASSES)[:N_CLASSES]
    w = 1 / np.sqrt(np.maximum(counts, 1) / max(counts.sum(), 1))
    w = np.minimum(w / w[counts > 0].min(), cap)
    w[counts == 0] = 0
    return torch.tensor(w, dtype=torch.float32), counts


def fit(train_paths, val_paths, test_sets, tc: TrainConfig, run_dir: Path, tag: str):
    torch.manual_seed(tc.seed); np.random.seed(tc.seed)
    device = _device(tc)
    amp = device.type == "cuda"
    px = np.load(train_paths[0])["x"].shape[-1]
    in_ch = np.load(train_paths[0])["x"].shape[0]
    model = SegFormer(num_classes=N_CLASSES, input_image_size=px, weights=tc.weights,
                      in_channels=in_ch, stem_stride=tc.stem_stride).to(device)

    stem = next(m for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d))
    new_ids = {id(p) for p in stem.parameters()}
    new_ids |= {id(p) for mod in (model.decode_core, model.classifier) for p in mod.parameters()}
    params = [{"params": [p for p in model.parameters() if id(p) not in new_ids], "lr": tc.lr_backbone},
              {"params": [p for p in model.parameters() if id(p) in new_ids], "lr": tc.lr_new}]
    opt = torch.optim.AdamW(params, weight_decay=tc.weight_decay)
    tr_loader = _loader(train_paths, tc, True)
    steps = tc.epochs * len(tr_loader)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    weights, counts = class_weights(train_paths)
    loss_fn = JaccardLoss(num_classes=N_CLASSES, ignore_index=IGNORE, weight=weights.to(device),
                          alpha=tc.alpha)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    val_loader = _loader(val_paths, tc, False)

    best, best_ep, hist = -1.0, -1, []
    ckpt = run_dir / f"{tag}_best.pt"
    t0 = time.time()
    for ep in range(tc.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
        v = evaluate(model, val_loader, device, amp)
        score = v["mangrove_iou"]
        hist.append(dict(epoch=ep, loss=tot / max(n, 1), val_mangrove_iou=score, val_miou=v["miou"]))
        log.info("%s ep %02d loss %.4f val mangrove IoU %.3f mIoU %.3f (%.0fs)",
                 tag, ep, tot / max(n, 1), score, v["miou"], time.time() - t0)
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    res = dict(tag=tag, best_epoch=best_ep, val_mangrove_iou=best, history=hist,
               train_chips=len(train_paths), val_chips=len(val_paths),
               train_class_px={CLASS_NAMES[i]: int(c) for i, c in enumerate(counts)},
               config=asdict(tc), test={})
    for name, paths in test_sets.items():
        res["test"][name] = evaluate(model, _loader(paths, tc, False), device, amp)
        res["test"][name]["chips"] = len(paths)
    (run_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def run(tc: TrainConfig, protocol="loro", holdout="all", regions=None):
    df = add_blocks(load_index(tc.chips_dir, regions), tc.block_deg)
    if tc.max_chips_per_region:
        df = (df.groupby("region", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), tc.max_chips_per_region), random_state=tc.seed)))
    all_regions = sorted(df.region.unique())
    print(df.groupby(["region", "kind"]).size().unstack(fill_value=0).to_string())
    run_dir = Path(tc.out_dir) / protocol
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if protocol == "loro":
        if len(all_regions) < 2:
            raise SystemExit("loro needs chips from at least 2 regions.")
        targets = all_regions if holdout == "all" else [holdout]
        for r in targets:
            tr = df[df.region != r]
            val = split_blocks(tr, tc.val_frac, tc.seed)
            res = fit(list(tr.path[~val]), list(tr.path[val]),
                      {r: list(df.path[df.region == r])}, tc, run_dir, f"holdout_{r}")
            rows.append(dict(protocol="loro", region=r, **_flat(res["test"][r])))
            print(_line(rows[-1]))
    elif protocol == "pooled":
        test = split_blocks(df, 0.2, tc.seed + 1)
        tr = df[~test]
        val = split_blocks(tr, tc.val_frac, tc.seed)
        tests = {r: list(df.path[test & (df.region == r).values]) for r in all_regions}
        tests = {r: p for r, p in tests.items() if p}
        res = fit(list(tr.path[~val]), list(tr.path[val]), tests, tc, run_dir, "pooled")
        for r, m in res["test"].items():
            rows.append(dict(protocol="pooled", region=r, **_flat(m)))
            print(_line(rows[-1]))
    else:
        raise ValueError(protocol)

    summary = pd.DataFrame(rows)
    path = run_dir / "summary.csv"
    if path.exists() and holdout != "all":  # accumulate single-holdout runs
        prev = pd.read_csv(path)
        summary = pd.concat([prev[prev.region != holdout], summary], ignore_index=True)
    summary.to_csv(path, index=False)
    cols = ["region", "mangrove_f1", "mangrove_iou", "mangrove_ap", "miou"]
    print("\n" + summary[cols].round(3).to_string(index=False))
    print("mean:", summary[cols[1:]].mean().round(3).to_dict())
    return summary


def _flat(m):
    return {k: v for k, v in m.items() if not isinstance(v, dict)}


def _line(r):
    return (f"[{r['protocol']}] {r['region']}: mangrove F1 {r['mangrove_f1']:.3f} "
            f"IoU {r['mangrove_iou']:.3f} AP {r['mangrove_ap']:.3f} | mIoU {r['miou']:.3f}")
