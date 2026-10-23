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
import contextlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from .. import CLASS_NAMES
from ..thresholds import tuned_metrics
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
    model: str = "segformer"      # segformer | fcn (dilated CNN) | mlp (per-pixel, no context)
    pca: int | None = None        # fixed PCA 64 -> k, fit on training-region pixels only
    input_proj: int | None = None # learned 1x1 conv 64 -> k in front of the model
    fcn_width: int = 64
    fcn_dilations: tuple = (1, 2, 4, 8, 16)  # one residual block per entry; 0 = 1x1 (no context)
    mlp_hidden: tuple = (64,)     # per-pixel MLP hidden widths
    name_seed: bool = False       # include the seed in the results folder name (set by --seed)
    from_scratch: bool = False    # segformer with random init (same architecture)
    amp: bool | None = None       # mixed precision; None = on for CUDA, off for MPS/CPU
    wandb: bool = False           # log to Weights & Biases (needs the [track] extra)
    wandb_project: str = "mangrove-embed"
    wandb_entity: str | None = None
    wandb_name: str | None = None   # run name; the split is appended (name/holdout_mekong)
    wandb_group: str | None = None  # default: variant folder (loro_fcn, ...)
    wandb_tags: tuple = ()
    wandb_notes: str | None = None


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

    def scores(self):
        """Uniform pixel sample of (is_mangrove, p_mangrove), for AP and threshold tuning."""
        p = np.concatenate(self.p) if self.p else np.array([])
        t = np.concatenate(self.t) if self.t else np.array([], bool)
        if len(p) > self.ap_sample:
            idx = np.random.default_rng(0).choice(len(p), self.ap_sample, replace=False)
            p, t = p[idx], t[idx]
        return t, p

    def result(self):
        cm = self.cm
        tp = np.diag(cm).astype(float)
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        present = cm.sum(1) > 0
        prec = tp[MANGROVE] / max(tp[MANGROVE] + fp[MANGROVE], 1)
        rec = tp[MANGROVE] / max(tp[MANGROVE] + fn[MANGROVE], 1)
        t, p = self.scores()
        return dict(
            mangrove_iou=float(iou[MANGROVE]), mangrove_precision=float(prec),
            mangrove_recall=float(rec), mangrove_f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
            mangrove_ap=float(average_precision_score(t, p)) if t.any() else float("nan"),
            miou=float(iou[present].mean()), overall_acc=float(tp.sum() / max(cm.sum(), 1)),
            mangrove_px=int(cm[MANGROVE].sum()), valid_px=int(cm.sum()),
            per_class_iou={CLASS_NAMES[i]: float(iou[i]) for i in range(N_CLASSES) if present[i]},
        )


# ------------------------------------------------------------------ input reduction + models
class PCAProject(torch.nn.Module):
    """Fixed linear map x -> components @ (x - mean), per pixel. No-data pixels stay zero."""
    def __init__(self, mean, components):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("w", torch.as_tensor(components, dtype=torch.float32))

    def forward(self, x):
        nodata = (x == 0).all(1, keepdim=True)
        z = torch.einsum("kc,bchw->bkhw", self.w, x - self.mean)
        return z.masked_fill(nodata, 0.0)


class LearnedProject(torch.nn.Sequential):
    """Learned 1x1 projection 64 -> k (linear + BatchNorm). Unlike PCA it can keep the
    directions that matter for the labels rather than the ones with most variance."""
    def __init__(self, c_in, k):
        super().__init__(torch.nn.Conv2d(c_in, k, 1, bias=False), torch.nn.BatchNorm2d(k))


def receptive_field(dilations):
    """Pixels seen by each output pixel: 1x1 input conv, then each 3x3 block adds 2*d."""
    return 1 + 2 * sum(dilations)


class TinyFCN(torch.nn.Module):
    """Small dilated residual CNN, ~190k params at width 64 with the default dilations
    (receptive field 63 px = 630 m). Baseline for whether SegFormer's capacity buys anything over
    local context. A dilation of 0 makes that block a 1x1 conv (per-pixel, no spatial context),
    so receptive field can be swept at fixed depth; all-zero dilations = a per-pixel MLP."""
    def __init__(self, c_in, n_classes, width=64, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.receptive_field = receptive_field(self.dilations)
        self.inp = torch.nn.Sequential(torch.nn.Conv2d(c_in, width, 1, bias=False),
                                       torch.nn.BatchNorm2d(width), torch.nn.ReLU(inplace=True))
        def block(d):
            conv = (torch.nn.Conv2d(width, width, 1, bias=False) if d == 0 else
                    torch.nn.Conv2d(width, width, 3, padding=d, dilation=d, bias=False))
            return torch.nn.Sequential(conv, torch.nn.BatchNorm2d(width), torch.nn.ReLU(inplace=True))
        self.blocks = torch.nn.ModuleList(block(d) for d in self.dilations)
        self.drop = torch.nn.Dropout2d(0.1)
        self.head = torch.nn.Conv2d(width, n_classes, 1)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.head(self.drop(h))


class PixelMLP(torch.nn.Sequential):
    """Per-pixel MLP: 1x1 convs = the same small MLP applied to every pixel's 64-d embedding.
    No spatial context, no normalization layers. Default 64 -> 64 -> 11 is ~4.9k params; small
    enough to run as matrix ops on array images inside Earth Engine."""
    def __init__(self, c_in, n_classes, hidden=(64,)):
        layers, c = [], c_in
        for h in hidden:
            layers += [torch.nn.Conv2d(c, h, 1), torch.nn.ReLU(inplace=True)]
            c = h
        super().__init__(*layers, torch.nn.Conv2d(c, n_classes, 1))
        self.hidden = tuple(hidden)


class Net(torch.nn.Module):
    def __init__(self, pre, core):
        super().__init__()
        self.pre, self.core = pre, core

    def forward(self, x):
        return self.core(self.pre(x))


def fit_pca(paths, k, n_px=200_000, seed=0):
    """PCA on valid pixels sampled evenly from the given (training) chips."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    per = max(1, n_px // len(paths))
    samples = []
    for p in paths:
        a = np.load(p)["x"]
        x = a.astype(np.float32).reshape(a.shape[0], -1).T
        x = x[np.abs(x).sum(1) > 0]
        if len(x):
            samples.append(x[rng.choice(len(x), min(per, len(x)), replace=False)])
    pca = PCA(n_components=k, random_state=seed).fit(np.concatenate(samples))
    return pca.mean_, pca.components_, float(pca.explained_variance_ratio_.sum())


def build_model(tc, in_ch, px, train_paths):
    """Returns (model, new-parameter ids, info). New params get lr_new; pretrained get lr_backbone."""
    info = {}
    if tc.pca and tc.input_proj:
        raise ValueError("use either pca or input_proj, not both")
    if tc.pca:
        mean, comps, var = fit_pca(train_paths, tc.pca, seed=tc.seed)
        pre, k = PCAProject(mean, comps), tc.pca
        info["pca_variance"] = var
        log.info("PCA %d -> %d (fit on %d training chips): %.3f variance", in_ch, k, len(train_paths), var)
    elif tc.input_proj:
        pre, k = LearnedProject(in_ch, tc.input_proj), tc.input_proj
    else:
        pre, k = torch.nn.Identity(), in_ch
    if tc.model == "segformer":
        core = SegFormer(num_classes=N_CLASSES, input_image_size=px, weights=tc.weights,
                         in_channels=k, stem_stride=tc.stem_stride, pretrained=not tc.from_scratch)
        stem = next(m for m in core.backbone.modules() if isinstance(m, torch.nn.Conv2d))
        # from scratch: nothing is pretrained, so everything trains at lr_new
        new = [core, pre] if tc.from_scratch else [stem, core.decode_core, core.classifier, pre]
    elif tc.model == "fcn":
        if tc.from_scratch:
            log.info("--from-scratch is implied for fcn")
        core = TinyFCN(k, N_CLASSES, width=tc.fcn_width, dilations=tc.fcn_dilations)
        new = [core, pre]
        info["fcn_dilations"] = list(core.dilations)
        info["receptive_field_px"] = core.receptive_field
        info["receptive_field_m"] = core.receptive_field * 10
    elif tc.model == "mlp":
        core = PixelMLP(k, N_CLASSES, hidden=tc.mlp_hidden)
        new = [core, pre]
        info["mlp_hidden"] = list(core.hidden)
        info["receptive_field_px"], info["receptive_field_m"] = 1, 10
    else:
        raise ValueError(f"model must be segformer, fcn or mlp, got {tc.model!r}")
    model = Net(pre, core)
    new_ids = {id(p) for m in new for p in m.parameters()}
    info["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["model_in_channels"] = k
    info["pretrained"] = tc.model == "segformer" and not tc.from_scratch
    return model, new_ids, info


# ------------------------------------------------------------------ tracking
class Tracker:
    """Thin W&B wrapper; a no-op when disabled. One W&B run per split, grouped by variant."""
    def __init__(self, tc, group, name, job_type, extra_config):
        self.run = None
        if not tc.wandb:
            return
        try:
            import wandb
        except ImportError as e:
            raise SystemExit("--wandb needs `pip install -e .[track]`") from e
        cfg = {**asdict(tc), **extra_config, "variant": group, "split": name}
        g = tc.wandb_group or group
        run_name = f"{tc.wandb_name}/{name}" if tc.wandb_name else f"{group}/{name}"
        tags = [job_type, tc.model] + (["scratch"] if tc.from_scratch else []) + list(tc.wandb_tags)
        self.run = wandb.init(project=tc.wandb_project, entity=tc.wandb_entity, group=g,
                              name=run_name, job_type=job_type, config=cfg, tags=tags,
                              notes=tc.wandb_notes,
                              reinit="finish_previous")

    def log(self, d, step=None):
        if self.run:
            self.run.log(d, step=step)

    def finish(self, res):
        if not self.run:
            return
        import wandb
        summ = {"best_epoch": res["best_epoch"], "val/best_mangrove_iou": res["val_mangrove_iou"]}
        for region, m in res["test"].items():
            for k, v in m.items():
                if isinstance(v, dict):
                    summ.update({f"test/{region}/iou_{c}": x for c, x in v.items()})
                else:
                    summ[f"test/{region}/{k}"] = v
            # region-agnostic keys so the W&B group view can average across held-out regions
            if len(res["test"]) == 1:
                summ.update({f"test/{k}": v for k, v in m.items() if not isinstance(v, dict)})
        self.run.summary.update(summ)
        rows = [[r, m["mangrove_iou"], m["mangrove_f1"], m["mangrove_ap"], m["miou"]]
                for r, m in res["test"].items()]
        self.run.log({"test_table": wandb.Table(
            columns=["region", "mangrove_iou", "mangrove_f1", "mangrove_ap", "miou"], data=rows)})
        self.run.finish()


# ------------------------------------------------------------------ training
def _device(tc):
    """CUDA (NVIDIA) > MPS (Apple Metal) > CPU, unless --device is given."""
    if tc.device:
        d = torch.device(tc.device)
        if d.type == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available (needs Apple Silicon, macOS 12.3+, "
                             "and a PyTorch build with MPS: torch.backends.mps.is_built()).")
        if d.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _use_amp(tc, device):
    """fp16 autocast. Default on for CUDA; opt-in on MPS (--amp), where fp16 autocast works on
    recent PyTorch but is less battle-tested; never on CPU."""
    if device.type == "cpu":
        return False
    return (device.type == "cuda") if tc.amp is None else bool(tc.amp)


def _autocast(device, amp):
    return torch.autocast(device.type, dtype=torch.float16) if amp else contextlib.nullcontext()


def _loader(paths, tc, train, device=None):
    workers = tc.workers
    return DataLoader(ChipDataset(paths, augment=train), batch_size=tc.batch_size, shuffle=train,
                      num_workers=workers,
                      pin_memory=device is not None and device.type == "cuda",
                      drop_last=train and len(paths) > tc.batch_size,
                      persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, device, amp, return_scores=False):
    model.eval()
    m = Meter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device, amp):
            m.update(model(x), y)
    return (m.result(), *m.scores()) if return_scores else m.result()


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


def fit(train_paths, val_paths, test_sets, tc: TrainConfig, run_dir: Path, tag: str,
        job_type="loro"):
    torch.manual_seed(tc.seed); np.random.seed(tc.seed)
    device = _device(tc)
    amp = _use_amp(tc, device)
    px = np.load(train_paths[0])["x"].shape[-1]
    in_ch = np.load(train_paths[0])["x"].shape[0]
    model, new_ids, info = build_model(tc, in_ch, px, train_paths)
    model = model.to(device)
    log.info("%s: %s, %d params, model input %d channels | device %s, amp %s", tag, tc.model,
             info["params"], info["model_in_channels"], device, amp)
    info["device"], info["amp"] = str(device), amp
    tracker = Tracker(tc, run_dir.name, tag, job_type,
                      {**info, "train_chips": len(train_paths), "val_chips": len(val_paths),
                       "test_regions": sorted(test_sets)})
    params = [{"params": [p for p in model.parameters() if id(p) not in new_ids], "lr": tc.lr_backbone},
              {"params": [p for p in model.parameters() if id(p) in new_ids], "lr": tc.lr_new}]
    params = [g for g in params if g["params"]]
    opt = torch.optim.AdamW(params, weight_decay=tc.weight_decay)
    tr_loader = _loader(train_paths, tc, True, device)
    steps = tc.epochs * len(tr_loader)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    weights, counts = class_weights(train_paths)
    loss_fn = JaccardLoss(num_classes=N_CLASSES, ignore_index=IGNORE, weight=weights.to(device),
                          alpha=tc.alpha)
    # loss scaling guards fp16 underflow; CUDA has GradScaler, other backends train in fp32 by
    # default (or fp16 autocast without scaling, which is fine at these loss magnitudes)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    val_loader = _loader(val_paths, tc, False, device)

    best, best_ep, hist = -1.0, -1, []
    ckpt = run_dir / f"{tag}_best.pt"
    t0 = time.time()
    for ep in range(tc.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _autocast(device, amp):
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
        tracker.log({"epoch": ep, "train/loss": tot / max(n, 1), "lr": sched.get_last_lr()[-1],
                     **{f"val/{k}": x for k, x in v.items() if not isinstance(x, dict)},
                     **{f"val/iou_{c}": x for c, x in v["per_class_iou"].items()}}, step=ep)
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    res = dict(tag=tag, best_epoch=best_ep, val_mangrove_iou=best, history=hist,
               train_chips=len(train_paths), val_chips=len(val_paths),
               train_class_px={CLASS_NAMES[i]: int(c) for i, c in enumerate(counts)},
               model_info=info,
               config=asdict(tc), test={})
    # mangrove threshold tuned on validation pixels (training regions only), applied to test
    _, val_t, val_p = evaluate(model, val_loader, device, amp, return_scores=True)
    for name, paths in test_sets.items():
        m, tt, tp = evaluate(model, _loader(paths, tc, False, device), device, amp, return_scores=True)
        m.update(tuned_metrics(val_t, val_p, tt, tp))
        m["chips"] = len(paths)
        res["test"][name] = m
    (run_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    tracker.finish(res)
    return res


def run_name(tc, protocol):
    """Results subfolder; variants get their own so runs don't overwrite each other."""
    parts = [protocol]
    if tc.model != "segformer":
        parts.append(tc.model)
    if tc.model == "fcn" and tuple(tc.fcn_dilations) != (1, 2, 4, 8, 16):
        parts.append(f"rf{receptive_field(tc.fcn_dilations)}")
    elif tc.model == "mlp" and tuple(tc.mlp_hidden) != (64,):
        parts.append("h" + "-".join(str(h) for h in tc.mlp_hidden))
    elif tc.from_scratch and tc.model == "segformer":
        parts.append("scratch")
    if tc.pca:
        parts.append(f"pca{tc.pca}")
    if tc.input_proj:
        parts.append(f"proj{tc.input_proj}")
    if tc.name_seed:
        parts.append(f"s{tc.seed}")
    return "_".join(parts)


def run(tc: TrainConfig, protocol="loro", holdout="all", regions=None, tag=None):
    df = add_blocks(load_index(tc.chips_dir, regions), tc.block_deg)
    if tc.max_chips_per_region:
        # explicit concat: pandas 3 drops the grouping column inside groupby.apply
        df = pd.concat([g.sample(min(len(g), tc.max_chips_per_region), random_state=tc.seed)
                        for _, g in df.groupby("region")], ignore_index=True)
    all_regions = sorted(df.region.unique())
    print(df.groupby(["region", "kind"]).size().unstack(fill_value=0).to_string())
    run_dir = Path(tc.out_dir) / (tag or run_name(tc, protocol))
    print(f"results -> {run_dir}")
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
        res = fit(list(tr.path[~val]), list(tr.path[val]), tests, tc, run_dir, "pooled",
                  job_type="pooled")
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
    cols += [c for c in ("mangrove_iou_tuned", "threshold") if c in summary]
    print("\n" + summary[cols].round(3).to_string(index=False))
    print("mean:", summary[cols[1:]].mean().round(3).to_dict())
    return summary


def _flat(m):
    return {k: v for k, v in m.items() if not isinstance(v, dict)}


def _line(r):
    tuned = (f" | tuned@{r['threshold']:.2f}: IoU {r['mangrove_iou_tuned']:.3f}"
             if "threshold" in r else "")
    return (f"[{r['protocol']}] {r['region']}: mangrove F1 {r['mangrove_f1']:.3f} "
            f"IoU {r['mangrove_iou']:.3f} AP {r['mangrove_ap']:.3f} | mIoU {r['miou']:.3f}{tuned}")


def compare(out_dir="results/seg", metric="mangrove_iou", region=None):
    """Side-by-side table of every run folder's summary.csv; `region` restricts to one held-out
    region so single-holdout runs compare like-for-like with that split of full runs."""
    rows = []
    for f in sorted(Path(out_dir).glob("*/summary.csv")):
        s = pd.read_csv(f)
        s["run"] = f.parent.name
        info = {}
        js = sorted(f.parent.glob("*.json"))
        if js:
            info = json.loads(js[0].read_text()).get("model_info", {})
        s["params"] = info.get("params")
        s["rf_m"] = info.get("receptive_field_m")
        rows.append(s)
    if not rows:
        raise SystemExit(f"No summary.csv under {out_dir}.")
    df = pd.concat(rows, ignore_index=True)
    if region:
        df = df[df.region == region]
        if df.empty:
            raise SystemExit(f"No runs with held-out region {region!r}.")
    cols = ["mangrove_iou", "mangrove_f1", "mangrove_ap", "miou"]
    cols += [c for c in ("mangrove_iou_tuned", "threshold") if c in df]
    agg = df.groupby("run").agg(**{c: (c, "mean") for c in cols}, regions=("region", "nunique"),
                                params=("params", "first"), rf_m=("rf_m", "first")
                                ).sort_values(metric, ascending=False)
    print(agg.round(3).to_string())
    print(f"\n{metric} by region:")
    print(df.pivot_table(index="run", columns="region", values=metric).round(3)
            .reindex(agg.index).to_string())
    seeds = seed_summary(agg, metric)
    if seeds is not None:
        print(f"\n{metric} across seeds (mean over regions per run, then mean ± std over seeds):")
        print(seeds.round(3).to_string())
    return agg


def seed_summary(agg, metric="mangrove_iou"):
    """Group run folders that differ only by seed suffix (_s<N>; no suffix counts as a seed).
    Returns None when no variant has more than one seed."""
    a = agg.reset_index()
    a["variant"] = a.run.str.replace(r"_s\d+$", "", regex=True)
    g = a.groupby("variant").agg(mean=(metric, "mean"), std=(metric, "std"), seeds=("run", "size"),
                                 regions=("regions", "min"), params=("params", "first"),
                                 rf_m=("rf_m", "first"))
    if (g.seeds < 2).all():
        return None
    return g.sort_values("mean", ascending=False)


def bench(tc: TrainConfig, models=("mlp", "fcn", "segformer"), seeds=(0, 1, 2), holdout="all",
          regions=None, metric="mangrove_iou"):
    """Same LORO splits, several seeds per model, then the seed-aggregated comparison.
    Each (model, seed) writes to its usual folder (loro_mlp_s0, ...); existing complete runs are
    skipped, so an interrupted bench resumes."""
    from dataclasses import replace
    done = []
    for m in models:
        for sd in seeds:
            t = replace(tc, model=m, seed=sd, name_seed=True,
                        wandb_name=" ".join(x for x in (tc.wandb_name, m, f"s{sd}") if x))
            folder = Path(t.out_dir) / run_name(t, "loro")
            want = None if holdout == "all" else {holdout}
            if (folder / "summary.csv").exists():
                have = set(pd.read_csv(folder / "summary.csv").region)
                if (want and want <= have) or (not want and len(have) >= 2 and
                        have >= set(load_index(t.chips_dir, regions).region.unique())):
                    print(f"skip {folder.name} (done)")
                    done.append(folder.name)
                    continue
            run(t, "loro", holdout, regions)
            done.append(folder.name)
    agg = compare(tc.out_dir, metric, None if holdout == "all" else holdout)
    return agg.loc[agg.index.isin(done)]
