import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mangrove_embed.change import (IGNORE, N_CLASSES, ChangeConfig, HazardForecaster, change_loss,
                                   flicker, variant)


def test_variant_names():
    assert variant(ChangeConfig()) is None
    cc = ChangeConfig(loss="jaccard", change_weight=5, mask_uncertain=0.6, mask_flicker=True)
    assert variant(cc) == "jaccard_cw5_conf0.6_noflicker"
    assert variant(ChangeConfig(loss="jaccard", jaccard_alpha=0.2)) == "jaccard0.2"


def test_flicker_only_masks_reverting_change():
    # columns = years; target column 1 for every pixel
    y = np.array([[0, 1, 0],          # change that reverts: flicker
                  [0, 1, 1],          # durable change
                  [0, 0, 1],          # stable this year
                  [0, 1, IGNORE],     # can't tell: kept
                  [IGNORE, 1, IGNORE]])
    P = np.c_[np.arange(5), np.ones(5, int)]
    assert flicker(y, P, allowed=[1, 2]).tolist() == [True, False, False, False, False]
    assert not flicker(y, P, allowed=[1]).any()   # year after is not a training year: not read


def _batch():
    torch.manual_seed(0)
    logits = torch.randn(64, N_CLASSES, requires_grad=True)
    prev = torch.randint(0, N_CLASSES, (64,))
    tgt = prev.clone()
    tgt[:8] = (prev[:8] + 1) % N_CLASSES       # 8 changes
    tgt[-4:] = IGNORE
    return logits, tgt, prev


def test_change_loss_ce_matches_cross_entropy():
    logits, tgt, prev = _batch()
    got = change_loss(logits, tgt, prev, ChangeConfig())
    assert torch.allclose(got, F.cross_entropy(logits, tgt, ignore_index=IGNORE))
    # upweighting change moves the loss toward the changing pairs' CE
    ce = F.cross_entropy(logits[:60], tgt[:60], reduction="none")
    w = torch.ones(60)
    w[:8] = 5
    assert torch.allclose(change_loss(logits, tgt, prev, ChangeConfig(change_weight=5)), (ce * w).sum() / w.sum())


def test_jaccard_loss_rewards_correct_change():
    _, tgt, prev = _batch()
    cc = ChangeConfig(loss="jaccard")
    good = F.one_hot(tgt.clamp(max=N_CLASSES - 1), N_CLASSES).float() * 20
    stay = F.one_hot(prev, N_CLASSES).float() * 20             # persistence: misses every change
    lg, ls = change_loss(good, tgt, prev, cc), change_loss(stay, tgt, prev, cc)
    assert lg < 1e-3 and ls > lg + 0.2
    logits = torch.randn(64, N_CLASSES, requires_grad=True)
    change_loss(logits, tgt, prev, cc).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    # all targets ignored: zero loss that still backpropagates
    none = torch.full_like(tgt, IGNORE)
    assert change_loss(logits, none, prev, cc).item() == 0


def test_hazard_probabilities():
    torch.manual_seed(0)
    net = HazardForecaster(2, (16,), 0.0, rate=0.1).eval()
    xh, prev = torch.randn(32, 2, 64), torch.randint(0, N_CLASSES, (32,))
    logp, e = net(xh, prev)
    p = logp.exp()
    assert e is None and torch.allclose(p.sum(1), torch.ones(32), atol=1e-5)
    h = net.change(net.body(xh.flatten(1)))[:, 0]
    assert torch.allclose(p[torch.arange(32), prev], 1 - torch.sigmoid(h), atol=1e-5)
    assert torch.allclose(logp.softmax(1), p, atol=1e-5)         # usable as logits


def _data(tmp_path):
    rng = np.random.default_rng(0)
    years = np.arange(2017, 2026)
    for reg in ("a", "b"):
        d = tmp_path / reg
        d.mkdir()
        rows = []
        for c in range(6):
            x = rng.normal(size=(len(years), 64, 6, 6))
            x[:, :11] += 3 * rng.normal(size=(1, 11, 6, 6))   # class signal shared across years
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y = (x[:, :11].argmax(1) + 1).astype(np.uint8)
            y[years > 2020] = 255
            np.savez(d / f"chip_{c:04d}.npz", x=x.astype(np.float16), y=y, years=years.astype(np.int16))
            rows.append(dict(chip=f"chip_{c:04d}", kind="random", x0=0, y1=0, lon=0.15 * c + 0.01,
                             lat=0.01, region=reg, crs="EPSG:32618"))
        pd.DataFrame(rows).to_csv(d / "index.csv", index=False)


def test_variants_end_to_end(tmp_path):
    from mangrove_embed.change import run
    _data(tmp_path)
    base = dict(temporal_dir=str(tmp_path), out_dir=str(tmp_path / "out"), px_per_chip=20, epochs=2,
                batch_size=64, hidden=(16,), device="cpu", history=2, labels="logreg",
                methods=("classify_current", "next_class", "next_class_hazard"))
    s0 = run(ChangeConfig(**base))
    assert set(s0.method) == {"classify_current", "next_class", "next_class_hazard"}
    v = ChangeConfig(**base, loss="jaccard", change_weight=3, mask_uncertain=0.5, mask_flicker=True)
    s1 = run(v)
    d0 = tmp_path / "out" / "change_logreg_k2_y2025_s0"
    d1 = tmp_path / "out" / "change_logreg_k2_y2025_jaccard_cw3_conf0.5_noflicker_s0"
    assert (d0 / "change.json").exists() and (d1 / "change.json").exists()   # baseline untouched
    p0, p1 = (json.loads((d / "change.json").read_text())["pairs"] for d in (d0, d1))
    assert p0["train_masked"] == {} and p0["train"] == p0["train_before_masks"]
    m = p1["train_masked"]
    assert m["uncertain"] > 0 and p1["train"] == p1["train_before_masks"] - m["uncertain"] - m["flicker"]
    # masks touch training only: every variant is scored on the same test pixels
    a0, a1 = (s[s.region == "all"].set_index("method") for s in (s0, s1))
    assert (a0.px == a1.px).all() and (a0.changed_px == a1.changed_px).all()
    assert np.isfinite(a1.loc[["next_class", "next_class_hazard"], "change_ap"]).all()
