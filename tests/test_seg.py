import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mangrove_embed.seg import JaccardLoss, SegFormer  # noqa: E402

W = "nvidia/segformer-b0-finetuned-ade-512-512"


@pytest.fixture(scope="module")
def emb():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 64, 64)
    return x / x.norm(dim=1, keepdim=True)  # unit-length, like AlphaEarth


def _stem(m):
    return next(x for x in m.backbone.modules() if isinstance(x, torch.nn.Conv2d))


def test_native_64ch_keeps_pretrained_and_trains(emb):
    ref = SegFormer(num_classes=11, input_image_size=64, weights=W)
    m = SegFormer(num_classes=11, input_image_size=64, weights=W, in_channels=64)
    assert _stem(m).in_channels == 64
    a, b = ref.backbone.state_dict(), m.backbone.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if a[k].shape == b[k].shape)
    out = m(emb)
    assert out.shape == (2, 11, 64, 64)
    loss = JaccardLoss(num_classes=11)(out, torch.randint(0, 11, (2, 64, 64)))
    loss.backward()
    assert _stem(m).weight.grad.abs().sum() > 0


def test_binary_mean_init_and_stride(emb):
    m = SegFormer(num_classes=1, input_image_size=64, weights=W, in_channels=64,
                  stem_init="mean", stem_stride=2)
    assert _stem(m).stride == (2, 2)
    out = m(emb)
    assert out.shape == (2, 1, 64, 64)
    assert torch.isfinite(JaccardLoss(num_classes=1)(out, torch.randint(0, 2, (2, 64, 64))))


def test_channel_mismatch_raises():
    m = SegFormer(num_classes=1, input_image_size=64, weights=W, in_channels=64)
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 64, 64))


def _synthetic_chips(root, regions=("a", "b", "c"), n=6, px=32, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(11, 64))
    for ri, r in enumerate(regions):
        d = root / r
        d.mkdir(parents=True)
        shift = rng.normal(size=64) * 0.5
        rows = []
        for i in range(n):
            yy, xx = np.mgrid[:px, :px]
            cx, cy = rng.integers(0, px, 2)
            y = np.where((xx - cx) ** 2 + (yy - cy) ** 2 < (px // 3) ** 2, 1, 4).astype(np.uint8)
            y[:2] = 255
            cls = np.where(y == 255, 4, y) - 1
            x = dirs[cls].transpose(2, 0, 1) + shift[:, None, None] + rng.normal(size=(64, px, px)) * 0.5
            x = (x / np.linalg.norm(x, axis=0, keepdims=True)).astype(np.float16)
            np.savez_compressed(d / f"chip_{i:04d}.npz", x=x, y=y)
            rows.append(dict(chip=f"chip_{i:04d}", kind="mangrove", x0=0, y1=0,
                             lon=ri * 10 + i * 0.2, lat=i * 0.2, region=r, crs="EPSG:32601"))
        pd.DataFrame(rows).to_csv(d / "index.csv", index=False)


def test_train_seg_loro_and_pooled(tmp_path):
    import json
    from mangrove_embed.seg.train import TrainConfig, run
    _synthetic_chips(tmp_path / "chips")
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                     epochs=2, batch_size=4, workers=0, block_deg=0.3, device="cpu")
    s = run(tc, "loro", "all")
    assert set(s.region) == {"a", "b", "c"} and s.mangrove_iou.between(0, 1).all()
    res = json.loads((tmp_path / "out" / "loro" / "holdout_a.json").read_text())
    assert res["best_epoch"] >= 0 and "mangrove" in res["test"]["a"]["per_class_iou"]
    p = run(tc, "pooled")
    assert len(p) >= 1


def test_chip_helpers():
    import numpy as np
    from mangrove_embed import BANDS
    from mangrove_embed.chips import IGNORE, dedupe, snap_origin, to_arrays
    kept = dedupe(np.array([[0, 0], [10, 0], [100, 0]]), 50)
    assert kept.tolist() == [[0, 0], [100, 0]]
    x0, y1 = snap_origin(1003, 2007, 0, 0, 128)
    assert x0 % 10 == 0 and y1 % 10 == 0 and abs(x0 + 640 - 1003) <= 10
    dt = [(b, "f4") for b in BANDS] + [("class", "f4"), ("valid", "f4")]
    s = np.zeros((4, 4), dtype=dt)
    s["class"] = 1; s["valid"] = 1; s["class"][0, 0] = 255; s["valid"][1, 1] = 0
    x, y = to_arrays(s)
    assert x.shape == (64, 4, 4) and y[0, 0] == IGNORE and y[1, 1] == IGNORE and y[2, 2] == 1


def test_plan_index_topup(monkeypatch):
    import numpy as np
    from mangrove_embed import chips, config
    cfg = config.load(None)
    cfg.chip_px, cfg.chip_n_mangrove, cfg.chip_n_random = 16, 4, 3
    rng = np.random.default_rng(0)
    # candidates spread ~2 km apart around Jamaica, so dedupe (80 m) keeps all
    def fake(bbox, cfg_, labels, coastal, proj, n_m, n_r, seed):
        lon = -77 + rng.random(n_m + n_r) * 0.5
        lat = 18 + rng.random(n_m + n_r) * 0.5
        pts = list(zip(lon, lat))
        return pts[:n_m], pts[n_m:]
    monkeypatch.setattr(chips, "chip_centers", fake)
    args = ("jamaica", None, cfg, None, None, None, "EPSG:32618", 0.0, 0.0)
    idx, changed = chips.plan_index(*args)
    assert changed and idx.kind.value_counts().to_dict() == {"mangrove": 4, "random": 3}
    short = idx[idx.kind == "random"].copy()           # simulate a region missing mangrove chips
    short = short.assign(chip=["chip_0000", "chip_0001", "chip_0005"])
    idx2, changed2 = chips.plan_index(*args, index=short)
    assert changed2 and (idx2.kind == "mangrove").sum() == 4 and (idx2.kind == "random").sum() == 3
    assert idx2.chip.iloc[:3].tolist() == ["chip_0000", "chip_0001", "chip_0005"]   # kept
    assert idx2.chip.iloc[3] == "chip_0006" and idx2.chip.is_unique
    assert not chips.plan_index(*args, index=idx2)[1]                                # complete


def test_variants_pca_proj_fcn(tmp_path):
    import json
    import numpy as np
    from mangrove_embed.seg.train import PCAProject, TrainConfig, run
    _synthetic_chips(tmp_path / "chips", n=4)
    base = dict(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                epochs=1, batch_size=4, workers=0, block_deg=0.3, device="cpu")
    for kw, folder in [(dict(pca=16), "loro_pca16"), (dict(input_proj=16), "loro_proj16"),
                       (dict(model="fcn"), "loro_fcn"), (dict(model="fcn", pca=16), "loro_fcn_pca16")]:
        s = run(TrainConfig(**base, **kw), "loro", "a")
        assert s.mangrove_iou.between(0, 1).all()
        info = json.loads((tmp_path / "out" / folder / "holdout_a.json").read_text())["model_info"]
        assert info["model_in_channels"] == (16 if ("pca" in kw or "input_proj" in kw) else 64)
        if "pca" in kw:
            assert 0 < info["pca_variance"] <= 1
    fcn = json.loads((tmp_path / "out" / "loro_fcn" / "holdout_a.json").read_text())["model_info"]
    assert fcn["params"] < 300_000
    # PCA keeps no-data pixels at zero
    p = PCAProject(np.ones(64), np.eye(64)[:8])
    x = torch.zeros(1, 64, 2, 2); x[0, :, 0, 0] = 2.0
    z = p(x)
    assert z.shape == (1, 8, 2, 2) and torch.all(z[0, :, 1, 1] == 0) and torch.allclose(z[0, :, 0, 0], torch.ones(8))
