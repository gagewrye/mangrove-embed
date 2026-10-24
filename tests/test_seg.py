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
    tc.max_chips_per_region = 3                       # subsampling path (pandas 3 regression)
    s3 = run(tc, "loro", "a", tag="sub")
    assert set(s3.region) == {"a"}
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


def test_from_scratch_is_random_but_same_architecture():
    import json
    pre = SegFormer(num_classes=11, input_image_size=64, weights=W, in_channels=64)
    scr = SegFormer(num_classes=11, input_image_size=64, weights=W, in_channels=64, pretrained=False)
    a, b = pre.backbone.state_dict(), scr.backbone.state_dict()
    assert a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)
    stem = [k for k in a if a[k].dim() == 4][0]
    differ = [k for k in a if k != stem and a[k].dim() > 1 and not torch.equal(a[k], b[k])]
    assert len(differ) > 10   # weights not loaded


def test_from_scratch_run_folder(tmp_path):
    import json
    from mangrove_embed.seg.train import TrainConfig, run
    _synthetic_chips(tmp_path / "chips", n=4)
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                     epochs=1, batch_size=4, workers=0, block_deg=0.3, device="cpu", from_scratch=True)
    run(tc, "loro", "a")
    info = json.loads((tmp_path / "out" / "loro_scratch" / "holdout_a.json").read_text())["model_info"]
    assert info["pretrained"] is False
    from mangrove_embed.seg.train import compare
    agg = compare(str(tmp_path / "out"))
    assert "loro_scratch" in agg.index
    assert (compare(str(tmp_path / "out"), region="a").regions == 1).all()


def test_wandb_offline(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from mangrove_embed.seg.train import TrainConfig, run
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    _synthetic_chips(tmp_path / "chips", n=4)
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                     epochs=2, batch_size=4, workers=0, block_deg=0.3, device="cpu", wandb=True,
                     model="fcn", wandb_name="rf63-test", wandb_group="rf-sweep",
                     wandb_tags=("sweep",))
    run(tc, "loro", "a")
    runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(runs) == 1
    blob = next(runs[0].glob("*.wandb")).read_bytes()
    assert b"rf63-test/holdout_a" in blob and b"rf-sweep" in blob and b"sweep" in blob


def test_device_and_amp_selection(monkeypatch):
    from mangrove_embed.seg.train import TrainConfig, _device, _use_amp
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    tc = TrainConfig()
    assert _device(tc).type == "mps"                       # Apple Metal picked when no CUDA
    assert _use_amp(tc, torch.device("mps")) is False      # off by default on MPS
    assert _use_amp(TrainConfig(amp=True), torch.device("mps")) is True
    assert _use_amp(tc, torch.device("cuda")) is True      # on by default on CUDA
    assert _use_amp(TrainConfig(amp=True), torch.device("cpu")) is False
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _device(tc).type == "cpu"
    with pytest.raises(SystemExit):
        _device(TrainConfig(device="mps"))


def test_fcn_dilation_sweep(tmp_path):
    import json
    from mangrove_embed.seg.train import TinyFCN, TrainConfig, receptive_field, run
    assert receptive_field((1, 2, 4, 8, 16)) == 63 and receptive_field((0,) * 5) == 1
    x = torch.randn(1, 64, 32, 32)
    pix = TinyFCN(64, 11, dilations=(0,) * 5).eval()
    x2 = x.clone(); x2[..., 0, 0] += 5.0           # perturb one pixel
    with torch.no_grad():
        d = (pix(x) - pix(x2)).abs().sum(1)[0]
    assert d[0, 0] > 0 and d[1:, :].sum() == 0 and d[0, 1:].sum() == 0   # no spatial mixing
    _synthetic_chips(tmp_path / "chips", n=4)
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                     epochs=1, batch_size=4, workers=0, block_deg=0.3, device="cpu", model="fcn",
                     fcn_dilations=(1, 1, 1, 1, 1), seed=3, name_seed=True)
    run(tc, "loro", "a")
    info = json.loads((tmp_path / "out" / "loro_fcn_rf11_s3" / "holdout_a.json").read_text())["model_info"]
    assert info["receptive_field_px"] == 11 and info["receptive_field_m"] == 110


def test_thresholds():
    import numpy as np
    from mangrove_embed.thresholds import binary_scores, tune_threshold
    t = np.array([1, 1, 1, 0, 0, 0], bool)
    p = np.array([0.9, 0.8, 0.35, 0.3, 0.2, 0.1])
    thr, iou = tune_threshold(t, p)
    assert 0.3 < thr <= 0.35 and iou == 1.0
    assert binary_scores(t, p, 0.5)["iou"] == 2 / 3


def test_compare_pixel_and_tuned_seg(tmp_path):
    import json
    from mangrove_embed import pixel
    from mangrove_embed.seg.train import TrainConfig, compare, run
    _synthetic_chips(tmp_path / "chips", n=6)
    out = tmp_path / "out"
    res = pixel.run(str(tmp_path / "chips"), str(out), ("knn", "logreg", "mlp"), "all",
                    n_train_per_region=500, n_val=300, n_test=800, block_deg=0.3)
    for m in ("knn", "logreg", "mlp"):
        s = res[m]
        assert set(s.region) == {"a", "b", "c"}
        assert s.mangrove_iou_tuned.between(0, 1).all() and s.threshold.between(0, 1).all()
    info = json.loads((out / "pixel_logreg" / "holdout_a.json").read_text())["model_info"]
    assert info["params"] == 64 * 2 + 2 or info["params"] > 0
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(out), weights=W, epochs=1,
                     batch_size=4, workers=0, block_deg=0.3, device="cpu", model="fcn")
    s = run(tc, "loro", "a")
    assert {"mangrove_iou_tuned", "threshold"} <= set(s.columns)
    agg = compare(str(out), region="a")
    assert {"pixel_knn", "pixel_logreg", "pixel_mlp", "loro_fcn"} <= set(agg.index)
    assert "mangrove_iou_tuned" in agg.columns


def test_pixel_mlp_and_bench(tmp_path):
    import json
    import torch
    from mangrove_embed.seg.train import PixelMLP, TrainConfig, bench
    m = PixelMLP(64, 11)
    assert sum(p.numel() for p in m.parameters()) == 64 * 64 + 64 + 64 * 11 + 11
    x = torch.randn(2, 64, 8, 8)
    y = m(x)
    assert y.shape == (2, 11, 8, 8)
    # per-pixel: shuffling pixels shuffles outputs identically (no spatial context)
    perm = torch.randperm(64)
    xs = x.flatten(2)[..., perm].view_as(x)
    assert torch.allclose(m(xs).flatten(2), y.flatten(2)[..., perm], atol=1e-6)
    _synthetic_chips(tmp_path / "chips", n=4)
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                     epochs=1, batch_size=4, workers=0, block_deg=0.3, device="cpu")
    agg = bench(tc, models=("mlp", "fcn"), seeds=(0, 1), holdout="a")
    assert set(agg.index) == {"loro_mlp_s0", "loro_mlp_s1", "loro_fcn_s0", "loro_fcn_s1"}
    info = json.loads((tmp_path / "out" / "loro_mlp_s0" / "holdout_a.json").read_text())["model_info"]
    assert info["receptive_field_m"] == 10 and info["mlp_hidden"] == [64]
    bench(tc, models=("mlp",), seeds=(0,), holdout="a")   # resumes: skips the finished run


def test_seed_summary():
    import pandas as pd
    from mangrove_embed.seg.train import seed_summary
    agg = pd.DataFrame(dict(mangrove_iou=[0.8, 0.82, 0.7], regions=[4, 4, 4], params=[1, 1, 2],
                            rf_m=[10, 10, 630]), index=pd.Index(["loro_mlp_s0", "loro_mlp_s1", "loro_fcn"], name="run"))
    s = seed_summary(agg)
    assert s.loc["loro_mlp", "seeds"] == 2 and abs(s.loc["loro_mlp", "mean"] - 0.81) < 1e-9
    assert seed_summary(agg.iloc[[0, 2]]) is None


def test_context_pool_masked_mean():
    import torch
    from mangrove_embed.seg.train import ContextPool
    x = torch.zeros(1, 2, 5, 5)
    x[0, :, 1:4, 1:4] = 1.0          # 3x3 valid block of ones, no-data elsewhere
    x[0, :, 2, 2] = 10.0             # center
    out = ContextPool((1,))(x)
    assert out.shape == (1, 4, 5, 5)
    assert torch.allclose(out[0, :2], x[0])                      # first block = raw pixel
    assert torch.isclose(out[0, 2, 2, 2], torch.tensor(18 / 9))  # (8*1 + 10)/9
    assert torch.isclose(out[0, 2, 1, 1], torch.tensor(13 / 4))  # corner: 4 valid px, no-data ignored


def test_mlp_context_run(tmp_path):
    import json
    from mangrove_embed.seg.train import TrainConfig, run
    _synthetic_chips(tmp_path / "chips", n=4)
    tc = TrainConfig(chips_dir=str(tmp_path / "chips"), out_dir=str(tmp_path / "out"), weights=W,
                     epochs=1, batch_size=4, workers=0, block_deg=0.3, device="cpu", model="mlp",
                     mlp_hidden=(128, 64), mlp_norm=True, mlp_dropout=0.1, mlp_context=(1, 3))
    s = run(tc, "loro", "a")
    assert s.mangrove_iou.between(0, 1).all()
    info = json.loads((tmp_path / "out" / "loro_mlp_h128-64_bn_do0.1_ctx1-3" / "holdout_a.json")
                      .read_text())["model_info"]
    assert info["receptive_field_m"] == 70 and info["mlp_context"] == [1, 3]
    assert info["params"] == (64 * 3 * 128 + 128 * 2) + (128 * 64 + 64 * 2) + (64 * 11 + 11)
