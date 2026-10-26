import numpy as np
import pandas as pd
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from mangrove_embed import BANDS  # noqa: E402

YEARS = np.arange(2017, 2026)


def _temporal(tmp_path):
    """Synthetic temporal chips (as test_temporal.test_change_end_to_end)."""
    rng = np.random.default_rng(0)
    for reg in ("a", "b"):
        d = tmp_path / reg
        d.mkdir()
        rows = []
        for c in range(6):
            x = rng.normal(size=(len(YEARS), 64, 6, 6))
            x[:, :11] += 3 * rng.normal(size=(1, 11, 6, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y = (x[:, :11].argmax(1) + 1).astype(np.uint8)
            y[YEARS > 2020] = 255
            np.savez(d / f"chip_{c:04d}.npz", x=x.astype(np.float16), y=y, years=YEARS.astype(np.int16))
            rows.append(dict(chip=f"chip_{c:04d}", kind="random", x0=0, y1=0, lon=0.15 * c + 0.01,
                             lat=0.01, region=reg, crs="EPSG:32618"))
        pd.DataFrame(rows).to_csv(d / "index.csv", index=False)


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    from mangrove_embed.change import ChangeConfig, run
    tmp = tmp_path_factory.mktemp("change")
    _temporal(tmp)
    path = tmp / "bundle.pt"
    run(ChangeConfig(temporal_dir=str(tmp), out_dir=str(tmp / "out"), px_per_chip=20, epochs=2,
                     batch_size=64, hidden=(16,), device="cpu", history=2, labels="logreg",
                     methods=("persistence", "classify_current", "next_class"), save_bundle=str(path)))
    return path


def _write(path, x, transform=from_origin(500000, 2000000, 10, 10), names=BANDS):
    """x [64, h, w] -> GeoTIFF with band descriptions."""
    with rasterio.open(path, "w", driver="GTiff", width=x.shape[2], height=x.shape[1], count=len(x),
                       dtype="float32", crs="EPSG:32618", transform=transform) as ds:
        ds.write(x.astype(np.float32))
        ds.descriptions = tuple(names)
    return path


def _rasters(tmp_path, h=37, w=53):
    """Two years of unit-norm embeddings with no-data holes (zeros in year 1, NaN in year 2)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 64, h, w))
    x[:, :11] += 3 * rng.normal(size=(1, 11, h, w))
    x[1, :11, :10, :10] += 4 * rng.normal(size=(11, 10, 10))      # some pixels change
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x[0, :, 0, :5] = 0
    x[1, :, -1, -3:] = np.nan
    return x, [_write(tmp_path / f"e{i}.tif", x[i]) for i in range(2)]


def test_bundle_roundtrip_and_selfcheck(bundle_path, tmp_path):
    from mangrove_embed.deploy import Bundle
    b = Bundle.load(bundle_path, "cpu")
    assert b.history == 2 and b.names[2] == "tidal_flat_water"
    assert b.meta["test_year"] == 2025 and "change_ap" in b.meta["test"]
    f = b.forecast(torch.randn(10, 2, 64))
    assert f["probs"].shape == (10, 11) and np.allclose(f["probs"].sum(1), 1, atol=1e-5)
    assert (f["probs"][:, 3] < 1e-6).all() and (f["next"] != 3).all()   # water merged away
    with pytest.raises(ValueError):
        b.forecast(torch.randn(10, 3, 64))
    d = torch.load(bundle_path, weights_only=True)                      # tampered weights
    d["forecaster"]["state"]["cls.bias"] += 1.0
    with pytest.raises(ValueError, match="self-check"):
        Bundle(d)


def test_save_bundle_needs_logreg(tmp_path):
    from mangrove_embed.change import ChangeConfig, run
    _temporal(tmp_path)
    with pytest.raises(SystemExit):
        run(ChangeConfig(temporal_dir=str(tmp_path), labels="classifier", save_bundle=str(tmp_path / "b.pt")))


def test_predict_rasters_matches_bundle(bundle_path, tmp_path):
    from mangrove_embed.deploy import Bundle, predict_rasters
    x, paths = _rasters(tmp_path)
    p_cls, p_prob = predict_rasters(bundle_path, paths, tmp_path / "f", block=16, probs=True, device="cpu")
    with rasterio.open(p_cls) as c, rasterio.open(p_prob) as p:
        cls, pr = c.read(), p.read()
        assert c.transform == from_origin(500000, 2000000, 10, 10) and c.crs.to_epsg() == 32618
        assert p.descriptions[:2] == ("p_change", "p_mangrove") and p.count == 13
    ok = np.isfinite(x).all((0, 1)) & (np.abs(np.nan_to_num(x)).sum(1) > 0).all(0)
    assert (~ok).sum() == 8
    assert (cls[:, ~ok] == 255).all() and np.isnan(pr[:, ~ok]).all()
    f = Bundle.load(bundle_path, "cpu").forecast(torch.from_numpy(x[:, :, ok].transpose(2, 0, 1).copy()).float())
    assert (cls[0][ok] == f["current"] + 1).all() and (cls[1][ok] == f["next"] + 1).all()
    assert np.allclose(pr[0][ok], f["p_change"], atol=1e-5) and np.allclose(pr[2:, ok].T, f["probs"], atol=1e-5)


def test_band_order_by_description(bundle_path, tmp_path):
    from mangrove_embed.deploy import predict_rasters
    x, paths = _rasters(tmp_path)
    perm = np.random.default_rng(2).permutation(64)
    shuffled = [_write(tmp_path / f"s{i}.tif", x[i][perm], names=[BANDS[j] for j in perm]) for i in range(2)]
    a = predict_rasters(bundle_path, paths, tmp_path / "a", block=16, device="cpu")[1]
    b = predict_rasters(bundle_path, shuffled, tmp_path / "b", block=16, device="cpu")[1]
    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        assert np.array_equal(ra.read(), rb.read(), equal_nan=True)


def test_detect_rasters(bundle_path, tmp_path):
    from mangrove_embed.deploy import Bundle, detect_rasters
    x, paths = _rasters(tmp_path)
    p_cls, p_prob, p_tr, p_cl = detect_rasters(bundle_path, paths[0], paths[1], tmp_path / "d",
                                               block=16, min_p=0.5, device="cpu")
    with rasterio.open(p_cls) as c, rasterio.open(p_prob) as p:
        cls, pr = c.read(), p.read()
        assert p.descriptions == ("p_change", "p_mangrove_gain", "p_mangrove_loss", "cosine")
    ok = cls[0] != 255
    assert ok.sum() == 37 * 53 - 8
    b = Bundle.load(bundle_path, "cpu")
    d = b.detect(x[0][:, ok].T, x[1][:, ok].T)
    assert (cls[0][ok] == d["before"] + 1).all() and (cls[1][ok] == d["after"] + 1).all()
    assert np.allclose(pr[3][ok], (x[0][:, ok] * x[1][:, ok]).sum(0), atol=1e-5)   # unit vectors
    assert ((pr[0][ok] >= 0) & (pr[0][ok] <= 1)).all()
    tr, cl = pd.read_csv(p_tr), pd.read_csv(p_cl)
    sel = (d["before"] != d["after"]) & (d["p_change"] >= 0.5)
    assert tr.px.sum() == sel.sum() > 0
    assert np.allclose(tr.ha, tr.px * 0.01)                                          # 10 m pixels
    assert cl.before_px.sum() == cl.after_px.sum() == ok.sum()
    assert cl.gain_px.sum() == cl.loss_px.sum() == sel.sum()
    # the same year twice: nothing changes
    *_, p_tr2, _ = detect_rasters(b, paths[0], paths[0], tmp_path / "same", block=16)
    assert pd.read_csv(p_tr2).empty


def test_grid_mismatch(bundle_path, tmp_path):
    from mangrove_embed.deploy import predict_rasters
    x, paths = _rasters(tmp_path)
    moved = _write(tmp_path / "m.tif", x[1], transform=from_origin(500010, 2000000, 10, 10))
    with pytest.raises(ValueError, match="same grid"):
        predict_rasters(bundle_path, [paths[0], moved], tmp_path / "f", device="cpu")
