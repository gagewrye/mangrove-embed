import numpy as np
import pandas as pd

from mangrove_embed.temporal import add_change_rows, stack_years


def test_stack_years():
    per_year = [(np.full((64, 4, 4), i, np.float16), np.full((4, 4), i, np.uint8)) for i in range(3)]
    d = stack_years(per_year, [2018, 2019, 2020])
    assert d["x"].shape == (3, 64, 4, 4) and d["y"].shape == (3, 4, 4)
    assert d["x"][2, 0, 0, 0] == 2 and list(d["years"]) == [2018, 2019, 2020]


def test_add_change_rows_dedupes_and_snaps():
    px = 128                                       # 1280 m chip, min spacing 640 m
    index = pd.DataFrame(dict(chip=["chip_0000", "chip_0001"], kind=["mangrove", "random"],
                              x0=[500000.0, 510000.0], y1=[2000000.0, 2000000.0],
                              lon=[0.0, 0.0], lat=[0.0, 0.0], region="r", crs="EPSG:32618"))
    centers = np.array([[500640.0, 1999360.0],     # on chip_0000's center: dropped
                        [505003.0, 1999000.0],     # kept
                        [505100.0, 1999100.0],     # too close to the previous one: dropped
                        [520000.0, 1990000.0]])    # kept, but n=1 caps it
    out = add_change_rows(index, centers, 1, px, 0.0, 0.0, "r", "EPSG:32618")
    new = out[out.kind == "change"]
    assert len(new) == 1 and new.chip.iloc[0] == "chip_0002"
    assert new.x0.iloc[0] % 10 == 0 and new.y1.iloc[0] % 10 == 0
    assert abs(new.x0.iloc[0] + 640 - 505003) <= 5


def test_score_change_metrics():
    from mangrove_embed.change import score
    prev = np.array([0, 0, 1, 1, 255])
    tgt = np.array([0, 1, 1, 0, 0])               # one loss (idx 1), one gain (idx 3)
    s = score(prev.copy(), tgt, prev)             # persistence
    assert s["px"] == 4 and s["changed_px"] == 2 and s["changed_acc"] == 0 and s["change_iou"] == 0
    s = score(tgt.copy(), tgt, prev)              # perfect
    assert s["changed_acc"] == 1 and s["gain_iou"] == 1 and s["loss_iou"] == 1 and s["acc"] == 1


def test_change_end_to_end(tmp_path):
    from mangrove_embed.change import METHODS, ChangeConfig, run
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
    cc = ChangeConfig(temporal_dir=str(tmp_path), out_dir=str(tmp_path / "out"), px_per_chip=20,
                      epochs=2, batch_size=64, hidden=(16,), device="cpu", history=2)
    s = run(cc)
    allr = s[s.region == "all"].set_index("method")
    assert set(allr.index) == set(METHODS)
    assert allr.loc["persistence", "changed_acc"] == 0
    assert (tmp_path / "out" / "change_k2_s0" / "change.json").exists()

    # classifier maps as targets: forecast 2025; classifying the true 2025 embedding is exact
    s = run(ChangeConfig(**{**cc.__dict__, "labels": "classifier"}))
    allr = s[s.region == "all"].set_index("method")
    assert allr.loc["classify_true_next", "acc"] == 1.0
    assert (tmp_path / "out" / "change_clf_k2_s0" / "change.json").exists()
