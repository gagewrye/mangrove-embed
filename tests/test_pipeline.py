import numpy as np
import pandas as pd

from mangrove_embed import diagnostics, synthetic
from mangrove_embed.extract import allocate, feature_columns, quadrants, tile_grid


def test_tile_grid_covers_bbox():
    t = tile_grid([-78.4, 17.7, -76.2, 18.6], 0.25)
    assert len(t) == 9 * 4
    assert np.isclose(min(x[0] for x in t), -78.4) and np.isclose(max(x[2] for x in t), -76.2)
    assert np.isclose(min(x[1] for x in t), 17.7) and np.isclose(max(x[3] for x in t), 18.6)


def test_quadrants_partition():
    q = quadrants([0, 0, 2, 2])
    assert sum((a[2] - a[0]) * (a[3] - a[1]) for a in q) == 4


def test_allocate():
    assert allocate([10, 30], 100).tolist() == [33, 98]
    a = allocate([0, 1, 99], 100)
    assert a[0] == 0 and a[1] >= 1 and a.sum() >= 100
    assert allocate([0, 0], 10).sum() == 0


def test_feature_columns():
    c = feature_columns([0, 3])
    assert len(c) == 64 * 2 + 2 and c[0] == "s0_A00"


def test_diagnostics_end_to_end(tmp_path):
    data = synthetic.make(tmp_path / "d", n=120)
    diagnostics.run(data, tmp_path / "r", models=("knn", "logreg"),
                    fsets=["pixel", "ctx7"], anchor="jamaica", block_deg=0.05)
    s = pd.read_csv(tmp_path / "r" / "results_summary.csv")
    assert {"loro", "within_blocked", "jamaica_only", "loro_edge", "within_edge"} <= set(s.protocol)
    assert not s[s.protocol.str.endswith("_edge")].centered.any()
    assert s.mangrove_f1.between(0, 1).all() and s.overall_acc.between(0, 1).all()
    assert s.centered.any()
    for f in ["q1_geometry.csv", "q1_mangrove_centroids.png", "q2_loro_by_region.png",
              "q2_false_mangrove_sources.csv", "q4_spatial_ablation.png"]:
        assert (tmp_path / "r" / f).exists()
