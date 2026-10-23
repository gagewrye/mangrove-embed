"""Synthetic stand-in data (multi-class, per-region shift) for verifying the pipeline offline."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import BANDS

REGIONS = {"jamaica": (-77, 18), "everglades": (-81, 25.4),
           "sundarbans": (89, 22), "rufiji": (39.2, -8.0)}
CLASSES = [1, 4, 8, 9]  # mangrove, water, coastal_forest, cropland


def make(out_dir, n=150, scales=(0, 1, 3, 7), seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_dir = {c: rng.normal(size=64) for c in CLASSES}
    class_dir[8] = 0.6 * class_dir[1] + 0.8 * class_dir[8]  # coastal forest resembles mangrove
    for r, (lo, la) in REGIONS.items():
        shift = rng.normal(size=64) * 1.2

        def gen(k, c, mix=None):
            base = shift + (class_dir[c] if c else 0) + rng.normal(size=(k, 64))
            if mix is not None:  # boundary pixel: blended with a neighbouring class
                base = base + 0.6 * np.stack([class_dir[m] for m in mix])
            d = {}
            for s in scales:
                v = base + rng.normal(size=(k, 64)) / (1 + s)  # context averages out noise
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                d.update({f"s{s}_{b}": v[:, i] for i, b in enumerate(BANDS)})
            d.update(longitude=lo + rng.uniform(0, .3, k), latitude=la + rng.uniform(0, .3, k),
                     region=r)
            return pd.DataFrame(d)

        lab = pd.concat([gen(n, c).assign(**{"class": c}) for c in CLASSES], ignore_index=True)
        lab.to_parquet(out / f"{r}_labeled.parquet", index=False)
        gen(2 * n, None).to_parquet(out / f"{r}_unlabeled.parquet", index=False)
        edge = []
        for c in CLASSES:
            others = rng.choice([o for o in CLASSES if o != c], size=n // 2)
            edge.append(gen(n // 2, c, mix=others).assign(**{"class": c}))
        pd.concat(edge, ignore_index=True).to_parquet(out / f"{r}_edge.parquet", index=False)
    return out
