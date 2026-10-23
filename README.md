# mangrove-embed

Cross-region diagnostics for mangrove classification on Google AlphaEarth embeddings
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`). Labels are a Python port of the v3 Code Editor
script (`legacy/mangrove_training_knn.js`): 11 exclusive classes with hard negatives, ambiguity mask,
coastal-zone restriction. Everything runs from the terminal; Earth Engine is only a remote
compute API (no Code Editor, Drive exports, or assets).

## Setup
```bash
conda activate gee            # or: python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
mangrove auth                 # one-time browser login (skip if already authenticated)
mangrove demo                 # verify install on synthetic data, no Earth Engine
```
Project defaults to `intricate-web-334523` (`configs/default.yaml`; override with `EE_PROJECT`).

## Run
```bash
mangrove extract --regions sundarbans mekong everglades jamaica   # quick subset
mangrove diagnose --models knn logreg --skip-within                # fast pass
mangrove extract                                                   # all 11 regions
mangrove diagnose                                                  # full run (MLP + within-CV is the slow part)
```
Extraction tiles each region, allocates the per-class budget by class area, pulls tiles
concurrently straight to parquet, splits oversized tiles, and caches under `data/raw/`.
Rerun to resume. Output: `data/processed/<region>_{labeled,unlabeled}.parquet`.

## Classes
1 mangrove · 2 salt_marsh · 3 tidal_flat · 4 water · 5 seasonal_water · 6 built_up · 7 barren ·
8 coastal_forest · 9 cropland · 10 herbaceous_wetland · 11 other_land

Models train on all 11; scoring is on mangrove (precision/recall/F1/IoU from argmax, AP from
p(mangrove)) plus overall accuracy. Main metric: `mangrove_f1`.

## Diagnostics
| | Question | Readout |
|---|---|---|
| Q1 | Are a region's mangroves closer to other regions' mangroves than to their most confusable local class? | `q1_geometry.csv` margin |
| Q2 | Does a parametric model transfer better than kNN (k=5, as in v3)? | LORO: logreg / MLP vs kNN; `q2_false_mangrove_sources.csv` |
| Q3 | Domain gap | within-region spatially blocked CV − LORO; anchor-only (Jamaica) → others |
| Q4 | Does spatial context help, and cross-region? | 3×3 / 7×7 / 15×15 mean-pooled vs pixel |
| + | Does label-free per-region centering help? | centered vs uncentered LORO |

AlphaEarth pixels already encode their surroundings, so Q4 measures *marginal* context gain.
Mean pooling is a lower bound (composition, not layout). A LORO gain there justifies a patch
CNN/transformer; a within-only gain means it's learning region-specific layout.

## Layout
```
configs/default.yaml        project, regions, sampling, tiling
src/mangrove_embed/
  extract.py                Earth Engine -> parquet (v3 labels)
  diagnostics.py            Q1-Q4
  synthetic.py              offline test data
  cli.py                    `mangrove` command
tests/
legacy/                     v3 Code Editor kNN + change baseline (reference)
```
