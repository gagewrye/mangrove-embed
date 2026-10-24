# Results

All scores are **held-out mangrove IoU under leave-one-region-out (LORO)**: train on the other
regions, test on one, for each of Everglades, Jamaica, Mekong and Sundarbans. Model weights
(`*.pt`) are not versioned. Regenerate the figures with `mangrove plots`.

| folder | test | produced by |
|---|---|---|
| [`model_comparison/`](model_comparison/) | segmentation and per-pixel models on the same LORO chip splits | `mangrove train-seg`, `bench`, `compare-pixel`; table: `compare-seg` |
| [`receptive_field/`](receptive_field/) | FCN receptive-field sweep (10 m to 1.27 km) | W&B export: `scripts/export_rf_wandb.py` |
| [`generalization/`](generalization/) | saved checkpoints scored on an unseen test region (`baja_north`: Baja California north of 25.4 N, outside `mexico_pacific`) | `mangrove --config configs/baja_test.yaml chips`, then `mangrove eval-seg --chips data/test/chips` |
| [`calibration/`](calibration/) | per-region centering / EM prior adaptation; kNN k sweep | `scripts/calibration_probe.py`, `scripts/knn_k_sweep.py` |
| [`point_diagnostics/`](point_diagnostics/) | Q1-Q4 on sampled points (geometry, transfer, domain gap, mean-pooled context) | `mangrove diagnose` |
| `synthetic/` | offline smoke test (not versioned) | `mangrove demo` |

## Model comparison
![model comparison](model_comparison/model_comparison.png)

| run folder | model | runs |
|---|---|---|
| `pixel_mlp`, `_s1`, `_s2` | sklearn MLP 128-64 on sampled chip pixels (`pixel_mlp` = seed 0) | 3 |
| `loro_fcn_s0`, `_s1`, `_s2`, `loro_fcn_pc` | dilated FCN, 630 m receptive field, ~190k params (`loro_fcn_pc` = seed 42) | 4 |
| `loro_fcn_rf1_s0`, `_s1`, `_s2` | same FCN with all dilations 0 (1x1, no context, width 184, ~185k params) | 3 |
| `loro` | SegFormer-B0, 64-band stem | 1 |
| `pixel_logreg` | logistic regression on sampled chip pixels | 1 |
| `pixel_knn` | kNN, k=5, cosine | 1 |
| `loro_mlp_s0`, `_s1`, `_s2` | per-pixel MLP 64-64-11 trained through the chip pipeline (4.9k params) | 3 |

Each folder has `holdout_<region>.json` (full metrics and config per split) and `summary.csv`
(one row per held-out region). The top models (630 m FCN 0.841, 10 m FCN 0.834, SegFormer
0.832, logreg 0.825, pixel MLP 0.824) sit within ~0.02, inside the seed-to-seed spread (sd
0.015-0.043). Spatial context adds nothing measurable: 630 m minus 10 m on matched seeds is
+0.025, -0.005, +0.014 (seeds 0-2) and -0.031 (seed 42, W&B), mean ~0. Mekong is the least stable region (IoU 0.57-0.84 across runs, with AP
steady at ~0.97, so the variation is in the threshold, not the ranking); Jamaica is the hardest.

## Receptive field
![receptive field](receptive_field/receptive_field.png)

`rf_runs.csv` has one row per (W&B run, held-out region). These runs were trained on another
machine and exist only in W&B, hence the export. Check `train_chips` before comparing:
- `rf-sweep`: Mekong only, trained on **9 other regions** (~2,400 chips), 1 seed. Not comparable
  in level with the 4-region results; within the sweep the spread (0.767-0.808) is smaller than
  the seed-to-seed spread of one config on Mekong (0.726-0.818).
- `loro_fcn_rf1_all` (10 m) and `loro_fcn` (630 m): all 4 holdouts, the original 4-region
  splits (~800 chips), seed 42, identical splits, so they pair up directly. 10 m scored higher
  on all 4 (mean 0.861 vs 0.830), but the other 630 m runs average 0.841, so the gap is within
  about one seed's noise.

## Generalization to an unseen region (Baja California north)
`baja_north` = `[-113.6, 25.5, -111.3, 27.0]` (San Juanico, Laguna San Ignacio, Loreto, Mulege,
Bahia Concepcion): 300 chips, 15% mangrove, never used in training. Models are pooled on all chip
regions; "10 regions" leaves out `mexico_pacific` (Magdalena Bay / La Paz, the nearest Baja
data), so it is the true out-of-region test. Mangrove IoU, mean over seeds 0-2 (FCN: 0, 1, 42).

| model | params | 11 regions | 10 regions (no mexico_pacific) | AP (10) |
|---|---|---|---|---|
| per-pixel FCN (1x1 convs, BatchNorm, residual) | 185k | 0.869 (1 run) | 0.852 | 0.968 |
| MLP 128-64 + BatchNorm + dropout 0.1, Jaccard loss | 17.5k | - | 0.849 | 0.969 |
| logistic regression | 715 | 0.834 | 0.846 | 0.966 |
| MLP 128-64, Jaccard loss | 17.5k | - | 0.747 | 0.946 |
| MLP 128-64, sklearn | 17.5k | 0.870 | 0.636 | 0.947 |

Files: `generalization/baja_north.csv` (train-seg checkpoints, `mangrove eval-seg`) and
`generalization/baja_north_pixel.csv` (sklearn models, `compare-pixel --test-chips`).

Conclusion: no spatial context or large model is needed. Spatial context did not help under LORO
or on Baja, and SegFormer and the dilated FCN were no better than per-pixel models. What
mattered for a new region was regularization: an unregularized MLP overfits its training regions
(0.64-0.75 on Baja), while BatchNorm + dropout brings it level with logistic regression (~0.85).
The FCN and SegFormer are dropped from further work; remaining comparison is logistic regression
vs the regularized per-pixel MLP.
