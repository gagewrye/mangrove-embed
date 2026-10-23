# Results

All scores are **held-out mangrove IoU under leave-one-region-out (LORO)**: train on the other
regions, test on one, for each of Everglades, Jamaica, Mekong and Sundarbans. Model weights
(`*.pt`) are not versioned. Regenerate the figures with `mangrove plots`.

| folder | test | produced by |
|---|---|---|
| [`model_comparison/`](model_comparison/) | segmentation and per-pixel models on the same LORO chip splits | `mangrove train-seg`, `bench`, `compare-pixel`; table: `compare-seg` |
| [`receptive_field/`](receptive_field/) | FCN receptive-field sweep (10 m to 1.27 km) | W&B export: `scripts/export_rf_wandb.py` |
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
