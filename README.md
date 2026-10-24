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

Outputs: `results/point_diagnostics/`.

AlphaEarth pixels already encode their surroundings, so Q4 measures *marginal* context gain.
Mean pooling is a lower bound (composition, not layout). A LORO gain there justifies a patch
CNN/transformer; a within-only gain means it's learning region-specific layout.

## Segmentation model
`pip install -e ".[seg]"` (torch, transformers<5). Ported from UCSD-E4E/ml-mangrove; the only
change is a native multi-band stem:
```python
from mangrove_embed.seg import SegFormer, JaccardLoss
model = SegFormer(num_classes=11, input_image_size=64, in_channels=64, stem_stride=2)
loss_fn = JaccardLoss(num_classes=11)
```
`in_channels=64` replaces the stage-1 patch-embedding conv (all other pretrained weights kept).
`stem_stride` (pretrained 4) can be lowered for 10 m embeddings; 4 puts stage 1 at 40 m.

## Multi-region segmentation
```bash
mangrove chips --regions sundarbans mekong everglades jamaica   # dense 64-band chips + labels
mangrove train-seg --protocol loro --holdout all                # train on N-1, test held-out
mangrove train-seg --protocol pooled                            # in-distribution reference
```
Chips: 128x128 px (1.28 km) on the native 10 m UTM grid, half centered on mangrove, half on random
coast; cached in `data/chips/<region>/`. Training: SegFormer-B0, 64-band stem, stem stride 2,
JaccardLoss (inverse-sqrt class weights), dihedral augmentation, model selection on a validation
split of the training regions only. Results in `results/model_comparison/<protocol>/summary.csv`.
pooled − loro = segmentation-level domain gap.

Input reduction and a capacity baseline (each variant writes to its own results folder):
```bash
mangrove train-seg --pca 16                # fixed PCA 64->16, fit on training regions only
mangrove train-seg --input-proj 16         # learned 1x1 projection 64->16
mangrove train-seg --model fcn             # ~190k-param dilated CNN (630 m receptive field)
mangrove train-seg --model fcn --pca 16
mangrove train-seg --from-scratch          # same SegFormer, random init (are RGB weights useful?)
mangrove compare-seg                       # all runs side by side
```
Point-level check: 16 PCs retain 93% of variance and match full 64-d LORO kNN F1 (0.806 vs
0.800); 8 PCs lose ~0.06 AP for logreg.

## Per-pixel baselines and threshold tuning
```bash
mangrove compare-pixel                     # knn / logreg / mlp on chip pixels, same LORO splits
mangrove compare-seg --region mekong       # everything side by side on one held-out region
```
`compare-pixel` trains on pixels from the other regions' chips and tests on a uniform pixel sample
of the held-out region's chips (the population `train-seg` scores), writing to
`results/model_comparison/pixel_<model>/` so `compare-seg` lists it with the seg models. Both
`train-seg` and `compare-pixel` also tune the mangrove threshold on validation pixels from the training regions
(`threshold`, `mangrove_iou_tuned`).

Findings (4 regions, held-out mangrove IoU): per-pixel MLP 0.846, SegFormer-B0 0.832, logreg 0.825,
kNN 0.758; AP 0.975 for all but kNN (0.882). The FCN receptive-field sweep (10 m to 1.27 km) on
Mekong was flat within run-to-run noise (~0.025). Thresholds tuned on training regions did not
transfer (calibration shift between regions, e.g. Jamaica).

## Per-pixel MLP and multi-seed benchmark
```bash
mangrove train-seg --model mlp                     # 64 -> 64 -> 11 per pixel (~4.9k params)
mangrove train-seg --model mlp --mlp-hidden 128,64 # wider / deeper
mangrove bench                                     # mlp, fcn, segformer x seeds 0 1 2, LORO
mangrove bench --models mlp fcn --seeds 0 1 2 3 4 --holdout mekong
```
`--model mlp` is a stack of 1x1 convs (no context, no normalization), trained through the same
chip pipeline as the seg models: same splits, JaccardLoss, validation-based model selection and
threshold tuning. Folder `loro_mlp` (non-default widths: `loro_mlp_h128-64`).

`bench` runs each model at each seed into `loro_<model>_s<N>`, skips runs that are already
complete (so it resumes), then prints `compare-seg` plus a seed-aggregated table: per run, mean
over held-out regions; per variant, mean ± std over seeds. `compare-seg` prints that table
whenever any variant has 2+ seeds (an unsuffixed folder counts as one seed).

## Generalization to an unseen region
```bash
mangrove --config configs/baja_test.yaml chips            # test-only chips -> data/test/chips
mangrove train-seg --protocol pooled --model mlp --mlp-hidden 128,64 --mlp-norm --mlp-dropout 0.1
mangrove eval-seg --chips data/test/chips --wandb         # score every saved checkpoint
mangrove compare-pixel --models logreg mlp --test-chips data/test/chips --wandb
```
Test regions live under their own `data_dir`, so no training run picks them up. `eval-seg`
rebuilds each split from its saved config, applies its validation-tuned threshold, writes
`results/generalization/<region>.csv`, and with `--wandb` adds `test/<region>/...` to the split's
W&B training run. `compare-pixel --test-chips` trains on every region in `--chips` (or
`--regions`) and tests on the test chips. `--mlp-norm` / `--mlp-dropout` add BatchNorm / dropout
after each MLP hidden layer. Findings: `results/README.md`.

## Spatial context for the regularized MLP
```bash
mangrove train-seg --model mlp --mlp-hidden 128,64 --mlp-norm --mlp-dropout 0.1 --mlp-context 1,3,7
python scripts/context_sweep.py --wandb     # ctx none/1/3/7/15/1,3,7 x seeds 0-2, pooled + Baja
```
`--mlp-context r1,r2,...` concatenates, for each radius r, the mean embedding over the
(2r+1)x(2r+1) window to every pixel's own embedding before the MLP (no-data pixels excluded from
the mean). The model stays per-pixel apart from these parameter-free pooled features, so it is
still deployable in Earth Engine (`reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(r))`).
Only the first layer widens (64 x (1 + #radii) inputs). Folder suffix `_ctx1-3-7`.

`context_sweep.py` reuses the pooled 10-region splits of the `*_no_mexpac` runs (same seed =
same test blocks), skips finished runs, scores every checkpoint on Baja north, and writes
`results/context/mlp_context.csv` plus paired (same-seed) deltas against the per-pixel MLP.

## Receptive-field sweep (FCN)
Five residual blocks in every run (fixed depth); only the dilations change. A dilation of 0 is a
1x1 block, so `0,0,0,0,0` is a per-pixel MLP with no spatial context.

| dilations | receptive field | params | folder |
|---|---|---|---|
| `0,0,0,0,0` + `--fcn-width 184` | 1 px (10 m) | ~190k | `loro_fcn_rf1` |
| `1,1,1,1,1` | 11 px (110 m) | 190k | `loro_fcn_rf11` |
| `1,2,4,4,4` | 31 px (310 m) | 190k | `loro_fcn_rf31` |
| default `1,2,4,8,16` | 63 px (630 m) | 190k | `loro_fcn` |
| `1,2,8,20,32` | 127 px (1.27 km ≈ chip) | 190k | `loro_fcn_rf127` |

`--fcn-width 184` keeps the pixel-only model at the same parameter count (width 64 would be 26k).
Add `--seed N` for repeats (folder gets `_sN`).

The sweep runs so far live in W&B; `python scripts/export_rf_wandb.py` copies them to
`results/receptive_field/rf_runs.csv`.

## Figures
```bash
mangrove plots    # results/model_comparison/model_comparison.png, results/receptive_field/receptive_field.png
```
`results/README.md` indexes every results folder.

## Hardware
`train-seg` picks CUDA > Apple Metal (MPS) > CPU automatically; force with `--device`.
- **NVIDIA:** fp16 autocast + GradScaler on by default. Install a CUDA torch wheel, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- **Apple Silicon (MPS):** default `pip install torch` includes Metal. fp32 by default; `--amp`
  enables fp16 autocast. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so any op Metal lacks runs on CPU
  instead of failing. Quick check: `python -c "import torch; print(torch.backends.mps.is_available())"`.
  Smoke test: `mangrove train-seg --model fcn --epochs 1 --max-chips-per-region 20 --tag mps_smoke`.

## Experiment tracking (W&B)
```bash
pip install -e ".[track]" && wandb login
mangrove train-seg --wandb                         # project mangrove-embed
mangrove train-seg --wandb --model fcn --wandb-entity <team>
```
One W&B run per split, grouped by variant (`loro`, `loro_scratch`, `loro_fcn`, ...). Logged:
config + model info (params, PCA variance), per-epoch train loss / lr / val metrics / per-class
IoU, and held-out test metrics in the run summary (`test/<region>/...` plus region-agnostic
`test/...` so the group view averages across held-out regions). `WANDB_MODE=offline` works.
Naming: `--wandb-name rf11` (split appended → `rf11/holdout_mekong`), `--wandb-group rf-sweep`
(default: results folder), `--wandb-tags sweep mekong`, `--wandb-notes "..."`.

## Layout
```
configs/default.yaml        project, regions, sampling, tiling
src/mangrove_embed/
  extract.py                Earth Engine -> parquet (v3 labels)
  chips.py                  Earth Engine -> dense chips for segmentation
  diagnostics.py            Q1-Q4
  plots.py                  `mangrove plots` summary figures
  synthetic.py              offline test data
  cli.py                    `mangrove` command
  seg/                      SegFormer (native 64-band stem) + JaccardLoss, from ml-mangrove;
                            train.py: multi-region training, loro / pooled
tests/
legacy/                     v3 Code Editor kNN + change baseline (reference)
```
