# Mapping and forecasting mangroves from AlphaEarth embeddings

## Abstract
We ask what model is needed to map mangroves from Google's AlphaEarth annual embeddings
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64 bands, 10 m), whether it holds up in regions it was
not trained on, and whether the embeddings can forecast next year's land cover. Across 11 coastal
regions:

- **A per-pixel model is enough.** With one region held out, logistic regression (715 parameters),
  a pixel MLP, a dilated FCN and SegFormer-B0 all score 0.82-0.84 mangrove IoU. That gap is
  smaller than the seed-to-seed spread. kNN, the earlier baseline, scores 0.76.
- **Spatial context adds nothing, since the embeddings already account for it.** Receptive fields
  from 10 m to 1.27 km score the same within noise.
- **Embeddings forecast where change happens, modestly.** From the two prior years, a per-pixel
  MLP ranks next year's class changes at AP 0.46-0.50, against 0.41-0.43 for persistence and 0.05
  for random. For mangrove loss it is no better than today's map, and regional hectare totals are
  off by up to 3×.
- **But much of the change is not land cover.** A fifth of detected change is tidal flat to water,
  caused by tide stage. Global Mangrove Watch's own year-to-year changes are barely visible in the
  embeddings (AP ≤ 0.05).
- **The maps see loss, not damage.** Checked against Sentinel-2, Hurricane Melissa damaged the
  canopy of 91% of southwest Jamaica's mangrove; the annual map flags 12% as lost. Classifier
  confidence does not track damage either.

The classifier and forecaster ship as one model file that maps, forecasts and detects change from
exported GeoTIFFs ([§6](#6-use-the-change-models)). Full results: [`results/README.md`](results/README.md).

![Model comparison, leave-one-region-out](results/model_comparison/model_comparison.png)
*Figure 1. Mangrove IoU on the held-out region (bar = mean, dot = one seed). Jamaica is the
hardest region and the one where logistic regression falls behind.*

## 1. Data and methods
- **Inputs:** AlphaEarth annual embeddings on each region's native UTM 10 m grid.
- **Labels:** 11 exclusive classes, a Python port of the v3 Code Editor script
  (`legacy/mangrove_training_knn.js`), built from existing global products:

  | class | source |
  |---|---|
  | 1 mangrove | Global Mangrove Watch (GMW) v3 extent |
  | 2 salt marsh | GWL_FCS30 wetland map (186) and ESA WorldCover (90) |
  | 3 tidal flat | GWL_FCS30 (187) and WorldCover (60 / 80 / 90) |
  | 4 water, 5 seasonal water | JRC Global Surface Water: ≥ 10 / 1-9 months a year |
  | 6 built-up | WorldCover (50) or GHSL built-up ≥ 50% |
  | 7 barren, 8 coastal forest, 9 cropland | WorldCover (60, 10, 40) |
  | 10 herbaceous wetland, 11 other land | WorldCover (90; 20 / 30 / 70 / 100) |

  Pixels that WorldCover or GWL_FCS30 call mangrove but the label does not are masked as ambiguous. Non-mangrove
  classes are only sampled within 5 km of mangrove.
- **Training labels:** mapping (2.1-2.3) uses these labels for **2020**, the last GMW v3 year.
  Change (2.4-2.5) uses a classifier's maps for 2017-2025 as targets, fit on these labels for
  2017-2020; the shipped model uses logistic regression.
- **Regions:** Sundarbans, Mekong, Mahakam, Matang, Niger Delta, Pará, Papua, Everglades, Rufiji,
  Jamaica, Mexico Pacific (`configs/default.yaml`), plus Baja California north as an unseen test region.
- **Samples:** 128 × 128 px chips (1.28 km), 300 per region, half centred on mangrove.
- **Protocols:**
  - Leave one region out (LORO): Everglades, Jamaica, Mekong or Sundarbans held out.
  - Pooled: all regions, tested on held-out ~5 km blocks.
  - Unseen region: pooled training without Mexico Pacific, tested on Baja north.
- **Metric:** mangrove IoU; average precision (AP) when ranking. Seeds 0-2 unless noted.
- **Change:** matched pixels for 2017-2025 (400 chips per region, 100 centred on GMW change).
  Forecasters see the embeddings of the k prior years and are tested on held-out blocks.

## 2. Results
### 2.1 Model choice
| model | params | LORO IoU | pooled IoU |
|---|---|---|---|
| dilated FCN, 630 m context | 190k | 0.84 | — |
| per-pixel FCN | 185k | 0.83 | 0.91 |
| SegFormer-B0 | — | 0.83 | — |
| logistic regression | 715 | 0.83 | 0.90 |
| pixel MLP 128-64 | 17.5k | 0.82 | 0.90 (with BN + dropout) |
| kNN (k = 5) | — | 0.76 | — |

When a region is in training, logistic regression and the regularized MLP tie in every region
(within 0.02). Logistic regression's weakness in Jamaica (0.58) appears only when Jamaica is unseen.

### 2.2 Spatial context
![Receptive field](results/receptive_field/receptive_field.png)
*Figure 2. Left: Mekong IoU vs FCN receptive field; the curve is flat within one configuration's
seed range (0.73-0.82). Right: 10 m vs 630 m on matched splits, mean difference ≈ 0.*

### 2.3 Unseen region (Baja California north)
| model | IoU | AP |
|---|---|---|
| per-pixel FCN | 0.850 | 0.968 |
| **MLP 128-64 + BatchNorm + dropout 0.1** | **0.849** | 0.969 |
| logistic regression | 0.846 | 0.966 |
| MLP 128-64, no regularization | 0.64-0.75 | 0.946 |

### 2.4 Forecasting change
![Robustness](results/change/robustness_logreg.png)
*Figure 3. Forecasting next year from the two prior years, with logistic-regression maps as targets
(water and tidal flat merged), three seeds × three forecast years. The forecaster (blue) beats
persistence (grey) every year, by more on durable change. The mangrove-specific margins are within
seed spread in some years.*

- **The labels themselves cannot be forecast.** Trained directly on the yearly labels
  (forecasting 2020), every method ranks mangrove change near random (AP ≤ 0.05). About 70% of
  that label change is tidal flat ↔ water flicker in GWL_FCS30, and GMW's yearly mangrove
  changes are barely visible in the embeddings. Hence the classifier maps as targets.
- **Cleaner targets and change-focused losses don't help.** Dropping uncertain or reverting
  training pixels, upweighting change, a Jaccard loss and a separate change head all match or trail
  plain cross-entropy (details in [`results/README.md`](results/README.md)).

![Class transitions](results/change/change_model_k2_s0/transitions.png)
*Figure 4. Regularized-MLP maps as targets. Left: what changing pixels become. Tidal flat → water
alone is 21% of change. Right: forecasting helps most for barren, other land, tidal flat and
mangrove, and not at all for water, salt marsh or cropland.*

### 2.5 Observed change, forecast use, and damage vs loss
These results come from the partner report (`mangrove change-report`, [`results/change_report/`](results/change_report/README.md)),
which runs the shipped model over all 400 temporal chips per region. "Confirmed" loss means a
patch of ≥ 0.5 ha that was mangrove in both 2017 and 2018 and not mangrove in both 2024 and 2025.

![Regions compared](results/change_report/regions_compared.png)
*Figure 5. Confirmed and edge change per region. Rates use the 300 area-representative chips
per region. Baja California Sur (0.1% confirmed loss) is among the most stable regions; Rufiji, Mahakam and
Mekong lose 4-5%.*

![Forecast skill](results/change_report/forecast_skill.png)
*Figure 6. The 2025 forecast on held-out blocks. Left and middle: share of 2025 change caught when
flagging the highest-risk area first, against ranking by today's map's uncertainty. Right:
forecast vs observed hectares per region.*

- **Ranking where change will happen works, a little.** At 10% of area flagged, the forecaster
  catches 75% of change against 71% for today's map, consistent with the AP gain in 2.4.
- **It does not anticipate mangrove loss.** Loss concentrates on pixels the classifier is already
  unsure of (patch edges), and the forecast adds nothing beyond that (85% vs 86%).
- **Hectare totals are a prototype.** Summing probabilities gives 1,720 ha of forecast loss against
  2,111 ha observed across regions, but individual regions are off by 0.2-3.1×.

![Damage vs loss](results/change_report/damage_vs_loss.png)
*Figure 7. An independent check with Sentinel-2 (`scripts/storm_check.py`): 1,500 pixels per
storm that were mapped mangrove beforehand. Damage is NDVI change in cloud-masked
composites (0-4 months and ~1 year after landfall), relative to the same windows one year earlier.
Sundarbans cyclones (Amphan, Remal) are omitted: repeated cyclones and few clear looks make their
baselines unreliable.*

- **Storms damage far more than they remove, and most canopy recovers.** Melissa: 91% damaged,
  32% still damaged after ~1 year. Irma: 59% → 17%. Beryl: 27% → 5%. Controls stay at 1-6%.
- **The annual map captures the lasting part.** It flags 12%, 1% and 3% as lost, below the
  share still damaged a year on. It detects mangrove that is gone or no longer looks like mangrove,
  not damaged standing forest.
- **Classifier confidence is not a health signal.** Among Melissa pixels sampled by pre-storm
  P(mangrove) (`scripts/storm_damage_by_confidence.py`), canopy damage *rose* with confidence
  (median NDVI change −0.18 below 0.80, −0.49 above 0.999), while flips to non-mangrove fell (48% → 0%). Confidence
  measures how separable a pixel is from neighbouring cover, not its condition.
- **So use the two together.** The embedding map identifies *what* is mangrove and *where* it was
  lost or converted, consistently and through clouds. Sentinel-2 indices measure condition and
  timing (every ~5 days) after an event.

## 3. Limitations
- LORO covers 4 regions; SegFormer, logistic regression and kNN in Table 2.1 are single runs.
- Change targets are classifier maps, so some "change" is flicker on borderline pixels.
  Merging water with tidal flat and scoring durable change reduce this but do not remove it;
  dropping uncertain or reverting pixels from training does not improve the forecaster.
- Mangrove thresholds tuned on training regions do not transfer, so we report argmax.
- Annual maps miss sub-annual damage and recovery (2.5), and the 2025 map holds only ~2 months
  after Melissa. Loss and damage are checked against Sentinel-2, not field data.

## 4. Conclusion
Use a regularized per-pixel MLP (or logistic regression) on the raw embeddings. It matches
segmentation models, deploys as a per-pixel operation in Earth Engine, and is the safest choice on
a new coast. The same embeddings carry a usable, though tide-contaminated, signal of next year's
change, but not of mangrove loss specifically. The annual maps are best used to track extent and
find lost or converted mangrove; canopy damage needs sub-annual optical data such as Sentinel-2.

## 5. Reproduce
```bash
pip install -e ".[dev]"            # extras: seg (torch, transformers<5), deploy, track (wandb)
mangrove auth                      # one-time Earth Engine login
mangrove demo                      # check the install on synthetic data, no Earth Engine
```
The Earth Engine project defaults to `intricate-web-334523` (`configs/default.yaml`; override
with `EE_PROJECT`).

| result | commands |
|---|---|
| point diagnostics (`results/point_diagnostics/`) | `mangrove extract`, `mangrove diagnose` |
| 2.1 model comparison | `mangrove chips`, `mangrove train-seg --protocol loro\|pooled [--model fcn\|mlp]`, `mangrove compare-pixel`, `mangrove bench`, `mangrove compare-seg` |
| 2.2 spatial context | `mangrove train-seg --model fcn --fcn-dilations ...`, `python scripts/context_sweep.py` |
| 2.3 unseen region | `mangrove --config configs/baja_test.yaml chips`, `mangrove eval-seg --chips data/test/chips` |
| 2.4 change | `mangrove temporal-chips`, `mangrove change --labels logreg --history 2` |
| 2.4 training variants | `python scripts/change_sweep.py` |
| 2.5 change report for partners (`results/change_report/`) | `mangrove change-report` |
| 2.5 storm damage vs loss (Sentinel-2) | `python scripts/storm_check.py`, `python scripts/storm_damage_by_confidence.py` |
| figures | `mangrove plots` |

Every command has `--help`. Runs resume, and each variant writes its own results folder.
- **Hardware:** CUDA, Apple MPS and CPU are picked automatically (`--device` to force).
- **Tracking:** `--wandb` logs to Weights & Biases (`pip install -e ".[track]"`).

## 6. Use the change models
`models/change_logreg_k2.pt` holds the logistic-regression classifier and the 2-year forecaster
(2025 holdout: change AP 0.49, mangrove IoU 0.95). It refuses to load if the model code no longer
reproduces its stored reference outputs. Install with `pip install -e ".[deploy]"`.

**Inputs:** one 64-band AlphaEarth GeoTIFF per year, all on one grid. Export them from Earth
Engine:
```python
import ee; ee.Initialize(project="<your-project>")
BANDS = [f"A{i:02d}" for i in range(64)]
region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterBounds(region)
p = col.first().select(0).projection().getInfo()          # native UTM 10 m grid, shared by all years
for y in (2023, 2024, 2025):
    img = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic().select(BANDS).toFloat()
    ee.batch.Export.image.toDrive(img, description=f"emb_{y}", region=region, crs=p["crs"],
                                  crsTransform=p["transform"], maxPixels=1e10).start()
```
If the export is split into several files, merge them with `gdalbuildvrt` and pass the VRT.

```bash
# forecast 2026 from 2024-2025
mangrove change-predict --bundle models/change_logreg_k2.pt --inputs emb_2024.tif emb_2025.tif --out out/f2026
# what changed between 2023 and 2025
mangrove change-detect --bundle models/change_logreg_k2.pt --before emb_2023.tif --after emb_2025.tif --out out/d23_25
```
| output | contents |
|---|---|
| `f2026_classes.tif` | current class, forecast class (codes 1-11; 3 = tidal flat + water; 255 = no data) |
| `f2026_probs.tif` | `p_change`, `p_mangrove` next year; `--probs` adds every class |
| `d23_25_classes.tif` | class before, class after |
| `d23_25_probs.tif` | `p_change`, `p_mangrove_gain`, `p_mangrove_loss`, `cosine` (embedding similarity) |
| `d23_25_transitions.csv`, `d23_25_classes.csv` | area (ha) per from → to pair and per class; pixels with `p_change ≥ --min-p` (0.5) |

To rebuild the bundle, run `mangrove change --labels logreg --history 2 --save-bundle
models/change_logreg_k2.pt`. Keep the default `--methods`: this run writes to the benchmark's
results folder.

## Repository
```
configs/          regions, sampling, tiling
src/mangrove_embed/
  extract.py      Earth Engine -> labeled points (parquet)
  chips.py        Earth Engine -> dense chips;  temporal.py: multi-year chips
  diagnostics.py  point diagnostics
  seg/            SegFormer (64-band stem), FCN, MLP; multi-region training
  change.py       change forecasting;  deploy.py: model file, change-predict / change-detect
  plots.py        figures;  cli.py: `mangrove` command
scripts/          sweeps and W&B exports
results/          numbers and figures (index: results/README.md)
models/           shipped change model
legacy/           v3 Code Editor kNN + change baseline (reference)
```
