# Mangrove change report, 2017-2025

Prepared for the Aburto Lab and the coastal Baja marine program. Maps come from Google's AlphaEarth
annual satellite embeddings (10 m) and our mangrove classifier, run on 1.28 km sample tiles in
11 coastal regions. Regenerate with `mangrove change-report`.

## Headlines
- **Baja California Sur (Bahía Magdalena to La Paz) shows no large-scale mangrove loss.** Of 10,733 ha
  mapped in 2017-18, 11 ha (0.10%) was lost in patches of 0.5 ha or more by 2024-25, and 14 ha was
  gained. That ranks 10th of 11 regions; the average is 1.7%. ([baja/baja_overview.png](baja/baja_overview.png))
- **Most of the small Baja losses are edge shifts:** 1-2 pixel (10-20 m) changes along patch edges,
  mostly map noise. A few are continuous 20-30 m strips along the seaward edge (e.g. 24.786, -112.192;
  25.085, -112.169), which could be shoreline retreat and are worth a field check.
  ([baja/baja_change_tiles.png](baja/baja_change_tiles.png))
- **Elsewhere, loss is real and large.** Rufiji 5.4%, Mahakam 4.2%, Mekong 4.0%
  ([regions_compared.png](regions_compared.png)).
- **Hurricanes show up, but only the forest that is gone.** In southwest Jamaica (Black River), the
  mapped mangrove area fell ~20% over 2024-25, after Beryl (Cat 4, July 2024) and Melissa (Cat 5,
  landfall 28 Oct 2025); the rest of Jamaica held steady. Damage was far wider than loss: a
  Sentinel-2 check found canopy damage across 91% of sampled SW Jamaica mangrove after Melissa,
  while the map shows 12% of those pixels as lost. Hurricane Irma (Sept 2017) is visible in the
  Everglades' embeddings ([hurricanes.png](hurricanes.png)).

## Applications
The AlphaEarth embeddings are built partly from Sentinel-2, so this model complements Sentinel-2
rather than replacing it.

**Use this model for: where is mangrove, and where has it been lost or converted?**
- **It tells mangrove apart from other green cover.** Sentinel-2 greenness (NDVI) cannot separate
  mangrove from rice, palms or other vegetation. The classifier can, and it also says what replaced
  lost mangrove.
- **Clouds don't block it.** The embeddings include Sentinel-1 radar, so every year has a complete map
  even in very cloudy places (e.g. the Sundarbans).
- **It is consistent everywhere.** One map per year, with the same method in every region, so Baja can be
  compared directly with the Mekong or Rufiji.
- **It is cheap to run.** A whole coastline takes minutes.

**Use Sentinel-2 for: how is the forest doing now, and did that storm hurt it?**
- **Timing:** a new image every ~5 days, while this model makes one map a year, released months later.
- **Damage and stress:** after Hurricane Melissa, Sentinel-2 showed canopy damage across 91% of
  sampled SW Jamaica mangrove; this model mapped 12% as lost. The model finds forest that is gone,
  not forest that is damaged but standing. Classifier confidence is not a health measure either.
- **Transparency:** NDVI is a simple, citable index; the embeddings are a model output that only
  starts in 2017.

**Suggested workflow:** use the annual map as the base layer to find loss hotspots and track extent.
Check flagged sites and post-storm areas with Sentinel-2, then send field teams to what stands out.

## What is in each figure
| figure | shows |
|---|---|
| [regions_compared.png](regions_compared.png) | confirmed vs edge loss and gain per region, 2017-18 to 2024-25 |
| [extent_trend.png](extent_trend.png) | mangrove extent each year, per region (2017 = 100) |
| [loss_causes.png](loss_causes.png) | what replaced lost mangrove, and the year it was lost |
| [loss_hotspots.png](loss_hotspots.png) | the 12 largest loss patches across all regions |
| [hurricanes.png](hurricanes.png) | Jamaica by coast, Beryl and Melissa; year-to-year change of intact mangrove |
| [damage_vs_loss.png](damage_vs_loss.png) | Sentinel-2 canopy damage after storms vs mangrove mapped as lost |
| [forecast_skill.png](forecast_skill.png) | how well the 2025 forecast did on held-out tiles |
| [forecast_maps.png](forecast_maps.png) | forecast risk vs observed 2025 change, held-out tiles |
| [forecast_2026.png](forecast_2026.png) | **experimental:** forecast hectares lost / gained in 2026 |
| [baja/](baja/) | overview, change tiles, loss patches, 2026 risk (experimental) |

## Clues to causes
After loss, the "replaced by" class hints at a cause but does not diagnose it.
- **Water / tidal flat (Mekong, Mahakam, Pará, Sundarbans):** erosion and aquaculture ponds both look
  like this.
- **Other vegetation (Rufiji, Niger Delta):** conversion to farmland, palms or other cover.
- **Everglades:** 86% of its confirmed loss turned to water / tidal flat in 2024, including the largest
  patch in the dataset (144 ha at 25.725, -81.307). We have no explanation yet. It needs checking
  against imagery before anyone acts on it.

## How to read the numbers
- **Confirmed** change = connected patches of at least 0.5 ha that were stable in both 2017 and 2018
  and different in both 2024 and 2025. **Edge** change is the rest, mostly 1-2 pixel shifts along edges.
- Rates come from 300 area-representative tiles per region; 100 more tiles per region were placed on
  Global Mangrove Watch change and are used only for maps. Hectares are within the sample, not regional
  totals.
- The 2025 map holds only ~2 months after Hurricane Melissa; its full effect will show in 2026 data.

## Forecasts
- **Where will change happen?** On held-out tiles, flagging the riskiest 10% of area catches 75% of
  2025 land-cover change (today's map alone: 71%). For mangrove loss specifically, the forecast is no
  better than today's map (85% vs 86%): it mostly marks uncertain edges.
- **How many hectares? Experimental prototype.** Summed across regions the 2025 forecast was close
  (1,720 ha forecast vs 2,111 ha observed loss), but individual regions were off by 0.2-3.1x.
  Baja: 30 ha forecast vs 13 ha observed. Treat the 2026 numbers as a direction to develop, not
  estimates.

## GIS files
- [baja/baja_tiles.geojson](baja/baja_tiles.geojson): footprint of every Baja tile, with its
  hectares (mangrove, confirmed and edge loss / gain, 2026 forecast).
- `baja/tiles/<chip>.tif`: per-tile GeoTIFF, UTM 12N, 10 m. Band 1: 1 other, 2 mangrove kept,
  3 edge gain, 4 edge loss, 5 confirmed gain, 6 confirmed loss. Band 2: experimental 2026 loss risk.
- [loss_hotspots.geojson](loss_hotspots.geojson) / [loss_hotspots.csv](loss_hotspots.csv): every
  confirmed loss patch in all regions (location, ha, year, what it is now).

## Limits
- Tiles are samples, not wall-to-wall maps; northern Baja (Loreto, Mulegé, San Ignacio) is not covered
  yet.
- Accuracy is checked against global products, not field data. Field or drone surveys would be the
  real test.
- 10 m pixels cannot resolve narrow fringe or dwarf mangroves well.
