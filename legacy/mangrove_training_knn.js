// =============================================================================
// Mangrove Training Samples + kNN test across difficult coasts
// for Google Satellite Embeddings (AlphaEarth Foundations)
//
// v3: trains on hard regions (humid tropics) with hard negative classes.
//   - 10 regions: one easy arid reference (NW Mexico) plus 9 humid-tropical deltas
//     where mangroves border rainforest, swamp forest, nipa palm, rice and shrimp ponds.
//   - Hard negatives get their own classes instead of being lumped into "other":
//     coastal forest, seasonal water (ponds/paddies proxy), cropland, herbaceous wetland.
//   - Salt marsh and tidal flat need two datasets to agree (GWL_FCS30 + WorldCover),
//     because GWL alone mislabels tropical mosaics (e.g. Ca Mau shrimp-mangrove land
//     as salt marsh).
//   - Main score is leave-one-region-out: train on 9 regions, test on the unseen 10th.
//   - Map shows the classifier's mangrove prediction for FOCUS_REGION, trained without
//     that region, so you can see where it fails.
//
// Carried over from v2: exclusive 10 m labels, GWL filtered to TARGET_YEAR, GMW extent
// (not core), candidates re-checked at 10 m, sampling at 10 m.
//
// Classes:
//   1  mangrove            GMW v3 extent TARGET_YEAR (or core)
//   2  salt_marsh          GWL 186 AND WorldCover 90
//   3  tidal_flat          GWL 187 AND WorldCover 60/80/90
//   4  water               JRC water >= 10 months/year
//   5  seasonal_water      JRC water 1-9 months/year (shrimp ponds, paddies, floods)
//   6  built_up            WorldCover 50 or GHSL >= 50% built
//   7  barren              WorldCover 60
//   8  coastal_forest      WorldCover 10 (non-mangrove trees: rainforest, swamp forest, plantations)
//   9  cropland            WorldCover 40
//   10 herbaceous_wetland  WorldCover 90 not salt marsh (freshwater marsh, nipa-like)
//   11 other_land          WorldCover 20, 30, 70, 100 (shrub, grass, snow, moss)
//
// Workflow: Run -> start the exports in Tasks (sampling 10 regions takes a while) ->
// set SAMPLES_ASSET to the exported asset -> Run again for the evaluation and map.
// =============================================================================

// ---------------------------------------------------------------------------
// 0. CONFIG
// ---------------------------------------------------------------------------

var TARGET_YEAR     = 2020;     // must be a GMW v3 year (1996, 2007-2010, 2015-2020)
var MANGROVE_LABEL  = 'extent'; // 'extent' (all mangroves) or 'core' (stable dense only)
var N_SAMPLES       = 200;      // points per class per region
var OVERSAMPLE      = 4;        // candidates drawn per kept point (some fail the 10 m check)
var CANDIDATE_SCALE = 100;      // scale for drawing candidate points
var SAMPLE_SCALE    = 10;       // embeddings' native resolution
var COAST_BUFFER_M  = 5000;     // negatives must be within this distance of mangroves
var BLOCK_DEG       = 0.5;      // spatial CV block size (degrees)
var N_FOLDS         = 5;
var K_NEIGHBORS     = 5;

// [west, south, east, north]
var REGIONS = {
  mexico_pacific: [-112.3, 24.2, -110.2, 25.4],  // arid reference: Magdalena Bay / La Paz
  sundarbans:     [88.4, 21.5, 89.6, 22.4],      // rice, shrimp ponds, villages
  mekong:         [104.6, 8.5, 106.9, 10.5],     // shrimp-mangrove mosaic, Melaleuca
  mahakam:        [117.1, -1.1, 117.7, -0.3],    // nipa palm, aquaculture
  matang:         [100.4, 4.6, 100.8, 5.1],      // managed mangroves, oil palm
  niger_delta:    [5.6, 4.2, 7.0, 4.8],          // freshwater swamp forest, nipa invasion
  para:           [-47.6, -1.2, -45.8, -0.6],    // Amazon coast, várzea forest
  papua:          [137.5, -5.0, 138.6, -4.3],    // rainforest and swamp forest to the shore
  everglades:     [-81.5, 25.1, -80.7, 25.8],    // salt marsh, freshwater swamp
  rufiji:         [39.1, -8.2, 39.5, -7.7]       // Tanzania delta, rice, woodland
};
var FOCUS_REGION = 'mekong';  // region shown on the map (predicted without its own samples)

var SAMPLES_ASSET = null;  // e.g. 'projects/intricate-web-334523/assets/mangrove_samples_2020_v3'
var EXPORT_ASSET  = 'projects/intricate-web-334523/assets/mangrove_samples_' + TARGET_YEAR + '_v3';
var EXPORT_DRIVE  = 'mangrove_samples_' + TARGET_YEAR + '_v3';

var CLASS_NAMES = ['mangrove', 'salt_marsh', 'tidal_flat', 'water', 'seasonal_water', 'built_up',
                   'barren', 'coastal_forest', 'cropland', 'herbaceous_wetland', 'other_land'];
var CLASS_IDS   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
var PALETTE     = ['006400', '7CFC00', 'DAA520', '1E90FF', '87CEEB', 'E31A1C',
                   'D4B483', '8B4513', 'FFD700', '00CED1', 'A8C97F'];
var REGION_NAMES = Object.keys(REGIONS);


// ---------------------------------------------------------------------------
// 1. EMBEDDINGS
// ---------------------------------------------------------------------------

var EMBED_BANDS = ee.List.sequence(0, 63).map(function (i) {
  return ee.String('A').cat(ee.Number(i).format('%02d'));
});

var embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(TARGET_YEAR + '-01-01', (TARGET_YEAR + 1) + '-01-01')
  .mosaic()
  .select(EMBED_BANDS);


// ---------------------------------------------------------------------------
// 2. LABELS: one exclusive class band at 10 m
// ---------------------------------------------------------------------------

var gmwExtent = ee.ImageCollection('projects/sat-io/open-datasets/GMW/extent/GMW_V3')
  .filter(ee.Filter.eq('system:index', 'gmw_v3_' + TARGET_YEAR)).first()
  .select(0).unmask(0).gt(0);
var gmwCore = ee.Image('projects/sat-io/open-datasets/GMW/core/gmw_v3_mng_core')
  .select(0).unmask(0).eq(1);
var mangrove = MANGROVE_LABEL === 'core' ? gmwCore : gmwExtent;

var gwl = ee.ImageCollection('projects/sat-io/open-datasets/GWL_FCS30')
  .filter(ee.Filter.eq('system:index', 'GWL_FCS30_' + TARGET_YEAR)).first()
  .select(0).unmask(0);
var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first().select(0).unmask(0);
var wcIs = function (codes) {
  return worldCover.remap(codes, ee.List.repeat(1, codes.length), 0).eq(1);
};

// Two datasets must agree for the classes GWL alone gets wrong in the tropics.
var saltMarsh = gwl.eq(186).and(worldCover.eq(90));
var tidalFlat = gwl.eq(187).and(wcIs([60, 80, 90]));

var seasonality = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality').unmask(0);
var water = seasonality.gte(10);
var seasonalWater = seasonality.gte(1).and(seasonality.lt(10));

var ghslBuilt = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S_10m').first()
  .select('built_surface').unmask(0);
var built = worldCover.eq(50).or(ghslBuilt.gte(50));  // >= 50 of 100 m2 built
var barren = worldCover.eq(60);
var coastalForest = worldCover.eq(10);
var cropland = worldCover.eq(40);
var herbWetland = worldCover.eq(90);
var otherLand = wcIs([20, 30, 70, 100]);

// Another dataset calls it mangrove but the mangrove label doesn't: unreliable negative.
var ambiguous = mangrove.not().and(worldCover.eq(95).or(gwl.eq(185)).or(gmwExtent).or(gmwCore));

// Coastal zone: within COAST_BUFFER_M of any mangrove. Computed on a 250 m grid (max keeps
// small stands); the buffer is in pixels and reprojected so it isn't run at 10 m.
var COAST_GRID = {crs: 'EPSG:4326', scale: 250};
var coastal = mangrove.selfMask()
  .reduceResolution({reducer: ee.Reducer.max(), maxPixels: 1024})
  .reproject(COAST_GRID)
  .unmask(0)
  .focalMax({radius: Math.ceil(COAST_BUFFER_M / COAST_GRID.scale), units: 'pixels'})
  .reproject(COAST_GRID)
  .gt(0);

// Later .where() calls win: specific/rare classes override broad land-cover ones,
// and mangrove overrides everything.
var classImage = ee.Image(0)
  .where(otherLand, 11)
  .where(herbWetland, 10)
  .where(cropland, 9)
  .where(coastalForest, 8)
  .where(barren, 7)
  .where(built, 6)
  .where(seasonalWater, 5)
  .where(water, 4)
  .where(tidalFlat, 3)
  .where(saltMarsh, 2)
  .where(mangrove, 1)
  .updateMask(ambiguous.not())
  .rename('class')
  .toInt()
  .setDefaultProjection({crs: 'EPSG:4326', scale: SAMPLE_SCALE});
// Mangroves anywhere in the region; everything else only on the coast.
var labels = classImage.updateMask(classImage.eq(1).or(coastal)).selfMask();


// ---------------------------------------------------------------------------
// 3. SAMPLING
// ---------------------------------------------------------------------------

function positiveMod(n, m) { return n.mod(m).add(m).mod(m); }

function addBlocks(fc) {
  return fc.map(function (f) {
    var xy = f.geometry().coordinates();
    var bx = ee.Number(xy.get(0)).divide(BLOCK_DEG).floor();
    var by = ee.Number(xy.get(1)).divide(BLOCK_DEG).floor();
    // (bx + 2*by) mod N gives neighbouring blocks different folds.
    return f.set({
      block: bx.format('%d').cat('_').cat(by.format('%d')),
      fold: positiveMod(bx.add(by.multiply(2)), N_FOLDS)
    });
  });
}

function sampleRegion(name) {
  var region = ee.Geometry.BBox.apply(null, REGIONS[name]);
  var candidates = labels.stratifiedSample({
    numPoints: N_SAMPLES * OVERSAMPLE,
    classBand: 'class',
    region: region,
    scale: CANDIDATE_SCALE,
    seed: 42,
    geometries: true,
    tileScale: 4
  });
  // Re-check each candidate at 10 m and keep only exact label matches.
  var checked = embeddings.addBands(labels.rename('class10')).sampleRegions({
    collection: candidates,
    properties: ['class'],
    scale: SAMPLE_SCALE,
    geometries: true,
    tileScale: 4
  }).filter(ee.Filter.equals({leftField: 'class', rightField: 'class10'}));

  var perClass = ee.FeatureCollection(ee.List(CLASS_IDS).map(function (id) {
    return checked.filter(ee.Filter.eq('class', id)).limit(N_SAMPLES);
  })).flatten();

  return addBlocks(perClass).map(function (f) {
    var id = ee.Number(f.get('class'));
    return f.set({region: name, class_name: ee.List(CLASS_NAMES).get(id.subtract(1))});
  });
}

var samples = SAMPLES_ASSET
  ? ee.FeatureCollection(SAMPLES_ASSET)
  : ee.FeatureCollection(REGION_NAMES.map(sampleRegion)).flatten();

print('Samples per region', samples.aggregate_histogram('region'));
print('Samples per class', samples.aggregate_histogram('class_name'));


// ---------------------------------------------------------------------------
// 4. kNN EVALUATION
// ---------------------------------------------------------------------------

function knn(training) {
  return ee.Classifier.smileKNN(K_NEIGHBORS)
    .train({features: training, classProperty: 'class', inputProperties: EMBED_BANDS});
}

function scores(predicted) {
  var m = predicted.errorMatrix('class', 'pred', CLASS_IDS);
  var recall = ee.Number(m.producersAccuracy().get([0, 0]));    // true mangroves found
  var precision = ee.Number(m.consumersAccuracy().get([0, 0])); // "mangrove" calls that are right
  return {
    matrix: m,
    dict: ee.Dictionary({
      overall_accuracy: m.accuracy(),
      kappa: m.kappa(),
      mangrove_recall: recall,
      mangrove_precision: precision,
      mangrove_f1: recall.multiply(precision).multiply(2).divide(recall.add(precision).max(1e-9))
    })
  };
}

// What true class do the false "mangrove" calls come from?
function falseMangroveSources(predicted) {
  return predicted.filter(ee.Filter.eq('pred', 1)).filter(ee.Filter.neq('class', 1))
    .aggregate_histogram('class_name');
}

// a) Pooled spatial-block CV across all regions (test blocks never seen in training).
var cvPredictions = ee.FeatureCollection(ee.List.sequence(0, N_FOLDS - 1).map(function (k) {
  return samples.filter(ee.Filter.eq('fold', k))
    .classify(knn(samples.filter(ee.Filter.neq('fold', k))), 'pred');
})).flatten();
var cv = scores(cvPredictions);
print('A. Spatial-block CV, all regions pooled', cv.dict, cv.matrix);

// b) Leave-one-region-out: train on the other regions, test on the unseen one.
var loroPredictions = ee.FeatureCollection(REGION_NAMES.map(function (name) {
  return samples.filter(ee.Filter.eq('region', name))
    .classify(knn(samples.filter(ee.Filter.neq('region', name))), 'pred');
})).flatten();

var loroTable = ee.FeatureCollection(REGION_NAMES.map(function (name) {
  var predicted = loroPredictions.filter(ee.Filter.eq('region', name));
  return ee.Feature(null, scores(predicted).dict)
    .set({region: name, false_mangrove_from: falseMangroveSources(predicted)});
}));
print('B. Leave-one-region-out, per held-out region', loroTable);

var loroAll = scores(loroPredictions);
print('B. Leave-one-region-out, all held-out predictions', loroAll.dict, loroAll.matrix);
print('False "mangrove" calls by true class (leave-one-region-out)',
  falseMangroveSources(loroPredictions));


// ---------------------------------------------------------------------------
// 5. MAP: FOCUS_REGION predicted by a model that never saw it
// ---------------------------------------------------------------------------

var focus = ee.Geometry.BBox.apply(null, REGIONS[FOCUS_REGION]);
var heldOutModel = knn(samples.filter(ee.Filter.neq('region', FOCUS_REGION)));
var predictedMap = embeddings.clip(focus).classify(heldOutModel);

Map.centerObject(focus, 10);
Map.setOptions('SATELLITE');
Map.addLayer(predictedMap, {min: 1, max: 11, palette: PALETTE}, 'Predicted classes (held out)', false);
Map.addLayer(mangrove.selfMask().clip(focus), {palette: ['00FF00']}, 'GMW mangroves (label)', false, 0.6);
Map.addLayer(predictedMap.eq(1).selfMask(), {palette: ['FF00FF']}, 'Predicted mangrove (held out)', true, 0.6);
Map.addLayer(labels.clip(focus), {min: 1, max: 11, palette: PALETTE}, 'Training labels', false);
Map.addLayer(coastal.selfMask().clip(focus), {palette: ['00FFFF'], opacity: 0.3}, 'Coastal zone', false);
Map.addLayer(samples.filter(ee.Filter.eq('region', FOCUS_REGION)).style({color: '000000', pointSize: 2}),
  {}, 'Sample points', false);
REGION_NAMES.forEach(function (name) {
  Map.addLayer(ee.Image().paint(ee.Geometry.BBox.apply(null, REGIONS[name]), 0, 2),
    {palette: ['FFFFFF']}, 'Region: ' + name, name === FOCUS_REGION);
});


// ---------------------------------------------------------------------------
// 6. EXPORT (skipped when evaluating a saved asset)
// ---------------------------------------------------------------------------

if (!SAMPLES_ASSET) {
  Export.table.toAsset({
    collection: samples,
    description: 'mangrove_samples_asset_v3',
    assetId: EXPORT_ASSET
  });
  Export.table.toDrive({
    collection: samples,
    description: 'mangrove_samples_csv_v3',
    fileNamePrefix: EXPORT_DRIVE,
    fileFormat: 'CSV',
    selectors: ee.List(['class', 'class_name', 'region', 'block', 'fold', '.geo'])
      .cat(EMBED_BANDS).getInfo()
  });
  print('Start the exports in the Tasks tab, then set SAMPLES_ASSET to ' + EXPORT_ASSET + ' and Run again.');
}
