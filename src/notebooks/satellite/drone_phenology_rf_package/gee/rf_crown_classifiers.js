/****
Drone Phenology Monitoring: crown-level spatial classifiers in Google Earth Engine

Before running:
1. Upload data/iitd_sv_crowns_master_wgs84.geojson as a GEE table asset.
2. Replace CROWNS_ASSET below with your uploaded asset id.
3. Pick LABEL_PROPERTY from one of:
   - label_esd             : 0 evergreen, 1 semi-evergreen, 2 deciduous
   - label_deciduous       : 0 not deciduous, 1 deciduous
   - label_acacia          : 0 non-acacia, 1 acacia
   - label_yellow_strict   : 0 not yellow-showy, 1 yellow-showy
   - label_yellow_broad    : 0 not yellow, 1 yellow
   - label_red_showy       : 0 not red-showy, 1 red-showy
   - label_showy_flower    : 0 not showy, 1 showy
****/

// ================= USER SETTINGS =================
var CROWNS_ASSET = 'users/YOUR_USERNAME/iitd_sv_crowns_master_wgs84';
var LABEL_PROPERTY = 'label_esd';
var YEAR = 2025;  // Use a complete year first. Change later to match your OM year(s).
var GEOMETRY_MODE = 'buffer';  // 'polygon', 'point', or 'buffer'
var BUFFER_METERS = 20;        // Used only when GEOMETRY_MODE = 'buffer'
var SPLIT_MODE = 'random';     // 'random', 'leave_area_out', or 'leave_species_out'
var HOLDOUT_AREAS = ['SIT'];   // e.g. ['SIT'], ['A3'], ['SV_S1']
var HOLDOUT_SPECIES = ['Amaltas'];
var RANDOM_TRAIN_FRACTION = 0.70;
var SEED = 42;
var N_TREES = 200;
var EXPORT_PREFIX = 'crown_rf_export';
// =================================================

var crownsRaw = ee.FeatureCollection(CROWNS_ASSET);
var roi = crownsRaw.geometry().bounds();
Map.centerObject(roi, 15);
Map.addLayer(crownsRaw, {}, 'crowns raw', false);

// Convert geometry for feature extraction.
function setExtractionGeometry(f) {
  if (GEOMETRY_MODE === 'point') {
    return f.setGeometry(f.geometry().centroid(1));
  }
  if (GEOMETRY_MODE === 'buffer') {
    return f.setGeometry(f.geometry().centroid(1).buffer(BUFFER_METERS));
  }
  return f;
}

var crowns = crownsRaw
  .filter(ee.Filter.neq(LABEL_PROPERTY, -1))
  .map(setExtractionGeometry);

print('Label property', LABEL_PROPERTY);
print('Usable labelled crowns', crowns.size());
print('Label histogram', crowns.aggregate_histogram(LABEL_PROPERTY));
print('Area histogram', crowns.aggregate_histogram('area'));
print('Species histogram', crowns.aggregate_histogram('species_clean'));

// Sentinel-2 SR Harmonized feature stack.
var OPTICAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var INDEX_BANDS = [
  'NDVI','GNDVI','NDRE','NDMI','NBR','EVI',
  'blue_ratio','green_ratio','red_ratio','yellow_proxy'
];
var ALL_COMPOSITE_BANDS = OPTICAL_BANDS.concat(INDEX_BANDS);

function maskS2Sr(img) {
  // SCL codes masked: 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow/ice.
  var scl = img.select('SCL');
  var clear = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  var scaled = img.select(OPTICAL_BANDS).divide(10000);
  return img.addBands(scaled, null, true).updateMask(clear);
}

function addIndices(img) {
  var blue = img.select('B2');
  var green = img.select('B3');
  var red = img.select('B4');
  var nir = img.select('B8');
  var nirNarrow = img.select('B8A');
  var redEdge1 = img.select('B5');
  var swir1 = img.select('B11');
  var swir2 = img.select('B12');
  var rgbSum = blue.add(green).add(red).max(0.0001);

  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI');
  var ndre = nirNarrow.subtract(redEdge1).divide(nirNarrow.add(redEdge1)).rename('NDRE');
  var ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename('NDMI');
  var nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR');
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {NIR: nir, RED: red, BLUE: blue}
  ).rename('EVI');

  var blueRatio = blue.divide(rgbSum).rename('blue_ratio');
  var greenRatio = green.divide(rgbSum).rename('green_ratio');
  var redRatio = red.divide(rgbSum).rename('red_ratio');
  var yellowProxy = green.add(red).divide(rgbSum).rename('yellow_proxy');

  return img.addBands([
    ndvi, gndvi, ndre, ndmi, nbr, evi,
    blueRatio, greenRatio, redRatio, yellowProxy
  ]);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(ee.Date.fromYMD(YEAR - 1, 12, 1), ee.Date.fromYMD(YEAR + 1, 1, 1))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
  .map(maskS2Sr)
  .map(addIndices);

function renameWithPrefix(img, prefix) {
  var oldNames = img.bandNames();
  var newNames = oldNames.map(function(name) {
    return ee.String(prefix).cat('_').cat(ee.String(name));
  });
  return img.rename(newNames);
}

function compositeDates(start, end, prefix) {
  var img = s2.filterDate(start, end).median().select(ALL_COMPOSITE_BANDS);
  return renameWithPrefix(img, prefix);
}

// Winter is Dec of previous year through Feb of YEAR.
var winter = compositeDates(ee.Date.fromYMD(YEAR - 1, 12, 1), ee.Date.fromYMD(YEAR, 3, 1), 'winter');
var premonsoon = compositeDates(ee.Date.fromYMD(YEAR, 3, 1), ee.Date.fromYMD(YEAR, 6, 1), 'premonsoon');
var monsoon = compositeDates(ee.Date.fromYMD(YEAR, 6, 1), ee.Date.fromYMD(YEAR, 10, 1), 'monsoon');
var postmonsoon = compositeDates(ee.Date.fromYMD(YEAR, 10, 1), ee.Date.fromYMD(YEAR, 12, 1), 'postmonsoon');

var seasonalStack = ee.Image.cat([winter, premonsoon, monsoon, postmonsoon]);

function amplitudeBand(bandName) {
  var imgs = ee.ImageCollection.fromImages([
    seasonalStack.select('winter_' + bandName).rename(bandName),
    seasonalStack.select('premonsoon_' + bandName).rename(bandName),
    seasonalStack.select('monsoon_' + bandName).rename(bandName),
    seasonalStack.select('postmonsoon_' + bandName).rename(bandName)
  ]);
  return imgs.max().subtract(imgs.min()).rename(bandName + '_amp');
}

var amplitudeStack = ee.Image.cat([
  amplitudeBand('NDVI'), amplitudeBand('GNDVI'), amplitudeBand('NDRE'),
  amplitudeBand('NDMI'), amplitudeBand('NBR'), amplitudeBand('EVI'),
  amplitudeBand('green_ratio'), amplitudeBand('red_ratio'), amplitudeBand('yellow_proxy')
]);

var featureStack = seasonalStack.addBands(amplitudeStack);
var predictorBands = featureStack.bandNames();
print('Predictor band count', predictorBands.size());
print('Predictor bands', predictorBands);

Map.addLayer(featureStack.select('premonsoon_NDVI'), {min: 0, max: 0.8}, 'premonsoon NDVI', false);
Map.addLayer(featureStack.select('NDVI_amp'), {min: 0, max: 0.5}, 'NDVI amplitude', false);

// Extract one row per crown/buffer.
var samples = featureStack.reduceRegions({
  collection: crowns,
  reducer: ee.Reducer.median(),
  scale: 10,
  tileScale: 4
}).filter(ee.Filter.notNull(predictorBands));

print('Samples after feature extraction', samples.size());
print('Samples label histogram', samples.aggregate_histogram(LABEL_PROPERTY));

// Train/test splitting.
var train;
var test;
if (SPLIT_MODE === 'leave_area_out') {
  test = samples.filter(ee.Filter.inList('area', HOLDOUT_AREAS));
  train = samples.filter(ee.Filter.inList('area', HOLDOUT_AREAS).not());
} else if (SPLIT_MODE === 'leave_species_out') {
  test = samples.filter(ee.Filter.inList('species_clean', HOLDOUT_SPECIES));
  train = samples.filter(ee.Filter.inList('species_clean', HOLDOUT_SPECIES).not());
} else {
  var withRandom = samples.randomColumn('random', SEED);
  train = withRandom.filter(ee.Filter.lt('random', RANDOM_TRAIN_FRACTION));
  test = withRandom.filter(ee.Filter.gte('random', RANDOM_TRAIN_FRACTION));
}

print('Train size', train.size());
print('Test size', test.size());
print('Train labels', train.aggregate_histogram(LABEL_PROPERTY));
print('Test labels', test.aggregate_histogram(LABEL_PROPERTY));

var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: N_TREES,
  variablesPerSplit: null,
  minLeafPopulation: 2,
  bagFraction: 0.70,
  seed: SEED
}).train({
  features: train,
  classProperty: LABEL_PROPERTY,
  inputProperties: predictorBands
});

print('RF explanation / feature importance', rf.explain());

var trainPred = train.classify(rf);
var testPred = test.classify(rf);
var allPred = samples.classify(rf);

var trainCM = trainPred.errorMatrix(LABEL_PROPERTY, 'classification');
var testCM = testPred.errorMatrix(LABEL_PROPERTY, 'classification');
print('Train confusion matrix', trainCM);
print('Train accuracy', trainCM.accuracy());
print('Train kappa', trainCM.kappa());
print('Test confusion matrix', testCM);
print('Test accuracy', testCM.accuracy());
print('Test kappa', testCM.kappa());
print('Test producer accuracy', testCM.producersAccuracy());
print('Test consumer accuracy', testCM.consumersAccuracy());

// Probability export. For binary classifiers this is a scalar probability for class 1.
// For multiclass, use hard classification first; MULTIPROBABILITY can be added later if needed.
var probabilitySamples = allPred;
var isBinary = ee.Number(samples.aggregate_count_distinct(LABEL_PROPERTY)).eq(2);
print('Binary label task?', isBinary);

// Export feature table + predictions to Drive. You can then analyze locally in Python.
Export.table.toDrive({
  collection: allPred,
  description: EXPORT_PREFIX + '_' + LABEL_PROPERTY + '_' + GEOMETRY_MODE + '_' + YEAR,
  fileFormat: 'CSV'
});

// Optional: export the extracted features without prediction too.
Export.table.toDrive({
  collection: samples,
  description: EXPORT_PREFIX + '_features_' + LABEL_PROPERTY + '_' + GEOMETRY_MODE + '_' + YEAR,
  fileFormat: 'CSV'
});
