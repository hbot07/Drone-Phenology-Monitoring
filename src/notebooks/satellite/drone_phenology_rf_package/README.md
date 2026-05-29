# Drone Phenology Spatial Classifier Starter Package

This package prepares crown-level labels for the current IITD + Sanjay Van data and gives you a Google Earth Engine starter script for simple Random Forest classifiers.

## What is already prepared

### Main data

- `data/iitd_sv_crowns_master_wgs84.geojson`  
  Flat, GEE-ready FeatureCollection in EPSG:4326. Upload this to Earth Engine as a table asset.

- `data/crown_label_table.csv`  
  Same flat properties as CSV for sanity checking and local analysis.

### Label properties in the GeoJSON

Each crown has these classifier columns:

| Column | Meaning |
|---|---|
| `label_esd` | 0 evergreen, 1 semi-evergreen, 2 deciduous, -1 ignore |
| `label_deciduous` | 0 not deciduous, 1 deciduous, -1 ignore |
| `label_acacia` | 0 non-acacia, 1 acacia, -1 ignore |
| `label_yellow_strict` | 0 not yellow-showy, 1 yellow-showy, -1 ignore |
| `label_yellow_broad` | 0 not yellow, 1 yellow, -1 ignore |
| `label_red_showy` | 0 not red-showy, 1 red-showy, -1 ignore |
| `label_showy_flower` | 0 not showy-flowering, 1 showy-flowering, -1 ignore |

### Current usable label counts

| Label task | Ignore | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|---:|
| `label_esd` | 2870 | 75 evergreen | 152 semi-evergreen | 115 deciduous |
| `label_deciduous` | 2870 | 227 not deciduous | 115 deciduous | - |
| `label_acacia` | 2821 | 326 non-acacia | 65 acacia | - |
| `label_yellow_strict` | 2857 | 315 not yellow-showy | 40 yellow-showy | - |
| `label_yellow_broad` | 2857 | 255 not yellow | 100 yellow | - |
| `label_red_showy` | 2857 | 343 not red-showy | 12 red-showy | - |
| `label_showy_flower` | 2857 | 292 not showy | 63 showy | - |

## Recommended first experiments

Run these first, in this order:

1. `label_esd`, random split, buffer 20 m, year 2025  
2. `label_esd`, leave-one-area-out for `SIT`, `A3`, `A4`, `SV_S1`, `SV_S4`  
3. `label_deciduous`, same validation modes  
4. `label_acacia`, leave-one-area-out over the Sanjay Van areas  
5. `label_yellow_strict`, random split + leave-one-species-out for `Amaltas`, `Kasod`, `Caribbean trumpet`  
6. `label_showy_flower`, random split + leave-one-species-out  
7. `label_red_showy` only as an exploratory underpowered run

## Google Earth Engine steps

### Step 1: Upload the crown GeoJSON

In Earth Engine Code Editor:

1. Open the **Assets** tab.
2. Click **New** → **Table upload**.
3. Upload `data/iitd_sv_crowns_master_wgs84.geojson`.
4. Name it something like `iitd_sv_crowns_master_wgs84`.
5. Wait for ingestion to complete.

### Step 2: Paste the script

Open `gee/rf_crown_classifiers.js`, paste it into a new GEE script, and change:

```javascript
var CROWNS_ASSET = 'users/YOUR_USERNAME/iitd_sv_crowns_master_wgs84';
var LABEL_PROPERTY = 'label_esd';
var YEAR = 2025;
var GEOMETRY_MODE = 'buffer';
var BUFFER_METERS = 20;
var SPLIT_MODE = 'random';
```

### Step 3: Run one model

Start with:

```javascript
var LABEL_PROPERTY = 'label_esd';
var SPLIT_MODE = 'random';
```

Then try:

```javascript
var SPLIT_MODE = 'leave_area_out';
var HOLDOUT_AREAS = ['SIT'];
```

and later:

```javascript
var SPLIT_MODE = 'leave_species_out';
var HOLDOUT_SPECIES = ['Amaltas'];
```

### Step 4: Export predictions/features

The script creates two Drive export tasks:

- predictions for all crowns used in the task
- extracted features for all crowns used in the task

Run the tasks from the **Tasks** panel.

## Local Python steps after GEE export

After downloading the GEE CSV export, run:

```bash
python python/02_local_rf_from_gee_export.py \
  --csv exports/crown_rf_export_features_label_esd_buffer_2025.csv \
  --label label_esd \
  --split random
```

For leave-area-out:

```bash
python python/02_local_rf_from_gee_export.py \
  --csv exports/crown_rf_export_features_label_acacia_buffer_2025.csv \
  --label label_acacia \
  --split leave_area_out \
  --holdout SV_S4
```

This local script gives you:

- confusion matrix
- accuracy
- balanced accuracy
- macro-F1
- classification report
- feature importance CSV

## Local Sentinel-2 route without GEE

This package now also includes a Microsoft Planetary Computer STAC extractor:

```bash
python python/03_extract_sentinel2_stac_features.py \
  --year 2025 \
  --geometry-mode buffer \
  --buffer-meters 20 \
  --max-items-per-season 4 \
  --label-filter label_esd \
  --out-csv exports/stac_s2_features_2025_buffer20_items4_label_esd.csv
```

Then train/evaluate the same local RF script:

```bash
python python/02_local_rf_from_gee_export.py \
  --csv exports/stac_s2_features_2025_buffer20_items4_label_esd.csv \
  --label label_acacia \
  --split random \
  --outdir outputs/local_rf_stac
```

Notes:

- The STAC route uses Sentinel-2 L2A assets from Microsoft Planetary Computer.
- It applies SCL cloud/shadow/water masking and extracts seasonal median features over polygon, point, or centroid-buffer geometries.
- It is much slower than GEE because it streams COG windows band-by-band locally.
- Current local baseline outputs are in `outputs/local_rf_stac/`.

## Should you use GEE or local Python?

Use both:

- Use **Python locally** for data cleaning, label configs, sanity checks, plots, repeated validation, and final reports.
- Use **GEE** for extracting Sentinel-2 features and quick Random Forest baselines, because it avoids downloading and cloud-masking large satellite imagery manually.

For your first deliverable to sir, use the GEE script directly. Once it works, export the features and do cleaner evaluation locally in Python.

## Important caveats

- Many crown polygons are unlabeled; they are kept in the GeoJSON but have label value `-1` and are ignored during training.
- Ambiguous species such as mixed labels are intentionally not forced into a class.
- Exact drone crowns are much finer than Sentinel-2 pixels. Test centroid 10 m and 20 m buffers; these may work better than raw crown polygons.
- Do not trust random split alone. Always report leave-one-area-out and leave-one-species-out results.
- Red flower classification is currently underpowered.
- Acacia classification is now trainable only because Sanjay Van added field Acacia positives.
