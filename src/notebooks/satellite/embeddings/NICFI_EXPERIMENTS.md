# Planet NICFI Crown Feature Experiments

Status: paused. The current project direction is to skip Planet/NICFI and continue with GEE Satellite Embedding V1 only. This file is kept as an archive in case NICFI access becomes useful later.

This is the finer-resolution comparison lane for the current GEE Satellite Embedding experiments.

## Why This Test

The strongest current results are from GEE annual embeddings over original crown geometries, but those embeddings are still effectively 10 m pixels. Planet NICFI basemaps in Earth Engine provide 4.77 m Blue, Green, Red, and NIR bands for tropical regions, with monthly and biannual mosaics. That should reduce mixed-pixel effects for small tree crowns.

Official dataset page:

- https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_asia

Important caveat: NICFI access is restricted and must be enabled for the Earth Engine account/project before the export will work.

## Export A First NICFI Table

Start with the phenologically important pre-monsoon window:

```bash
python3 scripts/extract_gee_nicfi_features.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --region asia \
  --start-date 2024-03-01 \
  --end-date 2024-06-01 \
  --cadence monthly \
  --geometry-mode original \
  --description iitd_sv_nicfi_20240301_20240601_monthly_original \
  --no-wait
```

The exporter writes median and standard-deviation features for `B`, `G`, `R`, `N`, plus `NDVI`, `GNDVI`, `NDWI`, and `EVI` unless `--no-indices` is passed.

Recommended next variants:

```bash
python3 scripts/extract_gee_nicfi_features.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --region asia \
  --start-date 2024-03-01 \
  --end-date 2024-06-01 \
  --cadence monthly \
  --geometry-mode centroid_buffer \
  --buffer-meters 10 \
  --description iitd_sv_nicfi_20240301_20240601_monthly_buffer10 \
  --no-wait
```

```bash
python3 scripts/extract_gee_nicfi_features.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --region asia \
  --start-date 2024-01-01 \
  --end-date 2025-01-01 \
  --cadence monthly \
  --geometry-mode original \
  --description iitd_sv_nicfi_2024_monthly_original \
  --no-wait
```

## Run The Classifier Suite

After downloading the Drive CSV, normalize the GEE column names:

```bash
python3 src/notebooks/satellite/embeddings/run_gee_embedding_suite.py \
  --csv path/to/iitd_sv_nicfi_20240301_20240601_monthly_original.csv \
  --normalize-only
```

That writes a normalized CSV under `src/notebooks/satellite/embeddings/exports/`.

For the standard species/phenology labels:

```bash
python3 src/notebooks/satellite/embeddings/run_gee_embedding_suite.py \
  --csv path/to/iitd_sv_nicfi_20240301_20240601_monthly_original.csv \
  --label label_acacia \
  --label label_deciduous \
  --label label_esd \
  --label label_showy_flower \
  --label label_yellow_strict \
  --trees 300
```

For the cleaner visual Acacia experiment, attach the visual labels and derived Acacia label configurations first:

```bash
python3 src/notebooks/satellite/embeddings/prepare_visual_acacia_labels.py \
  --gee-original src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_nicfi_20240301_20240601_monthly_original.csv \
  --out-csv src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_nicfi_20240301_20240601_monthly_original_visual_acacia.csv
```

```bash
python3 src/notebooks/satellite/embeddings/prepare_acacia_label_configs.py \
  --gee-csv src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_nicfi_20240301_20240601_monthly_original_visual_acacia.csv \
  --out-csv src/notebooks/satellite/embeddings/exports/nicfi_original_acacia_label_configs.csv
```

```bash
python3 src/notebooks/satellite/drone_phenology_rf_package/python/12_run_model_suite.py \
  --csv src/notebooks/satellite/embeddings/exports/nicfi_original_acacia_label_configs.csv \
  --label label_acacia_visual \
  --label label_acacia_visual_or_species \
  --label label_acacia_species \
  --trees 300 \
  --max-holdouts 4 \
  --outdir src/notebooks/satellite/embeddings/outputs/nicfi_original_acacia_label_configs
```

## What To Compare

Use balanced accuracy and macro-F1, but prioritize leave-area-out over random split. The current GEE embedding baseline is promising on random split, yet still site-sensitive. NICFI is worth keeping only if it improves Sanjay Van holdouts such as `SV_S3` and `SV_S4`, not just random split.
