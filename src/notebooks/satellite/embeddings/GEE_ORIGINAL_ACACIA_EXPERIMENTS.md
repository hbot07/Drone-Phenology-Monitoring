# GEE Original Acacia Embedding Experiments

Date: 12 June 2026

This folder contains the current Acacia/non-Acacia experiments using Google Earth Engine Satellite Embedding V1 annual embeddings extracted over original crown geometries.

## Current Scope

- Data source: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Year: 2024
- Geometry: original crown polygons
- Buffer: none
- Predictor columns: `A00` to `A63`
- Main task: Acacia vs non-Acacia

## Key Files

Prepared data:

- `exports/normalized_iitd_sv_gee_embeddings_2024_original_visual_acacia.csv`
- `exports/gee_original_acacia_label_configs.csv`

Scripts:

- `prepare_visual_acacia_labels.py`
- `prepare_acacia_label_configs.py`
- `analyze_acacia_config_errors.py`
- `visual_label_quality_experiments.py`
- `rank_acacia_review_candidates.py`
- `summarize_results.py`
- `train_best_classifiers.py`

Outputs:

- `outputs/gee_original_acacia_label_configs_full/`
- `outputs/gee_original_acacia_visual_quality/`
- `outputs/gee_original_acacia_review_candidates/`

Models:

- `models/gee_original_acacia_label_configs/`

## Label Mapping Correction

The visual label sheet image names are zero-based, while crown IDs in the GeoJSON are one-based.

Example:

- `s1_tree_006.tif` maps to `SV_S1:crown_0007`

The visual preparation script uses `--id-offset 1` by default.

## Label Quality Summary

| Comparison | Comparable Crowns | Agreement |
|---|---:|---:|
| Visual vs species | 24 | 83.3% |
| Clustering vs species | 108 | 80.6% |
| Clustering vs species on same visual-overlap crowns | 24 | 66.7% |
| Visual vs clustering | 400 | 86.0% |

Visual labels appear cleaner than clustering labels, but the species-overlap set is small.

## Model Summary

Balanced accuracy:

| Label Config | Random Split | Leave-Site-Out Range |
|---|---:|---:|
| Visual-only | 0.776 | 0.567-0.812 |
| Visual or species | 0.837 | 0.572-0.700 |
| Clustering-only | 0.696 | 0.597-0.636 |
| Visual or clustering | 0.687 | 0.585-0.635 |
| Species or clustering | 0.725 | 0.579-0.656 |
| All priority | 0.721 | 0.578-0.631 |
| Species-only | 0.934 | 0.658-0.681 |

The recommended baseline is the visual-only classifier because it uses the cleanest label source.

Recommended artifact:

- `models/gee_original_acacia_label_configs/label_acacia_visual_rf_balanced.joblib`

## Main Diagnosis

Performance is limited by:

- Only 400 clean visual labels
- Small crowns relative to 10 m satellite embedding pixels
- Site-to-site distribution shift
- Noise in clustering-based labels

Adding clustering labels increases the number of training rows but generally does not improve leave-site-out generalization.

## Annotation Review Queue

The visual-only classifier was used to score all 3,212 crowns and create ranked manual-review queues.

Output folder:

- `outputs/gee_original_acacia_review_candidates/`

Summary:

| Quantity | Count |
|---|---:|
| Total crowns scored | 3,212 |
| Already visually labelled | 400 |
| Unlabelled by visual annotation | 2,812 |
| Unlabelled predicted Acacia | 915 |
| Unlabelled predicted non-Acacia | 1,897 |
| Unlabelled uncertain, probability 0.4-0.6 | 1,347 |
| Model-vs-clustering disagreements | 683 |
| High-confidence unlabelled, confidence >= 0.80 | 407 |
| High-confidence unlabelled, confidence >= 0.90 | 91 |

Regenerate:

```bash
python3 src/notebooks/satellite/embeddings/rank_acacia_review_candidates.py
```

## GEE-Only Next Step

The active direction is to stay with Google Earth Engine Satellite Embedding V1 and improve the label/evaluation workflow rather than switching imagery sources.

Recommended GEE-only experiments:

- compare original crown polygons, centroids, and centroid buffers;
- fuse geometry variants only when leave-area-out performance improves;
- prioritize visual-only Acacia and species-only Acacia for clean evaluation;
- treat clustering-heavy labels as review/triage signals, not primary training labels;
- report leave-area-out before random split.

Example geometry exports:

```bash
python3 scripts/extract_gee_embeddings.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --year 2024 \
  --geometry-mode original \
  --description iitd_sv_gee_embeddings_2024_original
```

```bash
python3 scripts/extract_gee_embeddings.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --year 2024 \
  --geometry-mode centroid_buffer \
  --buffer-meters 20 \
  --description iitd_sv_gee_embeddings_2024_buffer20m
```

## Regenerating Results

Prepare visual labels:

```bash
python3 src/notebooks/satellite/embeddings/prepare_visual_acacia_labels.py
```

Prepare label configs:

```bash
python3 src/notebooks/satellite/embeddings/prepare_acacia_label_configs.py
```

Run model suite:

```bash
python3 src/notebooks/satellite/drone_phenology_rf_package/python/12_run_model_suite.py \
  --csv src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv \
  --label label_acacia_species \
  --label label_acacia_clustering \
  --label label_acacia_visual \
  --label label_acacia_visual_or_species \
  --label label_acacia_visual_or_clustering \
  --label label_acacia_species_or_clustering \
  --label label_acacia_all_priority \
  --trees 300 \
  --max-holdouts 4 \
  --outdir src/notebooks/satellite/embeddings/outputs/gee_original_acacia_label_configs_full
```

Train best classifiers:

```bash
python3 src/notebooks/satellite/embeddings/train_best_classifiers.py \
  --csv src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv \
  --sweep-dir src/notebooks/satellite/embeddings/outputs/gee_original_acacia_label_configs_full \
  --models-dir src/notebooks/satellite/embeddings/models \
  --trees 500
```

Run visual-label quality checks:

```bash
python3 src/notebooks/satellite/embeddings/visual_label_quality_experiments.py
```
