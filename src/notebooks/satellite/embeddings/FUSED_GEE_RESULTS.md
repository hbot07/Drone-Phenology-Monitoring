# Fused GEE Original + Buffer20 Results

Date: 15 June 2026

## Context

Planet/NICFI is no longer part of the active experiment path. The current satellite classifier lane should stay focused on Google Earth Engine Satellite Embedding V1 and local model evaluation.

For record keeping, the attempted NICFI extractor run could read the uploaded crown asset, then failed when loading:

```text
projects/planet-nicfi/assets/basemaps/asia
```

```text
ImageCollection asset 'projects/planet-nicfi/assets/basemaps/asia' not found (does not exist or caller does not have access).
```

The useful follow-up experiment was to fuse the two existing GEE Satellite Embedding V1 tables:

- original crown geometry embeddings
- centroid/buffer20 m embeddings

Output table:

- `exports/gee_fused_original_buffer20_acacia_label_configs.csv`

This gives 128 predictor columns:

- `orig_A00` ... `orig_A63`
- `buf_A00` ... `buf_A63`

## Main Takeaways

Fusion helps a few specific tasks, but it is not universally better.

Best useful improvements:

| Task | Split | Best model | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Acacia species labels | leave `SV_S3` out | Extra Trees | 0.846 | 0.857 |
| Acacia species labels | leave `SV_S4` out | Logistic regression | 0.750 | 0.721 |
| Visual or species Acacia | random | Extra Trees + K-best + threshold | 0.849 | 0.808 |
| Yellow broad | random | Extra Trees + K-best + threshold | 0.825 | 0.818 |
| Yellow broad | leave `SV_S1` out | RF deeper | 0.733 | 0.698 |
| Yellow broad | leave `A3` out | Logistic regression | 0.677 | 0.612 |

Where fusion did not help:

- Clean visual-only Acacia still looks better on `SV_S4` with the original-only GEE baseline.
- `label_esd` remains weak, near chance.
- `label_red_showy` remains underpowered/chance.
- Broad clustering-heavy Acacia labels improve sample size but hurt leave-area transfer.
- Yellow broad is strong on random split and some holdouts, but weak on `A1`, `SIT`, `SV_S2`, and `MITTAL`.

## Recommended Current Classifiers

Use these as the current best saved artifacts:

| Purpose | Artifact |
|---|---|
| Best local/DINO Acacia random-split classifier | `models/dino_default_labels_2025_mar/label_acacia_rf_balanced.joblib` |
| Clean visual Acacia GEE baseline | `models/gee_original_acacia_label_configs/label_acacia_visual_rf_balanced.joblib` |
| Fused Acacia species classifier | `models/gee_fused_original_buffer20_acacia_label_configs/label_acacia_species_rf_balanced.joblib` |
| Fused visual-or-species Acacia classifier | `models/gee_fused_original_buffer20_acacia_label_configs/label_acacia_visual_or_species_extra_trees_kbest.joblib` |
| Fused yellow-broad classifier | `models/gee_fused_original_buffer20_default_labels/label_yellow_broad_extra_trees_kbest.joblib` |

## Output Folders

Sweeps:

- `outputs/gee_fused_original_buffer20_acacia_label_configs/`
- `outputs/gee_fused_original_buffer20_default_labels_random/`
- `outputs/gee_fused_original_buffer20_yellow_broad_holdouts/`

Models:

- `models/gee_fused_original_buffer20_acacia_label_configs/`
- `models/gee_fused_original_buffer20_default_labels/`

## Next Best Step

Stay on GEE embeddings and compare geometry/reducer choices. The immediate useful runs are:

```bash
python3 scripts/extract_gee_embeddings.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --geometry-mode original \
  --year 2024 \
  --description iitd_sv_gee_embeddings_2024_original
```

```bash
python3 scripts/extract_gee_embeddings.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --geometry-mode centroid \
  --year 2024 \
  --description iitd_sv_gee_embeddings_2024_centroid
```

```bash
python3 scripts/extract_gee_embeddings.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --geometry-mode centroid_buffer \
  --buffer-meters 20 \
  --year 2024 \
  --description iitd_sv_gee_embeddings_2024_buffer20m
```

After downloading a Drive CSV, normalize/run:

```bash
python3 src/notebooks/satellite/embeddings/run_gee_embedding_suite.py \
  --csv path/to/iitd_sv_gee_embeddings_2024_original.csv \
  --label label_acacia \
  --label label_yellow_broad \
  --trees 400
```
