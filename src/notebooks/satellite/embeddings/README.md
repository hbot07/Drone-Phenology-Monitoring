# Satellite Embedding Classifiers

This folder is the experiment lane for training the same crown classifiers with pretrained visual embeddings instead of hand-crafted Sentinel-2 band/index tables.

## What This Uses

- Labels and crown metadata from `../drone_phenology_rf_package/data/iitd_sv_crowns_master_wgs84.geojson`.
- Embedding extraction from `../drone_phenology_rf_package/python/14_extract_embedding_features.py`.
- Classifier sweeps from `../drone_phenology_rf_package/python/12_run_model_suite.py`.

The embedding extractor downloads small Sentinel-2 RGB patches from Microsoft Planetary Computer, runs a local DINOv2 encoder, and writes columns named like `dino_mar_d0000`. The model suite then treats those numeric embedding dimensions as predictor columns.

## First Run

Start with Acacia because it has a focused label set and Sanjay Van holdouts:

```bash
python src/notebooks/satellite/embeddings/run_embedding_experiment.py \
  --label label_acacia \
  --year 2025 \
  --seasons mar apr may \
  --max-items-per-season 1 \
  --random-only
```

Expected outputs:

- `src/notebooks/satellite/embeddings/exports/dino_label_acacia_2025_mar-apr-may.csv`
- `src/notebooks/satellite/embeddings/outputs/label_acacia_random_model_sweep.csv`

Remove `--random-only` once extraction works; the wrapper will also run leave-area-out sweeps over the strongest candidate holdout areas.

## Shared Multi-Label Suite

This extracts one table for the union of the default usable labels and runs all default classifier tasks:

```bash
python src/notebooks/satellite/embeddings/run_embedding_experiment.py \
  --all-default-labels \
  --year 2025 \
  --seasons mar apr may \
  --max-items-per-season 1
```

Train deployable full-data classifiers from the best random-split model per label:

```bash
python src/notebooks/satellite/embeddings/train_best_classifiers.py \
  --csv src/notebooks/satellite/embeddings/exports/dino_default_labels_2025_mar-apr-may.csv \
  --sweep-dir src/notebooks/satellite/embeddings/outputs/dino_default_labels_2025_mar-apr-may
```

## Google Earth Engine Embeddings

Install and authenticate once:

```bash
pip install earthengine-api python-dotenv
earthengine authenticate
earthengine set_project YOUR_GOOGLE_CLOUD_PROJECT_ID
```

Optional `.env` keys are shown in `scripts/gee_embeddings_env.example`.

Start a Drive export:

```bash
python scripts/extract_gee_embeddings.py \
  --project YOUR_GOOGLE_CLOUD_PROJECT_ID \
  --crowns-asset projects/YOUR_GOOGLE_CLOUD_PROJECT_ID/assets/drone_phenology/iitd_sv_crowns_master \
  --year 2024 \
  --buffer-meters 20
```

After downloading the exported CSV from Google Drive, run the same model suite:

```bash
python src/notebooks/satellite/embeddings/run_gee_embedding_suite.py \
  --csv path/to/iitd_sv_gee_embeddings_2024_buffer20m.csv
```

## Faster Plumbing Check

Use this before a full run if you only want to verify the extractor starts and writes a tiny table:

```bash
python src/notebooks/satellite/embeddings/run_embedding_experiment.py \
  --label label_acacia \
  --year 2025 \
  --seasons mar \
  --max-items-per-season 1 \
  --limit-crowns 20 \
  --extract-only
```

## Notes

- The first DINOv2 run may download weights through `torch.hub`, so it needs internet access.
- STAC patch extraction also needs internet access to stream Sentinel-2 COGs.
- DINOv2 patch sizes should be multiples of 14; the extractor snaps sizes down if needed.
- Current default is `dinov2_vits14` because it is the fastest 384-dimensional model.
