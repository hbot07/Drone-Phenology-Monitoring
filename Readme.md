# Drone Phenology Monitoring

This repository supports a repeated-drone-imagery workflow for monitoring tree phenology at crown level.

The high-level flow is:

1. Plan and fly repeated nadir drone missions over the same area.
2. Build one georeferenced orthomosaic per site/date using ODM/WebODM/NodeODM.
3. Detect individual tree crowns on each orthomosaic with Detectree2.
4. Track crowns through time with a graph-based matching pipeline.
5. Build consensus crown geometries from each tracked chain.
6. Crop every tracked tree from every orthomosaic.
7. Extract crown-level phenology signals such as GCC, RCC, grayscale texture, Laplacian variance, and vegetation fractions.
8. Label tree type and leaf-on/leaf-off state from the time series.
9. Use the drone-derived labels/signals as fine-grained reference data for satellite experiments.

## Start Here

Operational guides are in [misc/docs](misc/docs/README.md):

1. [Collecting drone imagery](misc/docs/01_collecting_drone_imagery.md)
2. [Building orthomosaics](misc/docs/02_building_orthomosaic_webodm.md)
3. [Running Detectree2](misc/docs/03_running_detectree2.md)
4. [Creating orthomosaic printouts with crown IDs](misc/docs/04_orthomosaic_printout_crown_ids.md)
5. [Preparing QField crown annotation projects](misc/docs/05_qfield_crown_annotation.md)

The reusable analysis pipeline is documented in [src/pipeline/README.md](src/pipeline/README.md).

## Setup

Create the two conda environments:

```bash
bash scripts/setup_dpm_detectree.sh
bash scripts/setup_dpm_tracking.sh
```

`dpm-detectree` is used for crown detection. `dpm-tracking` is used for tracking, consensus crowns, phenology, and viewer generation.

Detectree2 and Detectron2 may need platform-specific installation commands depending on CUDA/PyTorch support. The setup script installs the common geospatial/PyTorch base packages and then prints the import test to run after installing those two libraries.

## Configure A Run

Copy the example environment file and edit paths for your machine:

```bash
cp .env.example .env
```

Important paths:

- `DPM_OM_DIR`: folder of cleaned orthomosaic `.tif` files.
- `DPM_MODEL_PATH`: Detectree2 model path, or leave empty for auto-discovery under `input/detectree_models/`.
- `DPM_OUTPUT_DIR`: output folder for this run.
- `DPM_CROWNS_DIR`: optional existing crown-detection folder if you want to skip Detectree2.

Then run:

```bash
bash src/pipeline/run_pipeline.sh
```

You can also pass all paths as CLI flags instead of using `.env`.

## Main Outputs

A full pipeline run writes:

- `01_detectree/crowns_multithreshold/*.gpkg`: crown detections at multiple confidence thresholds.
- `02_tracking/consensus_crowns_complete_all.gpkg`: final tracked consensus crowns.
- `03_phenology/tree_master_geojson.geojson`: crown geometry plus phenology outputs.
- `04_viewer/index.html`: standalone interactive viewer.

## Repository Map

- `src/pipeline/`: reusable pipeline entry points.
- `src/flask_app_tracking/`: tracking and phenology implementation modules used by the pipeline.
- `misc/docs/`: generic workflow guides.
- `misc/ODM/`: ODM/NodeODM scripts and runbooks.
- `envs/`: conda environment specifications.
- `requirements/`: companion pip requirements.
- `scripts/`: setup and satellite/remote-sensing helpers.
- `src/notebooks/` and `src/notebook_archive/`: exploratory work and historical notebooks.
