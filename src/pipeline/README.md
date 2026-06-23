# Drone Phenology Pipeline

This is the reusable pipeline for going from dated orthomosaics to tracked consensus crowns, crown-level phenology metrics, and a browser viewer.

Flow:

```text
orthomosaics (.tif)
  -> 00_discover_oms.py
  -> 01_crown_detection.py
  -> 02_crown_tracking.py
  -> 03_phenology_analysis.py
  -> 04_interactive_viz.py
```

All steps share one `pipeline_config.json`, created by Step 0 and updated by later steps.

## Environments

Use two conda environments:

| Step | Environment | Purpose |
|---|---|---|
| 0 | `dpm-detectree` | discover inputs and write config |
| 1 | `dpm-detectree` | Detectree2 crown detection |
| 2 | `dpm-tracking` | crown alignment, graph tracking, consensus geometries |
| 3 | `dpm-tracking` | phenology metrics and labels |
| 4 | `dpm-tracking` | standalone HTML viewer |

Create or update them from the repository root:

```bash
bash scripts/setup_dpm_detectree.sh
bash scripts/setup_dpm_tracking.sh
```

The conda YAML files live in `envs/`. Companion pip requirement files live in `requirements/`.

## Input Naming

The pipeline expects one orthomosaic GeoTIFF per site/date. File stems should contain enough information to sort dates correctly.

Recommended names:

```text
<site>_DD-MM-YY.tif
<site>_DD-MM-YY_dateNotConfirmed.tif
<site>_spot<id>_DD-MM-YY.tif
```

Examples:

```text
site_a_15-01-26.tif
site_a_29-01-26.tif
sv_spot1_10-05-26.tif
```

For older local datasets, the discovery step also understands `lhc_DD-MM-YY`, `sit_DD-MM-YY`, `sv_spotX_DD-MM-YY`, `sit_omN`, and legacy `odm_orthophoto_D_M_YY` stems.

## Configure Paths With `.env`

Copy the root example file:

```bash
cp .env.example .env
```

Set at least:

```bash
DPM_OM_DIR=input/input_om_sit
DPM_OUTPUT_DIR=output/example_sit_run
DPM_MODEL_PATH=input/detectree_models/250312_flexi.pth
DPM_STEPS=0,1,2,3,4
```

Then run:

```bash
bash src/pipeline/run_pipeline.sh
```

CLI flags override `.env`, so this also works:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir /path/to/clean_orthomosaics \
  --output-dir /path/to/output/my_run \
  --model-path /path/to/model.pth \
  --steps 0,1,2,3,4
```

If crown detection was already run, reuse the crowns and skip Step 1:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir /path/to/clean_orthomosaics \
  --crowns-dir /path/to/crowns_multithreshold \
  --output-dir /path/to/output/my_run \
  --steps 0,2,3,4
```

## Step 0: Discover Orthomosaics

Script: `00_discover_oms.py`

```bash
conda run -n dpm-detectree python src/pipeline/00_discover_oms.py \
  --om-dir /path/to/clean_orthomosaics \
  --output-dir output/my_run \
  --model-path input/detectree_models/250312_flexi.pth \
  --run-name my_run
```

Useful options:

- `--exclude-stems stem1,stem2`: remove bad dates from the series.
- `--only-stems stem1,stem2`: run a quick subset.
- `--crowns-dir /path/to/crowns_multithreshold`: reuse existing Detectree2 outputs.
- `--print-config`: print the full generated config.

Output:

```text
<output_dir>/pipeline_config.json
```

## Step 1: Detect Crowns With Detectree2

Script: `01_crown_detection.py`

```bash
conda run -n dpm-detectree python src/pipeline/01_crown_detection.py \
  --config output/my_run/pipeline_config.json \
  --device cpu \
  --threads 6
```

Output:

```text
<output_dir>/01_detectree/crowns_multithreshold/{stem}_multithreshold.gpkg
```

Each GPKG contains multiple layers, one per confidence threshold, such as `conf_0p15`, `conf_0p45`, and `conf_0p65`.

Use `--device cuda` only when the environment and machine support GPU inference.

## Step 2: Track Crowns And Build Consensus Geometries

Script: `02_crown_tracking.py`

```bash
conda run -n dpm-tracking python src/pipeline/02_crown_tracking.py \
  --config output/my_run/pipeline_config.json \
  --base-threshold-tag conf_0p45 \
  --align-threshold-tag conf_0p65 \
  --align-method pcc_tiled
```

Main ideas:

- Orthomosaics are aligned through phase correlation, usually `pcc_tiled`.
- Crowns are shifted using the same alignment offsets.
- Consecutive dates are matched using IoU, overlap, centroid distance, and shape similarity.
- A graph is built where nodes are dated crowns and edges connect likely same-tree crowns.
- Full and partial chains are converted into consensus crown geometries.
- Consensus crowns are deduplicated before phenology extraction.

Common tuning options:

| Goal | Option |
|---|---|
| Denser crown population | `--base-threshold-tag conf_0p15` |
| Cleaner, sparser crown population | `--base-threshold-tag conf_0p65` |
| Stronger alignment anchors | `--align-threshold-tag conf_0p65` |
| Allow shorter partial chains | lower `--min-partial-len` |
| Require stricter temporal coverage | raise `--min-partial-ratio` |
| Faster debug run | `--skip-chain-viz --skip-consensus-viz --skip-diagnostics` |

Main outputs:

```text
02_tracking/consensus_crowns_complete_all.gpkg
02_tracking/consensus_crowns_om1_phenology.geojson
02_tracking/tracking_quality_report.txt
02_tracking/tracking_quality_metrics.json
02_tracking/diagnostics/
```

## Step 3: Phenology Analysis

Script: `03_phenology_analysis.py`

```bash
conda run -n dpm-tracking python src/pipeline/03_phenology_analysis.py \
  --config output/my_run/pipeline_config.json
```

This crops each consensus crown from each orthomosaic and extracts features such as GCC, RCC, channel statistics, grayscale texture, Laplacian variance, vegetation fraction, robust date-normalized signals, deciduous scores, and leaf-on/leaf-off/transition labels.

Main outputs:

```text
03_phenology/tree_master_geojson.geojson
03_phenology/phenology_features_raw.csv
03_phenology/leafshed_tree_scores.csv
03_phenology/leafshed_phenophase_by_om.csv
```

## Step 4: Interactive Viewer

Script: `04_interactive_viz.py`

```bash
conda run -n dpm-tracking python src/pipeline/04_interactive_viz.py \
  --config output/my_run/pipeline_config.json \
  --underlay-om last
```

Output:

```text
04_viewer/index.html
```

Open `index.html` directly in a browser. No server is required.

## Output Layout

After a full run:

```text
output/my_run/
  pipeline_config.json
  01_detectree/
    crowns_multithreshold/
  02_tracking/
    consensus_crowns_complete_all.gpkg
    diagnostics/
  03_phenology/
    tree_master_geojson.geojson
    phenology_features_raw.csv
  04_viewer/
    index.html
    crops/
```

## Fast Debug Workflow

Use a few orthomosaics, skip visualizations, and run only tracking:

```bash
conda run -n dpm-detectree python src/pipeline/00_discover_oms.py \
  --om-dir input/input_om_sit \
  --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
  --only-stems sit_03-08-25_dateNotConfirmed,sit_31-08-25_dateNotConfirmed,sit_20-11-25 \
  --output-dir output/debug_run \
  --run-name debug_run

conda run -n dpm-tracking python src/pipeline/02_crown_tracking.py \
  --config output/debug_run/pipeline_config.json \
  --base-threshold-tag conf_0p15 \
  --skip-chain-viz \
  --skip-consensus-viz \
  --skip-diagnostics
```

## Appendix: Local Dataset Settings

These are examples from the current IITD/Sanjay Van work. Treat them as starting points, not generic requirements.

LHC:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir input/input_om_lhc \
  --exclude-stems lhc_09-12-25 \
  --crowns-dir output/detectree_om_lhc_multithreshold/crowns_multithreshold \
  --steps 0,2,3,4
```

The `lhc_09-12-25` orthomosaic is excluded because it is badly misaligned relative to the rest of that local series.

SIT:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir input/input_om_sit \
  --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
  --base-threshold-tag conf_0p15 \
  --steps 0,2,3,4
```

For that local SIT crown set, `conf_0p15` has been useful because higher thresholds are too sparse.

## Troubleshooting

- If Step 0 cannot find a model, set `DPM_MODEL_PATH` or pass `--model-path`.
- If Step 1 imports fail, check the `dpm-detectree` environment and Detectree2/Detectron2 installation.
- If Step 2 produces very short chains, inspect `02_tracking/diagnostics/alignment_shifts.csv` and the match-rate plots.
- If a single date breaks many chains, visually inspect that orthomosaic and consider excluding that stem.
- If phenology features are mostly missing, check whether crowns fall outside later orthomosaic bounds or whether rasters have nodata/black edges.
