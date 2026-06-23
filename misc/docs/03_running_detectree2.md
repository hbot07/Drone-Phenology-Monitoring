# Running Detectree2

This guide covers crown detection on cleaned orthomosaic GeoTIFFs. The reusable pipeline scripts are in `src/pipeline/`.

The detection stage takes:

```text
folder of cleaned orthomosaic .tif files
```

and writes:

```text
<output_dir>/01_detectree/crowns_multithreshold/{stem}_multithreshold.gpkg
```

Each output GPKG contains multiple crown layers at different confidence thresholds.

## Setup

Create the detection environment from the repository root:

```bash
bash scripts/setup_dpm_detectree.sh
```

Then install Detectree2 and Detectron2 using versions that match your operating system, CUDA support, and PyTorch version. Test:

```bash
conda run -n dpm-detectree python -c "import detectree2, detectron2; print('ok')"
```

## Configure Paths

Copy and edit:

```bash
cp .env.example .env
```

Important values:

```bash
DPM_OM_DIR=/path/to/clean_orthomosaics
DPM_OUTPUT_DIR=output/my_run
DPM_MODEL_PATH=input/detectree_models/250312_flexi.pth
DPM_STEPS=0,1
```

Run discovery and detection:

```bash
bash src/pipeline/run_pipeline.sh
```

Equivalent explicit commands:

```bash
conda run -n dpm-detectree python src/pipeline/00_discover_oms.py \
  --om-dir /path/to/clean_orthomosaics \
  --output-dir output/my_run \
  --model-path input/detectree_models/250312_flexi.pth \
  --run-name my_run

conda run -n dpm-detectree python src/pipeline/01_crown_detection.py \
  --config output/my_run/pipeline_config.json \
  --device cpu \
  --threads 6
```

## Model Choice

Default first choice:

```text
input/detectree_models/250312_flexi.pth
```

This is the current generalist model used in this project. For a new site, start with it unless you have a reason to use a more specialised model.

Other local models may exist under:

```text
input/detectree_models/
```

Use `--model-path` or `DPM_MODEL_PATH` to choose one explicitly.

## Detection Parameters

Current defaults:

| Parameter | Default |
|---|---|
| Tile width | `25` metres |
| Tile height | `25` metres |
| Tile buffer | `15` metres |
| Confidence thresholds | `0.15` to `0.65` in steps of `0.05` |
| Fixed IoU cleanup | `0.7` |
| Simplify tolerance | `0.3` |
| Device | `cpu` |

Use `--device cuda` only when GPU inference is correctly installed and tested.

## Output Layers

The crown GPKG is multi-layer. Common layers:

```text
conf_0p15
conf_0p45
conf_0p65
```

Lower thresholds give denser crown sets. Higher thresholds give cleaner but sparser crown sets. Tracking can choose different layers later with `--base-threshold-tag` and `--align-threshold-tag`.

## Quality Check

Before tracking, open a few crown GPKGs in QGIS with the matching orthomosaic. Check:

1. Crown boundaries line up with the raster.
2. The crown count is plausible.
3. The chosen threshold layer is neither too sparse nor wildly over-detected.
4. No date is obviously corrupted.
5. All expected GPKG layers exist.

## Reusing Existing Crowns

If crown detection already exists, do not rerun it unnecessarily:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir /path/to/clean_orthomosaics \
  --crowns-dir /path/to/crowns_multithreshold \
  --output-dir output/my_run \
  --steps 0,2,3,4
```

## Troubleshooting

1. If imports fail, check `dpm-detectree`, Detectree2, and Detectron2.
2. If the model cannot be found, set `DPM_MODEL_PATH` or pass `--model-path`.
3. If one date fails, open that orthomosaic and check that it is a valid GeoTIFF.
4. If crowns are shifted or nonsensical, inspect the orthomosaic before tuning Detectree2.
5. If later tracking cannot read a GPKG, check that the expected confidence layers are present.
