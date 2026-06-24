# Run Detectree2

Start with checked orthomosaic GeoTIFFs in a clean folder. End with crown polygon GeoPackages.

## Inputs And Outputs

Input:

```text
/path/to/clean_orthomosaics/*.tif
```

Output:

```text
<output_dir>/01_detectree/crowns_multithreshold/{orthomosaic_stem}_multithreshold.gpkg
```

Each GeoPackage contains threshold layers such as:

```text
conf_0p15
conf_0p20
conf_0p25
...
conf_0p65
```

## Environment

From the repository root:

```bash
bash scripts/setup_dpm_detectree.sh
```

Test imports:

```bash
conda run -n dpm-detectree python -c "import detectree2, detectron2; print('ok')"
```

If `conda` is not on `PATH`, set:

```bash
DPM_CONDA_EXE=<path-to-conda-executable>
```

Detectree2, Detectron2, PyTorch, CUDA, and the operating system must match each other.

## Configure

```bash
cp .env.example .env
```

Set:

```bash
DPM_OM_DIR=/path/to/clean_orthomosaics
DPM_OUTPUT_DIR=output/site_a_run
DPM_MODEL_PATH=/path/to/detectree_model.pth
DPM_STEPS=0,1
```

Run:

```bash
bash src/pipeline/run_pipeline.sh
```

This runs:

1. Step 0: discover orthomosaics and write `pipeline_config.json`.
2. Step 1: run Detectree2 and export crown polygons.

## Direct Commands

```bash
conda run -n dpm-detectree python src/pipeline/00_discover_oms.py \
  --om-dir /path/to/clean_orthomosaics \
  --output-dir output/site_a_run \
  --model-path /path/to/detectree_model.pth \
  --run-name site_a_run
```

```bash
conda run -n dpm-detectree python src/pipeline/01_crown_detection.py \
  --config output/site_a_run/pipeline_config.json \
  --device cpu \
  --threads 6
```

Use `--device cuda` only after CUDA inference has been tested.

## Model

Set the model explicitly:

```bash
DPM_MODEL_PATH=/path/to/detectree_model.pth
```

Local models may live under:

```text
input/detectree_models/
```

Record the exact model file used for each run.

## Defaults

| Parameter | Default | Use |
|---|---|---|
| Tile width | `25` m | Raster chunk width. |
| Tile height | `25` m | Raster chunk height. |
| Tile buffer | `15` m | Tile overlap to reduce edge effects. |
| Confidence thresholds | `0.15` to `0.65` | Crown-density layers. |
| Fixed IoU cleanup | `0.7` | Duplicate prediction removal. |
| Simplify tolerance | `0.3` | Polygon simplification. |
| Device | `cpu` | Inference device. |

Keep defaults for the first pass. Tune after visual inspection.

## Threshold Layers

| Layer | Behaviour | Use |
|---|---|---|
| Low, e.g. `conf_0p15` | Dense, more false positives. | Tracking when missed crowns are costly. |
| Middle, e.g. `conf_0p45` | Balanced. | General inspection and printouts. |
| High, e.g. `conf_0p65` | Cleaner, more missed crowns. | Conservative checks and alignment anchors. |

There is no universal best threshold.

## CPU Or GPU

CPU:

```bash
conda run -n dpm-detectree python src/pipeline/01_crown_detection.py \
  --config output/site_a_run/pipeline_config.json \
  --device cpu \
  --threads 6
```

GPU:

```bash
conda run -n dpm-detectree python src/pipeline/01_crown_detection.py \
  --config output/site_a_run/pipeline_config.json \
  --device cuda \
  --threads 6
```

Test CUDA:

```bash
conda run -n dpm-detectree python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Quality Check

Open each crown GeoPackage with its matching orthomosaic.

Check:

1. Polygons align with visible crowns.
2. Crown count is plausible.
3. False positives are not dominating buildings, paths, shadows, or bare ground.
4. Crowns are not systematically missing in one area.
5. The same threshold layer exists for every date.
6. One date is not dramatically worse than nearby dates.
7. Crown CRS matches the orthomosaic.

If one date is poor, inspect the orthomosaic before changing model settings.

## Reuse Existing Detections

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir /path/to/clean_orthomosaics \
  --crowns-dir /path/to/crowns_multithreshold \
  --output-dir output/site_a_run \
  --steps 0,2,3,4
```

Use this when testing tracking without rerunning Detectree2.

## Detection Log

```csv
site,run_name,om_dir,output_dir,model_path,device,thresholds,status,notes
site_a,site_a_run,/path/to/clean_orthomosaics,output/site_a_run,/path/to/model.pth,cuda,0.15-0.65,checked,
```

## Handoff

Use the crown outputs for:

1. Tracking: `src/pipeline/README.md`.
2. Printouts: [04_orthomosaic_printout_crown_ids.md](04_orthomosaic_printout_crown_ids.md).
3. QField: [05_qfield_crown_annotation.md](05_qfield_crown_annotation.md).

For species annotation across dates, consensus crowns are usually better than raw single-date detections.

## Troubleshooting

1. Import failure: fix `dpm-detectree`, Detectree2, and Detectron2.
2. Missing model: set `DPM_MODEL_PATH` or `--model-path`.
3. One date fails: confirm the orthomosaic is a valid GeoTIFF.
4. Shifted crowns: check CRS and georeferencing.
5. Sparse detections: inspect lower threshold layers.
6. Noisy detections: inspect higher threshold layers and check shadows/non-tree objects.
7. GPKG read failure: confirm expected layers were written.
8. GPU failure: rerun on CPU to isolate environment issues.
