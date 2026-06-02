# Running Detectree2

The crown-detection workflow is under:

- `Drone-Phenology-Monitoring\\src\\pipeline`

The main script is:

- `Drone-Phenology-Monitoring\\src\\pipeline\\01_crown_detection.py`

The workflow is:

1. Run `00_discover_oms.py` to create `pipeline_config.json`.
2. Run `01_crown_detection.py` with that config.
3. Check the output GPKG files in `crowns_multithreshold`.
4. Use those outputs later for tracking.

This step is run after the orthomosaics are ready. The input is a folder of cleaned `.tif` orthomosaics. The output is a set of multi-threshold crown polygon files, one per orthomosaic.

Example:

```bash
python 00_discover_oms.py \
  --om-dir input/input_om_sit \
  --output-dir output/my_sit_run \
  --run-name my_sit_run

python 01_crown_detection.py \
  --config output/my_sit_run/pipeline_config.json \
  --device cpu \
  --threads 6
```

If you want to run through the shell wrapper instead of calling the scripts directly, the pipeline wrapper is:

```bash
bash run_pipeline.sh \
  --om-dir input/input_om_sit \
  --steps 0,1
```

The first command scans the orthomosaic folder, sorts the files by date, and writes the pipeline configuration. The second command runs Detectree2 on each orthomosaic and writes one multi-threshold crown file per orthomosaic.

The current script defaults are:

- tile width: `25`
- tile height: `25`
- tile buffer: `15`
- thresholds: `0.15` to `0.65` in steps of `0.05`
- fixed IoU: `0.7`
- simplify tolerance: `0.3`

The confidence thresholds become layer names in the output GPKG, for example:

1. `conf_0p15`
2. `conf_0p20`
3. `conf_0p25`
4. `conf_0p30`
5. `conf_0p35`
6. `conf_0p40`
7. `conf_0p45`
8. `conf_0p50`
9. `conf_0p55`
10. `conf_0p60`
11. `conf_0p65`

That means the output does not contain just one crown layer. It contains multiple layers such as `conf_0p15`, `conf_0p45`, and `conf_0p65`, which can then be used later during crown tracking.

The outputs go under:

```text
<output_dir>/01_detectree/crowns_multithreshold/
```

Each orthomosaic gets a multi-threshold GPKG such as:

```text
{stem}_multithreshold.gpkg
```

Along with the GPKG files, the detection step also writes metadata JSON files and a run summary.

The detection stage works like this:

1. The orthomosaic is tiled.
2. Detectree2 predicts crowns on each tile.
3. The tile predictions are projected back into map coordinates.
4. The predictions are stitched together.
5. The cleaned crowns are written out at multiple confidence thresholds.

Before moving on, it is worth opening a few outputs and checking that:

1. The crowns line up with the orthomosaic.
2. The number of crowns is plausible.
3. The expected threshold layers are present.
4. There is no obvious corruption in one date.

If crowns already exist and you do not want to rerun detection, the pipeline also supports reusing an existing `crowns_multithreshold` directory through step 0.

It is also worth remembering that later tracking does not need to use the same threshold layer for every dataset. The point of exporting multiple layers is that you can later choose a denser or cleaner crown set depending on the site.

Current parameter listing to fill later.

## Urban areas

| Parameter | Value |
|---|---|
| Detectree2 model path | TODO |
| Device | TODO |
| Threads | TODO |
| Tile width | TODO |
| Tile height | TODO |
| Tile buffer | TODO |
| Confidence thresholds | TODO |
| Fixed IoU | TODO |
| Simplify tolerance | TODO |

When these values are finalized, this section can be updated with the exact model path, thresholds, device, and any site-specific overrides actually used in those runs.

## Sanjay Van

| Parameter | Value |
|---|---|
| Detectree2 model path | TODO |
| Device | TODO |
| Threads | TODO |
| Tile width | TODO |
| Tile height | TODO |
| Tile buffer | TODO |
| Confidence thresholds | TODO |
| Fixed IoU | TODO |
| Simplify tolerance | TODO |

Troubleshooting:

1. If the detection step fails immediately, first check that `pipeline_config.json` exists and points to the right orthomosaic folder and model path.
2. If the script says the model cannot be found, check the `.pth` path in the config or place the model in the expected Detectree model folder.
3. If outputs are missing for one date only, open that orthomosaic and check that the file is valid and not corrupted.
4. If crowns are clearly shifted or nonsense, inspect the orthomosaic itself before blaming Detectree2. A bad orthomosaic will produce bad crowns.
5. If the output GPKG exists but later steps cannot use it, check whether the expected confidence layers are present.
6. If you already have good crown files, reuse them instead of rerunning the detection step unnecessarily.
