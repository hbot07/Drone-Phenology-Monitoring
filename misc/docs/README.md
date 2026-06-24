# Drone Mapping Workflow Guides

Use these guides in order:

1. [Collect drone imagery](01_collecting_drone_imagery.md)
2. [Build orthomosaics](02_building_orthomosaic_webodm.md)
3. [Run Detectree2](03_running_detectree2.md)
4. [Make crown-ID printouts](04_orthomosaic_printout_crown_ids.md)
5. [Prepare QField annotation](05_qfield_crown_annotation.md)
6. [Appendix: local scripts and datasets](06_appendix_local_project_scripts.md)

The tracking and phenology pipeline is documented in [../../src/pipeline/README.md](../../src/pipeline/README.md).

## Workflow

```text
saved flight path
  -> repeated drone flights
  -> raw image folders
  -> orthomosaic GeoTIFFs
  -> checked clean orthomosaics
  -> Detectree2 crown polygons
  -> printouts or QField annotation
  -> tracking, consensus crowns, crops, phenology metrics
```

## Main Scripts

| Script | Use |
|---|---|
| `misc/ODM/make_om.ps1` | Build one orthomosaic from one image folder. |
| `misc/ODM/run_odm_batch.ps1` | Build many orthomosaics from a CSV table. |
| `src/pipeline/run_pipeline.sh` | Run discovery, detection, tracking, phenology, and viewer steps. |
| `src/utility/numbered_crown_overlay.py` | Create printable numbered crown overlays. |

Configure scripts with command-line flags, `.env` files, and CSV tables. Do not hardcode machine paths into reusable scripts.

## ODM References

- [../ODM/ODM_QUICKSTART.md](../ODM/ODM_QUICKSTART.md): short ODM command examples.
- [02_building_orthomosaic_webodm.md](02_building_orthomosaic_webodm.md): full orthomosaic workflow.
- [../ODM/ODM_OM_RUNBOOK.md](../ODM/ODM_OM_RUNBOOK.md): local WSL/Docker/GPU setup notes.

## Reusable Workflow Rules

1. Pass paths as arguments, `.env` values, or CSV rows.
2. Treat site names and dates as data.
3. Keep raw images, processing outputs, and clean analysis inputs separate.
4. Record model versions, thresholds, and checked outputs.
5. Keep machine-specific notes out of the main guides.
