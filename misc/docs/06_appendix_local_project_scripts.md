# Appendix: Local Scripts And Datasets

Use this page for local repository history, machine-specific notes, and older helper scripts. Start new work from the main guides.

## Prefer These Scripts

- `misc/ODM/make_om.ps1`: one orthomosaic from one image folder.
- `misc/ODM/run_odm_batch.ps1`: many orthomosaics from a CSV file.
- `src/pipeline/run_pipeline.sh`: full analysis pipeline with `.env` support.
- `src/utility/numbered_crown_overlay.py`: printable numbered crown overlays.

These scripts take paths and settings as arguments or config values.

## Local ODM Helpers

These encode local folder layouts, backlog dates, or storage conventions:

- `misc/ODM/lhc_sit_make_oms.ps1`
- `misc/ODM/sv_make_oms.ps1`
- `misc/ODM/run_batch_parallel.ps1`
- `misc/ODM/lhc_sit_upload_oms.sh`
- `misc/ODM/sv_upload_oms.sh`
- `misc/ODM/sv_sync_input_oms.sh`

Use them for maintenance or as examples. For a new machine, make a CSV for `run_odm_batch.ps1` instead of editing hardcoded paths.

## Local Status Tables

- `misc/ODM/drone_data.csv`: LHC/SIT raw-data and orthomosaic status.
- `misc/ODM/sanjay_van_data.csv`: Sanjay Van spot/date status and filer notes.

These are local records, not general input formats.

## Machine-Specific Runbook

- `misc/ODM/ODM_OM_RUNBOOK.md`

Use this when maintaining the existing WSL/Docker/GPU setup. Do not copy its paths blindly into new instructions.

## Historical Notebooks

- `src/notebooks/`
- `src/notebook_archive/`

Use notebooks for research context and method history. Use scripts for repeatable execution.

## Make A Local Script Reusable

A script is ready for the main guides when:

1. Paths are arguments, CSV rows, or `.env` values.
2. Dataset names are not hardcoded.
3. Inputs and outputs are documented.
4. A small example config exists.
5. Cleanup actions are opt-in.
6. It can run on a new machine after dependencies are installed.

## Keep Out Of Main Guides

1. Personal machine paths.
2. One-time backlog commands.
3. Drive-letter assumptions.
4. Deleting commands without clear safety notes.
5. Dataset-specific exclusions unless clearly marked.
6. Old experiment results that are not part of the workflow.
