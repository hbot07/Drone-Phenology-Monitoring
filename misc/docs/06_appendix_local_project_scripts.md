# Appendix: Local Project Scripts And Datasets

This appendix lists scripts and files that are useful for the current IITD/Sanjay Van project history. They are not the starting point for a new site. For new work, use the generic guides first.

## Generic Scripts To Prefer

- `misc/ODM/make_om.ps1`: one orthomosaic from one image folder.
- `misc/ODM/run_odm_batch.ps1`: many orthomosaics from a CSV file.
- `src/pipeline/run_pipeline.sh`: full analysis pipeline with `.env` support.
- `src/utility/numbered_crown_overlay.py`: printable numbered crown overlays.

## Site-Specific ODM Helpers

These scripts encode local folder layouts, current backlog dates, and local storage conventions:

- `misc/ODM/lhc_sit_make_oms.ps1`
- `misc/ODM/sv_make_oms.ps1`
- `misc/ODM/run_batch_parallel.ps1`
- `misc/ODM/lhc_sit_upload_oms.sh`
- `misc/ODM/sv_upload_oms.sh`
- `misc/ODM/sv_sync_input_oms.sh`

Use them as examples or for the current machine only. If the same work is needed on another machine, create a CSV for `run_odm_batch.ps1` instead of editing hardcoded paths inside these files.

## Local Status Tables

- `misc/ODM/drone_data.csv`: LHC/SIT raw-data and orthomosaic status.
- `misc/ODM/sanjay_van_data.csv`: Sanjay Van spot/date status and filer notes.

These are project records, not general input formats.

## Machine-Specific Runbook

- `misc/ODM/ODM_OM_RUNBOOK.md`

This documents the currently working WSL/Docker/GPU setup on one machine. It is useful when maintaining that machine, but a new user should not copy its paths blindly.

## Historical Notebooks

Exploratory notebooks live under:

- `src/notebooks/`
- `src/notebook_archive/`

They are valuable for research context and method history, but the reusable entry points are the scripts documented in the main guides.
