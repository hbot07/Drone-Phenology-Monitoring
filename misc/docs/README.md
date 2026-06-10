# Drone Phenology Monitoring Guides

These guides describe the operational workflow for turning repeated drone flights into crown-level phenology data.

Recommended reading order:

1. `01_collecting_drone_imagery.md` - field collection habits that keep repeated flights comparable.
2. `02_building_orthomosaic_webodm.md` - creating orthomosaics with the local ODM and NodeODM scripts.
3. `03_running_detectree2.md` - running crown detection on cleaned orthomosaic inputs.
4. `04_orthomosaic_printout_crown_ids.md` - producing numbered crown overlays for field use.
5. `05_qfield_crown_annotation.md` - preparing QGIS/QField projects for crown annotation.

The ODM scripts and runbooks used by the orthomosaic workflow are stored in `misc/ODM/`. Start with `misc/ODM/ODM_QUICKSTART.md` for the short version and `misc/ODM/ODM_OM_RUNBOOK.md` for the detailed machine-specific setup.
