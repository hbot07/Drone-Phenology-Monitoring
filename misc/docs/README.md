# Drone Phenology Monitoring Guides

These guides describe the reusable workflow for turning repeated drone flights into crown-level phenology data. They are written for a new site or a new machine, so the main instructions avoid hardcoded local paths.

Recommended reading order:

1. [Collecting drone imagery](01_collecting_drone_imagery.md)
2. [Building orthomosaics with ODM/WebODM/NodeODM](02_building_orthomosaic_webodm.md)
3. [Running Detectree2 crown detection](03_running_detectree2.md)
4. [Creating orthomosaic printouts with crown IDs](04_orthomosaic_printout_crown_ids.md)
5. [Preparing QField crown annotation projects](05_qfield_crown_annotation.md)
6. [Appendix: local project scripts and datasets](06_appendix_local_project_scripts.md)

The reusable analysis pipeline is documented separately in [../../src/pipeline/README.md](../../src/pipeline/README.md).

The ODM scripts are in [../ODM](../ODM/). Use [../ODM/ODM_QUICKSTART.md](../ODM/ODM_QUICKSTART.md) for the short version and [../ODM/ODM_OM_RUNBOOK.md](../ODM/ODM_OM_RUNBOOK.md) for the current machine-specific runbook.
