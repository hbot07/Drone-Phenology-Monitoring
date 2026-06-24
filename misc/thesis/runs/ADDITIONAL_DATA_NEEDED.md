# Additional Data Needed For Thesis Results

This note records data that is needed before turning the current draft/results into final thesis claims.

## Needed For Final Quantitative Tables

1. Full LHC analysis run if the thesis should report the full dataset mentioned in the outline.
   - Current local input available to this run: 9 LHC orthomosaics.
   - Current paper-support run uses `output/lhc_pipeline_fixed`: 8 LHC orthomosaics, excluding `odm_orthophoto_9_12_25`.
   - The thesis outline mentions a larger LHC set. If that larger set is the intended final dataset, those orthomosaics/crown outputs need to be present locally.

2. Full SIT analysis run if the paper should report both primary sites.
   - Existing outputs include `output/sit_pipeline_fixed`, but this first artifact run only processed LHC because the current request said to use `input/input_om_lhc` for now.

## Needed For Species-Specific Phenology Figures

1. A species-joined master crown layer for the final LHC/SIT run.
   - The current `output/lhc_pipeline_fixed/03_phenology/tree_master_geojson.geojson` contains phenology/classification fields, but `field_data` is empty for the inspected rows.
   - For figures such as a Semal/Bombax crown through time, we need crown IDs linked to species labels.

2. Confirmation of which species examples should be featured.
   - The outline suggests Semal for a leafy/bare/red-flowering trajectory.
   - If Semal is not present in the LHC subset, use SIT or another labelled site.

## Needed For Field-Workflow Figures

1. Confirm that the paper-form photograph currently in the Overleaf folder is the one to use.
   - The outline originally marked the paper-form photo as pending.
   - The Overleaf project now contains `figures/field_paper_form.png`, so this may already be resolved.

## Needed For Detection Evaluation

1. Decide whether to regenerate Detectree2 metrics against `input/ground_truth.json`.
   - The outline records older placeholder metrics from a 124-crown annotation set.
   - For final claims, run the detection evaluation script against the exact model/settings used for final crown detections.
