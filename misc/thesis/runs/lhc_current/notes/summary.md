# LHC Thesis Artifact Run

This run extracts paper-support artifacts from the currently available LHC data and existing pipeline output.

## Sources

- Orthomosaics: `/Users/hbot07/VS Code/Drone-Phenology-Monitoring/input/input_om_lhc`
- Pipeline output: `/Users/hbot07/VS Code/Drone-Phenology-Monitoring/output/lhc_pipeline_fixed`
- Crown detections: `/Users/hbot07/VS Code/Drone-Phenology-Monitoring/output/detectree_om_lhc_multithreshold_smaller_tiles/crowns_multithreshold`

## Key Numbers

- Available LHC orthomosaics in input folder: **9**
- Pipeline usage status: `{'included': 8, 'excluded': 1}`
- Pipeline OMs used: **8**
- Excluded stems: `odm_orthophoto_9_12_25`
- Alignment method: `pcc_tiled`
- Base threshold: `conf_0p45`
- Cleaned consensus crowns: **87**
- Raw consensus chains before deduplication: **141**
- Overall match rate: **0.921**
- Crowns with phenology scores: **87.0**
- Deciduous crowns at threshold 0.70: **60.0**
- Deciduous fraction: **0.690**

## Generated Tables

- `tables/lhc_alignment_shifts.csv`
- `tables/lhc_chain_length_distribution.csv`
- `tables/lhc_detection_counts_by_threshold.csv`
- `tables/lhc_match_rates.csv`
- `tables/lhc_orthomosaic_inventory.csv`
- `tables/lhc_per_om_feature_summary.csv`
- `tables/lhc_phenology_event_counts.csv`
- `tables/lhc_phenology_summary.csv`
- `tables/lhc_tracking_summary.csv`

## Generated Figures

- `figures/lhc_alignment_shifts.png`
- `figures/lhc_chain_length_distribution.png`
- `figures/lhc_deciduous_score_hist.png`
- `figures/lhc_detection_counts_by_date_threshold.png`
- `figures/lhc_example_deciduous_crown_trajectory.png`
- `figures/lhc_match_rates.png`
- `figures/lhc_orthomosaic_contact_sheet.png`
- `figures/lhc_veg_fraction_timeseries.png`

## Notes For Thesis Use

- Treat this as an initial LHC subset run: it uses `output/lhc_pipeline_fixed`, which excludes the bad 2025-12-09 orthomosaic.
- The thesis outline says final results should eventually be regenerated on the full LHC/SIT datasets; do not silently present these subset numbers as final full-dataset numbers.
- The generated crown trajectory is chosen automatically by high deciduousness score, so inspect it before using it as an illustrative biological example.
