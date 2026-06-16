# Archived pre-exp01 label configs

These are the older v1/baseline label definitions, superseded by
`configs/exp01_species_review/` (the canonical, manually-reviewed configs).

Archived 2026-06-17 to avoid confusion. Key difference: the v1 yellow_broad
wrongly included Prosopis Juliflora and Kasod as yellow positives; exp01
excludes Prosopis (small greenish flowers, structural signal) and moves Kasod
to negative.

If regenerating labels, point the scripts at the exp01 folder:
  python python/01b_relabel.py --config-dir configs/exp01_species_review
