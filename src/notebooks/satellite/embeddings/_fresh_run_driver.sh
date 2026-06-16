#!/usr/bin/env bash
# Sweeps on the two legitimate small-crown feature sources (prof-endorsed):
#   - weighted mean over ORIGINAL crown geometry  (gee_mean)
#   - centroid pixel value                         (gee_centroid)
# 20 m buffer / fused sources are intentionally excluded (buffer too big for small crowns).
set -eo pipefail
cd "$(dirname "$0")"

PKG=../drone_phenology_rf_package
SUITE=$PKG/python/12_run_model_suite.py
PY=python3
COMMON="--trees 300 --max-holdouts 6"

DEF="--label label_acacia --label label_deciduous --label label_esd --label label_showy_flower --label label_yellow_strict --label label_yellow_broad --label label_red_showy"
CLEAN="--label label_acacia_visual --label label_acacia_species --label label_acacia_visual_or_species"
BIG="--label label_acacia_clustering --label label_acacia_visual_or_clustering --label label_acacia_species_or_clustering --label label_acacia_all_priority"

for MODE in mean centroid; do
  CSV=exports/gee_${MODE}_2024_acacia_label_configs.csv
  OUT=outputs/fresh_gee_${MODE}
  echo "## $MODE : default labels"
  $PY $SUITE --csv $CSV --outdir $OUT $COMMON $DEF
  echo "## $MODE : clean acacia configs"
  $PY $SUITE --csv $CSV --outdir $OUT $COMMON $CLEAN
  echo "## $MODE : big clustering acacia configs (random-only)"
  $PY $SUITE --csv $CSV --outdir $OUT $COMMON --random-only $BIG
done

echo "ALL_DONE"
