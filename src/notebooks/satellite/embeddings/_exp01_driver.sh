#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
SUITE=../drone_phenology_rf_package/python/12_run_model_suite.py
PY=python3
DEF="--label label_acacia --label label_deciduous --label label_esd --label label_showy_flower --label label_yellow_strict --label label_yellow_broad --label label_red_showy"
for T in exp01_gee_centroid exp01_s2temporal exp01_all3; do
  echo "## $T"
  $PY $SUITE --csv exports/$T.csv --outdir outputs/$T --trees 300 --max-holdouts 4 $DEF
done
echo "ALL_DONE"
