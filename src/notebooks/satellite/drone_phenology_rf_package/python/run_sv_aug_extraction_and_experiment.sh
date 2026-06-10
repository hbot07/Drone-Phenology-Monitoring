#!/usr/bin/env bash
# Extract S2 features for all SV crowns (clustering-labeled) for years 2022-2025,
# build multiyear table, then run the baseline vs augmented experiment.
#
# Usage:
#   cd src/notebooks/satellite/drone_phenology_rf_package
#   bash python/run_sv_aug_extraction_and_experiment.sh
#   bash python/run_sv_aug_extraction_and_experiment.sh --skip-extraction  # if CSVs exist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PKG_DIR"

CROWNS="data/sv_crowns_for_extraction.geojson"
BUFFER=10
ITEMS=4
CLOUD=70
SKIP_EXTRACTION=false

# Parse args
for arg in "$@"; do
  case $arg in
    --skip-extraction) SKIP_EXTRACTION=true ;;
  esac
done

echo "=== SV Clustering Augmentation Pipeline ==="
echo "Working dir: $PKG_DIR"
echo ""

# ── Step 1: Extract per-year S2 features ────────────────────────────────────
if [ "$SKIP_EXTRACTION" = false ]; then
  for YEAR in 2022 2023 2024 2025; do
    OUT="exports/sv_aug_s2_${YEAR}_buffer${BUFFER}_items${ITEMS}.csv"
    if [ -f "$OUT" ]; then
      echo "[SKIP] $OUT already exists"
    else
      echo "[extract] year=$YEAR → $OUT"
      python python/03_extract_sentinel2_stac_features.py \
        --crowns "$CROWNS" \
        --year "$YEAR" \
        --geometry-mode buffer \
        --buffer-meters "$BUFFER" \
        --max-cloud "$CLOUD" \
        --max-items-per-season "$ITEMS" \
        --label-filter label_acacia \
        --out-csv "$OUT" 2>&1 | tee "/tmp/sv_extract_${YEAR}.log"
      echo "[extract] year=$YEAR done"
    fi
  done
else
  echo "[skip-extraction] Using existing per-year CSVs"
fi

# ── Step 2: Build multiyear table ────────────────────────────────────────────
AUG_CSV="exports/sv_aug_s2_2022_2025_combined_label_acacia.csv"
echo ""
echo "[multiyear] Combining years 2022-2025 → $AUG_CSV"
python python/10_build_multiyear_s2_table.py \
  --year-csv 2022=exports/sv_aug_s2_2022_buffer${BUFFER}_items${ITEMS}.csv \
  --year-csv 2023=exports/sv_aug_s2_2023_buffer${BUFFER}_items${ITEMS}.csv \
  --year-csv 2024=exports/sv_aug_s2_2024_buffer${BUFFER}_items${ITEMS}.csv \
  --year-csv 2025=exports/sv_aug_s2_2025_buffer${BUFFER}_items${ITEMS}.csv \
  --out-csv "$AUG_CSV"

echo "[multiyear] Done. Rows: $(python -c "import pandas as pd; df=pd.read_csv('$AUG_CSV'); print(len(df))")"

# ── Step 3: Run baseline-only experiment (no augmented CSV) ──────────────────
echo ""
echo "[experiment] Running baseline (GT-only) ..."
python python/16_acacia_clustering_experiment.py \
  --baseline-csv exports/stac_s2_features_2022_2025_buffer10_items4_label_acacia.csv \
  --outdir outputs/acacia_clustering_experiment

# ── Step 4: Run augmented experiment ─────────────────────────────────────────
echo ""
echo "[experiment] Running augmented (GT + clustering) ..."
python python/16_acacia_clustering_experiment.py \
  --baseline-csv exports/stac_s2_features_2022_2025_buffer10_items4_label_acacia.csv \
  --augmented-csv "$AUG_CSV" \
  --outdir outputs/acacia_clustering_experiment

echo ""
echo "=== Pipeline complete ==="
echo "Results: outputs/acacia_clustering_experiment/"
