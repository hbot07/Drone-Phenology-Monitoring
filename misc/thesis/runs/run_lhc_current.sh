#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

export MPLCONFIGDIR="${SCRIPT_DIR}/lhc_current/_mplconfig"
export XDG_CACHE_HOME="${SCRIPT_DIR}/lhc_current/_cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

"$PYTHON_BIN" "${SCRIPT_DIR}/scripts/lhc_thesis_artifacts.py" \
  --project-root "${PROJECT_ROOT}" \
  --om-dir "${PROJECT_ROOT}/input/input_om_lhc" \
  --pipeline-output "${PROJECT_ROOT}/output/lhc_pipeline_fixed" \
  --crowns-dir "${PROJECT_ROOT}/output/detectree_om_lhc_multithreshold_smaller_tiles/crowns_multithreshold" \
  --run-dir "${SCRIPT_DIR}/lhc_current"
