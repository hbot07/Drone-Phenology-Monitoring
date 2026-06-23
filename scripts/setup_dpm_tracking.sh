#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dpm-tracking}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/envs/tracking-windows.yml"
REQ_FILE="${PROJECT_ROOT}/requirements/dpm-tracking.txt"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH. Install Miniconda/Anaconda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda run -n "$ENV_NAME" python -m pip install -r "$REQ_FILE"
conda run -n "$ENV_NAME" python -c "import geopandas, rasterio, networkx, cv2, sklearn; print('dpm-tracking ok')"
