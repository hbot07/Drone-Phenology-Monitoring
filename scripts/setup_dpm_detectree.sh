#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dpm-detectree}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/envs/detectree-windows.yml"
REQ_FILE="${PROJECT_ROOT}/requirements/dpm-detectree.txt"

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

cat <<'EOF'

dpm-detectree is ready for the base geospatial/PyTorch packages.
Install Detectree2 and Detectron2 next using versions that match this machine's
CUDA/PyTorch setup. After that, test:

  conda run -n dpm-detectree python -c "import detectree2, detectron2; print('ok')"
EOF
