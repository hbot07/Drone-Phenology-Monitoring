#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Drone Phenology Monitoring Full Pipeline Orchestrator
# =============================================================================
#
# Runs the complete 4-step pipeline:
#   Step 0: Discover orthomosaics and write pipeline_config.json  (base env)
#   Step 1: Multi-threshold Detectree2 crown detection            (base env)
#   Step 2: Crown tracking + consensus crown generation           (detectree env)
#   Step 3: Phenology / leaf-shed analysis                        (detectree env)
#   Step 4: Interactive HTML viewer generation                    (detectree env)
#
# Usage:
#   bash run_pipeline.sh --om-dir /path/to/orthomosaics [OPTIONS]
#
# Required:
#   --om-dir PATH           Folder containing .tif orthomosaics
#
# Options:
#   --run-name NAME         Human-readable run name (default: auto-generated)
#   --output-dir PATH       Output directory (default: <project_root>/output/<run_name>)
#   --model-path PATH       Path to .pth model (default: auto-discover)
#   --exclude-stems STEMS   Comma-separated stems to exclude from tracking
#                           IMPORTANT for LHC: always pass --exclude-stems odm_orthophoto_9_12_25
#                           (that OM has gross misalignment and corrupts tracking)
#   --crowns-dir PATH       Use existing crowns directory instead of running step 1
#   --tile-width N          Detectree tile width in metres (default: 25)
#   --tile-height N         Detectree tile height in metres (default: 25)
#   --tile-buffer N         Detectree tile buffer in metres (default: 15)
#   --device cpu|cuda       Inference device (default: cpu)
#   --threads N             CPU thread count for detection (default: 6)
#   --skip-existing         Skip crown detection if GPKG already valid (default: on)
#   --no-skip-existing      Force re-run detection even if GPKG is valid
#   --align-method METHOD   Alignment method for step 2 (default: pcc_tiled)
#                           Options: pcc_tiled, pcc, ecc, crowns
#   --align-threshold-tag T Crown threshold used for alignment (default: conf_0p65)
#   --min-partial-len N     Min chain length to include as partial chain (default: 5)
#   --min-partial-ratio R   Min one-to-one ratio for partial chains (default: 0.9)
#   --skip-chain-viz        Skip chain strip visualizations (faster tracking)
#   --skip-consensus-viz    Skip consensus strip visualizations (faster tracking)
#   --underlay-om last|first|N  Which OM to use as HTML viewer underlay (default: last)
#   --steps STEPS           Comma-separated steps to run: 0,1,2,3,4 (default: all)
#   --base-env NAME         Conda environment for step 1 (default: base)
#   --tracking-env NAME     Conda environment for steps 2-4 (default: detectree)
#
# Examples:
#   # Full pipeline, LHC dataset (MUST exclude the bad Dec-9 OM):
#   bash run_pipeline.sh \
#       --om-dir /path/to/project/input/input_om_lhc \
#       --exclude-stems odm_orthophoto_9_12_25 \
#       --crowns-dir /path/to/project/output/detectree_om_lhc_multithreshold_smaller_tiles/crowns_multithreshold \
#       --steps 0,2,3,4
#
#   # Resume tracking step only (skip detection), using existing crowns:
#   bash run_pipeline.sh \
#       --om-dir /path/to/project/input/input_om_lhc \
#       --exclude-stems odm_orthophoto_9_12_25 \
#       --crowns-dir /path/to/crowns_multithreshold \
#       --steps 2,3,4 \
#       --output-dir /path/to/existing/output/dir
#
#   # SIT dataset:
#   bash run_pipeline.sh \
#       --om-dir /path/to/project/input/input_om_sit \
#       --crowns-dir /path/to/project/output/detectree_om_sit_multithreshold/crowns_multithreshold \
#       --steps 0,2,3,4
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
OM_DIR=""
RUN_NAME=""
OUTPUT_DIR=""
MODEL_PATH=""
EXCLUDE_STEMS=""
TILE_WIDTH=25
TILE_HEIGHT=25
TILE_BUFFER=15
DEVICE="cpu"
THREADS=6
SKIP_EXISTING_FLAG="--skip-existing"
SKIP_CHAIN_VIZ=""
SKIP_CONSENSUS_VIZ=""
UNDERLAY_OM="last"
STEPS="0,1,2,3,4"
BASE_ENV="base"
TRACKING_ENV="detectree"
CROWNS_DIR=""
ALIGN_METHOD="pcc_tiled"
ALIGN_THRESH_TAG="conf_0p65"
MIN_PARTIAL_LEN=""
MIN_PARTIAL_RATIO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --om-dir)           OM_DIR="$2";        shift 2 ;;
        --run-name)         RUN_NAME="$2";      shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";    shift 2 ;;
        --model-path)       MODEL_PATH="$2";    shift 2 ;;
        --exclude-stems)    EXCLUDE_STEMS="$2"; shift 2 ;;
        --tile-width)       TILE_WIDTH="$2";    shift 2 ;;
        --tile-height)      TILE_HEIGHT="$2";   shift 2 ;;
        --tile-buffer)      TILE_BUFFER="$2";   shift 2 ;;
        --device)           DEVICE="$2";        shift 2 ;;
        --threads)          THREADS="$2";       shift 2 ;;
        --skip-existing)    SKIP_EXISTING_FLAG="--skip-existing";   shift ;;
        --no-skip-existing) SKIP_EXISTING_FLAG="--no-skip-existing"; shift ;;
        --skip-chain-viz)   SKIP_CHAIN_VIZ="--skip-chain-viz";     shift ;;
        --skip-consensus-viz) SKIP_CONSENSUS_VIZ="--skip-consensus-viz"; shift ;;
        --underlay-om)      UNDERLAY_OM="$2";   shift 2 ;;
        --steps)            STEPS="$2";         shift 2 ;;
        --base-env)         BASE_ENV="$2";      shift 2 ;;
        --tracking-env)     TRACKING_ENV="$2";  shift 2 ;;
        --crowns-dir)       CROWNS_DIR="$2";    shift 2 ;;
        --align-method)     ALIGN_METHOD="$2";  shift 2 ;;
        --align-threshold-tag) ALIGN_THRESH_TAG="$2"; shift 2 ;;
        --min-partial-len)  MIN_PARTIAL_LEN="$2"; shift 2 ;;
        --min-partial-ratio) MIN_PARTIAL_RATIO="$2"; shift 2 ;;
        -h|--help)
            sed -n '/#\s*Usage:/,/^# =====/p' "$0" | head -60
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$OM_DIR" ]]; then
    echo "ERROR: --om-dir is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve script directory and set up paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Utility: run a python command in a specific conda environment
# ---------------------------------------------------------------------------
run_in_env() {
    local env_name="$1"; shift
    echo ""
    echo "=== Running in conda env '${env_name}': $* ==="
    echo ""
    conda run --no-capture-output -n "${env_name}" python "$@"
}

# ---------------------------------------------------------------------------
# Helper to check if a step should run
# ---------------------------------------------------------------------------
should_run_step() {
    local step="$1"
    # Check if step number appears in the comma-separated STEPS list
    echo "$STEPS" | tr ',' '\n' | grep -qx "$step"
}

# ---------------------------------------------------------------------------
# Step 0: Discover OMs and write pipeline_config.json
# ---------------------------------------------------------------------------
CONFIG_PATH=""

if should_run_step 0; then
    echo ""
    echo "========================================================"
    echo "  STEP 0: Discovering orthomosaics"
    echo "========================================================"

    DISCOVER_ARGS=(
        "${SCRIPT_DIR}/00_discover_oms.py"
        --om-dir "${OM_DIR}"
        --tile-width "${TILE_WIDTH}"
        --tile-height "${TILE_HEIGHT}"
        --tile-buffer "${TILE_BUFFER}"
    )
    [[ -n "$RUN_NAME" ]]   && DISCOVER_ARGS+=(--run-name "${RUN_NAME}")
    [[ -n "$OUTPUT_DIR" ]] && DISCOVER_ARGS+=(--output-dir "${OUTPUT_DIR}")
    [[ -n "$MODEL_PATH" ]] && DISCOVER_ARGS+=(--model-path "${MODEL_PATH}")
    [[ -n "$EXCLUDE_STEMS" ]] && DISCOVER_ARGS+=(--exclude-stems "${EXCLUDE_STEMS}")
    [[ -n "$CROWNS_DIR" ]]    && DISCOVER_ARGS+=(--crowns-dir "${CROWNS_DIR}")

    # Capture output to extract config path
    STEP0_OUT=$(conda run --no-capture-output -n "${BASE_ENV}" python "${DISCOVER_ARGS[@]}" 2>&1)
    echo "$STEP0_OUT"

    CONFIG_PATH=$(echo "$STEP0_OUT" | grep "^PIPELINE_CONFIG=" | tail -1 | cut -d= -f2-)
    if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
        echo "ERROR: Step 0 failed to write pipeline_config.json" >&2
        exit 1
    fi
    echo ""
    echo "Config: $CONFIG_PATH"
else
    # Config path must be derived from output dir
    if [[ -n "$OUTPUT_DIR" ]]; then
        CONFIG_PATH="${OUTPUT_DIR}/pipeline_config.json"
    else
        echo "ERROR: When skipping step 0, --output-dir must be set and contain pipeline_config.json." >&2
        exit 1
    fi
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "ERROR: pipeline_config.json not found at: $CONFIG_PATH" >&2
        exit 1
    fi
    echo "Using existing config: $CONFIG_PATH"
fi

# ---------------------------------------------------------------------------
# Step 1: Crown detection  (base conda env — has detectree2/detectron2)
# ---------------------------------------------------------------------------
if should_run_step 1; then
    echo ""
    echo "========================================================"
    echo "  STEP 1: Crown detection (conda: ${BASE_ENV})"
    echo "========================================================"

    DETECT_ARGS=(
        "${SCRIPT_DIR}/01_crown_detection.py"
        --config "${CONFIG_PATH}"
        --device "${DEVICE}"
        --threads "${THREADS}"
        ${SKIP_EXISTING_FLAG}
    )

    run_in_env "${BASE_ENV}" "${DETECT_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Step 2: Crown tracking + consensus crowns  (detectree conda env)
# ---------------------------------------------------------------------------
if should_run_step 2; then
    echo ""
    echo "========================================================"
    echo "  STEP 2: Crown tracking (conda: ${TRACKING_ENV})"
    echo "========================================================"

    TRACK_ARGS=(
        "${SCRIPT_DIR}/02_crown_tracking.py"
        --config "${CONFIG_PATH}"
        --align-method "${ALIGN_METHOD}"
        --align-threshold-tag "${ALIGN_THRESH_TAG}"
        ${SKIP_CHAIN_VIZ}
        ${SKIP_CONSENSUS_VIZ}
    )
    [[ -n "$MIN_PARTIAL_LEN" ]]   && TRACK_ARGS+=(--min-partial-len "${MIN_PARTIAL_LEN}")
    [[ -n "$MIN_PARTIAL_RATIO" ]] && TRACK_ARGS+=(--min-partial-ratio "${MIN_PARTIAL_RATIO}")

    run_in_env "${TRACKING_ENV}" "${TRACK_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Step 3: Phenology analysis  (detectree conda env)
# ---------------------------------------------------------------------------
if should_run_step 3; then
    echo ""
    echo "========================================================"
    echo "  STEP 3: Phenology analysis (conda: ${TRACKING_ENV})"
    echo "========================================================"

    PHENO_ARGS=(
        "${SCRIPT_DIR}/03_phenology_analysis.py"
        --config "${CONFIG_PATH}"
    )

    run_in_env "${TRACKING_ENV}" "${PHENO_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Step 4: Interactive viewer  (detectree conda env)
# ---------------------------------------------------------------------------
if should_run_step 4; then
    echo ""
    echo "========================================================"
    echo "  STEP 4: Interactive viewer (conda: ${TRACKING_ENV})"
    echo "========================================================"

    VIZ_ARGS=(
        "${SCRIPT_DIR}/04_interactive_viz.py"
        --config "${CONFIG_PATH}"
        --underlay-om "${UNDERLAY_OM}"
    )

    run_in_env "${TRACKING_ENV}" "${VIZ_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE"
echo "========================================================"
echo ""
echo "Config: ${CONFIG_PATH}"

if [[ -f "$CONFIG_PATH" ]]; then
    OUTPUT_DIR_FINAL=$(python3 -c "import json; c=json.load(open('${CONFIG_PATH}')); print(c.get('output_dir','?'))" 2>/dev/null || echo "?")
    VIEWER_HTML=$(python3 -c "import json; c=json.load(open('${CONFIG_PATH}')); print(c.get('viewer_html','?'))" 2>/dev/null || echo "?")
    CONSENSUS_GPKG=$(python3 -c "import json; c=json.load(open('${CONFIG_PATH}')); print(c.get('consensus_gpkg','?'))" 2>/dev/null || echo "?")
    SCORES_CSV=$(python3 -c "import json; c=json.load(open('${CONFIG_PATH}')); print(c.get('phenology_scores_csv','?'))" 2>/dev/null || echo "?")
    STEPS_DONE=$(python3 -c "import json; c=json.load(open('${CONFIG_PATH}')); print(', '.join(c.get('steps_completed',[])))" 2>/dev/null || echo "?")

    echo "Output dir:     ${OUTPUT_DIR_FINAL}"
    echo "Consensus GPKG: ${CONSENSUS_GPKG}"
    echo "Scores CSV:     ${SCORES_CSV}"
    echo "Viewer:         ${VIEWER_HTML}"
    echo "Steps done:     ${STEPS_DONE}"
fi

echo ""
