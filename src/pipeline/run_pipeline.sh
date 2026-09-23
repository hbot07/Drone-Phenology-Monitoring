#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh  —  Drone Phenology Monitoring Full Pipeline Orchestrator
# =============================================================================
#
# Direct bash translation of the working run_pipeline.ps1.
# Runs all 6 steps in sequence and emits STEP: markers for the frontend.
#
# Usage (from server.py):
#   bash run_pipeline.sh \
#       --om-dir "..." --output-dir "..." --run-name "..." [OPTIONS]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script/project directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults (mirror PS1 param block exactly)
# ---------------------------------------------------------------------------
OM_DIR=""
RUN_NAME=""
OUTPUT_DIR=""
MODEL_PATH=""
EXCLUDE_STEMS=""
CROWNS_DIR=""
TILE_WIDTH=25
TILE_HEIGHT=25
TILE_BUFFER=15
SKIP_EXISTING="--skip-existing"
ALIGN_METHOD="pcc_tiled"
UNDERLAY_OM="first"
COG_TILE_SIZE=256
BASE_THRESH_TAG="conf_0p45"
ALIGN_THRESH_TAG="conf_0p65"
MIN_PARTIAL_LEN=""
MIN_PARTIAL_RATIO=""
SKIP_CHAIN_VIZ=""
SKIP_CONSENSUS_VIZ=""
STEPS="0,1,2,3,4a,4b"
BASE_ENV="dpm-detectree"
TRACKING_ENV="dpm-tracking"
PROJECT_ROOT_ARG=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --om-dir)              OM_DIR="$2";             shift 2 ;;
        --run-name)            RUN_NAME="$2";           shift 2 ;;
        --output-dir)          OUTPUT_DIR="$2";         shift 2 ;;
        --model-path)          MODEL_PATH="$2";         shift 2 ;;
        --exclude-stems)       EXCLUDE_STEMS="$2";      shift 2 ;;
        --crowns-dir)          CROWNS_DIR="$2";         shift 2 ;;
        --tile-width)          TILE_WIDTH="$2";         shift 2 ;;
        --tile-height)         TILE_HEIGHT="$2";        shift 2 ;;
        --tile-buffer)         TILE_BUFFER="$2";        shift 2 ;;
        --skip-existing)       SKIP_EXISTING="--skip-existing";    shift ;;
        --no-skip-existing)    SKIP_EXISTING="--no-skip-existing"; shift ;;
        --align-method)        ALIGN_METHOD="$2";       shift 2 ;;
        --underlay-om)         UNDERLAY_OM="$2";        shift 2 ;;
        --cog-tile-size)       COG_TILE_SIZE="$2";      shift 2 ;;
        --base-threshold-tag)  BASE_THRESH_TAG="$2";   shift 2 ;;
        --align-threshold-tag) ALIGN_THRESH_TAG="$2";  shift 2 ;;
        --min-partial-len)     MIN_PARTIAL_LEN="$2";   shift 2 ;;
        --min-partial-ratio)   MIN_PARTIAL_RATIO="$2"; shift 2 ;;
        --skip-chain-viz)      SKIP_CHAIN_VIZ="--skip-chain-viz";         shift ;;
        --skip-consensus-viz)  SKIP_CONSENSUS_VIZ="--skip-consensus-viz"; shift ;;
        --steps)               STEPS="$2";              shift 2 ;;
        --base-env)            BASE_ENV="$2";           shift 2 ;;
        --tracking-env)        TRACKING_ENV="$2";       shift 2 ;;
        --project-root)        PROJECT_ROOT_ARG="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OM_DIR" ]]; then
    echo "ERROR: --om-dir is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve paths (mirror PS1 logic)
# ---------------------------------------------------------------------------
if [[ -n "$PROJECT_ROOT_ARG" ]]; then
    PROJECT_ROOT="$PROJECT_ROOT_ARG"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/output/${RUN_NAME}"
fi

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="${PROJECT_ROOT}/input/detectree_models/250312_flexi.pth"
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
announce_step() {
    local step_key="$1"
    local label="$2"
    echo ""
    echo "========================================================"
    echo "  ${label}"
    echo "========================================================"
    echo ""
    # Machine-readable marker — server.py watches for "STEP:<key>"
    echo "STEP:${step_key}"
}

should_run() {
    local step="$1"
    echo "$STEPS" | tr ',' '\n' | grep -qx "$step"
}

run_in_env() {
    local env_name="$1"; shift
    echo ""
    echo "=== Running in conda env '${env_name}': python $* ==="
    echo ""
    # Prepend the conda env bin/ to PATH so subprocess calls like gdal2tiles
    # are found. conda run alone does NOT fully activate the env — it runs
    # Python with the right interpreter but leaves PATH unchanged, so tools
    # installed in the env are invisible to child processes.
    local env_prefix
    env_prefix=$(conda run -n "${env_name}" python -c "import sys; print(sys.prefix)" 2>/dev/null)
    if [[ -n "$env_prefix" ]]; then
        export PATH="${env_prefix}/bin:${PATH}"
        export GDAL_DATA="${env_prefix}/share/gdal"
        export PROJ_LIB="${env_prefix}/share/proj"
    fi
    conda run --no-capture-output -n "${env_name}" python "$@"
}

# ---------------------------------------------------------------------------
# Step 0: Discover OMs
# ---------------------------------------------------------------------------
CONFIG_PATH=""

if should_run 0; then
    announce_step "00_discover_oms" "STEP 0: Discovering orthomosaics"

    DISCOVER_SCRIPT="${SCRIPT_DIR}/00_discover_oms.py"
    if [[ ! -f "$DISCOVER_SCRIPT" ]]; then
        echo "ERROR: 00_discover_oms.py not found at $DISCOVER_SCRIPT" >&2
        exit 1
    fi

    DISCOVER_ARGS=(
        "$DISCOVER_SCRIPT"
        "--om-dir"      "$OM_DIR"
        "--tile-width"  "$TILE_WIDTH"
        "--tile-height" "$TILE_HEIGHT"
        "--tile-buffer" "$TILE_BUFFER"
    )
    [[ -n "$RUN_NAME" ]]         && DISCOVER_ARGS+=("--run-name"      "$RUN_NAME")
    [[ -n "$OUTPUT_DIR" ]]       && DISCOVER_ARGS+=("--output-dir"    "$OUTPUT_DIR")
    [[ -n "$EXCLUDE_STEMS" ]]    && DISCOVER_ARGS+=("--exclude-stems" "$EXCLUDE_STEMS")
    [[ -n "$CROWNS_DIR" ]]       && DISCOVER_ARGS+=("--crowns-dir"    "$CROWNS_DIR")
    [[ -n "$PROJECT_ROOT_ARG" ]] && DISCOVER_ARGS+=("--project-root"  "$PROJECT_ROOT_ARG")
    [[ -n "$MODEL_PATH" ]]       && DISCOVER_ARGS+=("--model-path"    "$MODEL_PATH")

    # Capture output to extract config path and stream it live
    STEP0_OUT=$(python "${DISCOVER_ARGS[@]}" 2>&1)
    echo "$STEP0_OUT"

    CONFIG_PATH=$(echo "$STEP0_OUT" | grep "^PIPELINE_CONFIG=" | tail -1 | cut -d= -f2-)

    if [[ -z "$CONFIG_PATH" || ! -f "$CONFIG_PATH" ]]; then
        echo "ERROR: Step 0 failed to write pipeline_config.json" >&2
        exit 1
    fi

    echo ""
    echo "Config: $CONFIG_PATH"
    echo ""
    echo "========================================================"
    echo "  STEP 0 COMPLETE"
    echo "========================================================"
else
    if [[ -z "$OUTPUT_DIR" ]]; then
        echo "ERROR: When skipping step 0, --output-dir must be set and contain pipeline_config.json." >&2
        exit 1
    fi
    CONFIG_PATH="${OUTPUT_DIR}/pipeline_config.json"
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "ERROR: pipeline_config.json not found at $CONFIG_PATH" >&2
        exit 1
    fi
    echo "Using existing config: $CONFIG_PATH"
fi

# ---------------------------------------------------------------------------
# Step 1: Crown detection  (dpm-detectree conda env)
# ---------------------------------------------------------------------------
if should_run 1; then
    announce_step "01_crown_detection" "STEP 1: Crown detection (conda: ${BASE_ENV})"

    DETECT_ARGS=(
        "${SCRIPT_DIR}/01_crown_detection.py"
        "--config" "$CONFIG_PATH"
        "--device" "cuda"
        "$SKIP_EXISTING"
    )
    # Note: model_path comes from pipeline_config.json (written by Step 0), not CLI.

    run_in_env "$BASE_ENV" "${DETECT_ARGS[@]}"

    echo ""
    echo "========================================================"
    echo "  STEP 1 COMPLETE"
    echo "========================================================"
fi

# ---------------------------------------------------------------------------
# Step 2: Crown tracking + consensus crowns  (dpm-tracking conda env)
# ---------------------------------------------------------------------------
if should_run 2; then
    announce_step "02_crown_tracking" "STEP 2: Crown tracking (conda: ${TRACKING_ENV})"

    TRACK_ARGS=(
        "${SCRIPT_DIR}/02_crown_tracking.py"
        "--config"              "$CONFIG_PATH"
        "--base-threshold-tag"  "$BASE_THRESH_TAG"
        "--align-method"        "$ALIGN_METHOD"
        "--align-threshold-tag" "$ALIGN_THRESH_TAG"
    )
    [[ -n "$SKIP_CHAIN_VIZ" ]]     && TRACK_ARGS+=("$SKIP_CHAIN_VIZ")
    [[ -n "$SKIP_CONSENSUS_VIZ" ]] && TRACK_ARGS+=("$SKIP_CONSENSUS_VIZ")
    [[ -n "$MIN_PARTIAL_LEN" ]]    && TRACK_ARGS+=("--min-partial-len"   "$MIN_PARTIAL_LEN")
    [[ -n "$MIN_PARTIAL_RATIO" ]]  && TRACK_ARGS+=("--min-partial-ratio" "$MIN_PARTIAL_RATIO")

    run_in_env "$TRACKING_ENV" "${TRACK_ARGS[@]}"

    echo ""
    echo "========================================================"
    echo "  STEP 2 COMPLETE"
    echo "========================================================"
fi

# ---------------------------------------------------------------------------
# Step 3: Phenology analysis  (dpm-tracking conda env)
# ---------------------------------------------------------------------------
if should_run 3; then
    announce_step "03_phenology_analysis" "STEP 3: Phenology analysis (conda: ${TRACKING_ENV})"

    PHENO_ARGS=(
        "${SCRIPT_DIR}/03_phenology_analysis.py"
        "--config" "$CONFIG_PATH"
    )

    run_in_env "$TRACKING_ENV" "${PHENO_ARGS[@]}"

    echo ""
    echo "========================================================"
    echo "  STEP 3 COMPLETE"
    echo "========================================================"
fi

# ---------------------------------------------------------------------------
# Step 3b: Phenophase classification  (dpm-tracking conda env)
# ---------------------------------------------------------------------------
if should_run 3; then
    announce_step "03b_phenophase_classification" "STEP 3b: Phenophase classification (conda: ${TRACKING_ENV})"

    # Script must exist in the pipeline directory
    PHENOCLF_SCRIPT="${SCRIPT_DIR}/12_apply_phenophase_to_geojson.py"
    if [[ ! -f "$PHENOCLF_SCRIPT" ]]; then
        echo "WARNING: 12_apply_phenophase_to_geojson.py not found at ${PHENOCLF_SCRIPT} — skipping Step 3b" >&2
    else
        # Model: use --model-path CLI arg if provided, otherwise fall back to
        # the fixed location in the project tree
        PHENOCLF_MODEL=""
        if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH" && "$MODEL_PATH" == *phenophase* ]]; then
            PHENOCLF_MODEL="$MODEL_PATH"
        else
            # Fixed location where the model is stored
            FIXED_MODEL="${PROJECT_ROOT}/model_for_phenology/phenophase_gb.joblib"
            if [[ -f "$FIXED_MODEL" ]]; then
                PHENOCLF_MODEL="$FIXED_MODEL"
            fi
        fi

        if [[ -z "$PHENOCLF_MODEL" ]]; then
            echo "WARNING: phenophase_gb.joblib not found — skipping Step 3b" >&2
            echo "  Expected at: ${PROJECT_ROOT}/model_for_phenology/phenophase_gb.joblib" >&2
        else
            echo "  Using model: ${PHENOCLF_MODEL}"
            PHENOCLF_ARGS=(
                "$PHENOCLF_SCRIPT"
                "--config"     "$CONFIG_PATH"
                "--model"      "$PHENOCLF_MODEL"
                "--min-run"    "2"
                "--flip-below" "0.90"
            )

            run_in_env "$TRACKING_ENV" "${PHENOCLF_ARGS[@]}"
        fi
    fi

    echo ""
    echo "========================================================"
    echo "  STEP 3b COMPLETE"
    echo "========================================================"
fi

# ---------------------------------------------------------------------------
# Step 4a: COG tiling  (dpm-tracking conda env)
# ---------------------------------------------------------------------------
if should_run 4a; then
    announce_step "04a_cog_tiling" "STEP 4a: COG tiling (conda: ${TRACKING_ENV})"

    COG_ARGS=(
        "${SCRIPT_DIR}/04a_cog_tiling.py"
        "--config"      "$CONFIG_PATH"
        "--underlay-om" "$UNDERLAY_OM"
        "--tile-size"   "$COG_TILE_SIZE"
    )

    run_in_env "$TRACKING_ENV" "${COG_ARGS[@]}"

    echo ""
    echo "========================================================"
    echo "  STEP 4a COMPLETE"
    echo "========================================================"
fi

# ---------------------------------------------------------------------------
# Step 4b: Interactive viewer  (dpm-tracking conda env)
# ---------------------------------------------------------------------------
if should_run 4b; then
    announce_step "04b_interactive_viz" "STEP 4b: Interactive viewer (conda: ${TRACKING_ENV})"

    VIZ_ARGS=(
        "${SCRIPT_DIR}/04b_interactive_viz.py"
        "--config" "$CONFIG_PATH"
        # Note: --underlay-om not accepted by 04b; it reads underlay_om_id
        # from tile_manifest.json written by Step 4a.
    )

    run_in_env "$TRACKING_ENV" "${VIZ_ARGS[@]}"

    echo ""
    echo "========================================================"
    echo "  STEP 4b COMPLETE"
    echo "========================================================"
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
    python3 -c "
import json
c = json.load(open('${CONFIG_PATH}'))
print('Output dir:    ', c.get('output_dir', '?'))
print('Consensus GPKG:', c.get('consensus_gpkg', '?'))
print('Scores CSV:    ', c.get('phenology_scores_csv', '?'))
print('Viewer:        ', c.get('viewer_html', '?'))
print('Steps done:    ', ', '.join(c.get('steps_completed', [])))
" 2>/dev/null || true
fi

echo ""
