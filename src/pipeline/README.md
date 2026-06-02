# Drone Phenology Monitoring — Pipeline Reference

End-to-end pipeline for extracting tree crown phenology from drone orthomosaics.

**Flow:**  
`orthomosaics (.tif)` → **Step 0** (discover) → **Step 1** (detect crowns) → **Step 2** (track + consensus) → **Step 3** (phenology) → **Step 4** (HTML viewer)

All steps share a single `pipeline_config.json` written by Step 0 and updated by each subsequent step.

---

## Prerequisites

| Task | Conda env |
|---|---|
| Step 0 — discovery | `base` |
| Step 1 — crown detection | `base` |
| Steps 2, 3, 4 — tracking, phenology, viewer | `detectree` |

Activate the right env before running each step, or use `conda run -n <env>` (used in all examples below).

---

## Quick Start — Full Pipeline via Shell Script

```bash
cd /path/to/Drone-Phenology-Monitoring

# LHC dataset (must exclude the bad Dec-9 OM)
bash src/notebooks/pipeline/run_pipeline.sh \
    --om-dir input/input_om_lhc \
    --exclude-stems lhc_09-12-25 \
    --crowns-dir output/detectree_om_lhc_multithreshold/crowns_multithreshold \
    --steps 0,2,3,4

# SIT dataset
bash src/notebooks/pipeline/run_pipeline.sh \
    --om-dir input/input_om_sit \
    --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
    --steps 0,2,3,4
```

If you also need to run crown detection (Step 1), omit `--crowns-dir` and include `1` in `--steps`.

---

## Step-by-Step Reference

### Step 0 — Discover Orthomosaics

**Script:** `00_discover_oms.py`  
**Env:** `base`

Scans a folder of `.tif` files, sorts them chronologically, and writes `pipeline_config.json`.

```
python 00_discover_oms.py \
    --om-dir <PATH>          [required] folder containing .tif orthomosaics
    --output-dir <PATH>      output directory (default: <project_root>/output/<run_name>)
    --run-name <NAME>        human-readable run name (default: derived from om_dir + timestamp)
    --model-path <PATH>      path to .pth Detectree2 model (default: auto-discover from input/detectree_models/)
    --exclude-stems <LIST>   comma-separated stems to drop (e.g. lhc_09-12-25)
    --only-stems <LIST>      comma-separated subset of stems to include — useful for quick test runs
                             e.g. --only-stems sit_03-08-25_dateNotConfirmed,sit_31-08-25_dateNotConfirmed,sit_20-11-25
    --crowns-dir <PATH>      use an existing crowns directory instead of the default
                             <output_dir>/01_detectree/crowns_multithreshold/
                             Reads run_summary.json from that dir to map stems → correct GPKG paths
    --tile-width N           Detectree tile width in metres (default: 25)
    --tile-height N          Detectree tile height in metres (default: 25)
    --tile-buffer N          Detectree tile buffer in metres (default: 15)
    --print-config           print the written config JSON to stdout
```

**Outputs:** `<output_dir>/pipeline_config.json`

**Notes:**
- OMs are sorted chronologically from the cleaned dated filenames used in this repo: `lhc_DD-MM-YY.tif`, `sit_DD-MM-YY.tif`, `sit_DD-MM-YY_dateNotConfirmed.tif`, and `sv_spotX_DD-MM-YY.tif`.
- `_dateNotConfirmed` SIT files are placeholder-dated legacy orthomosaics. They sort by the placeholder date in the filename, not by a confirmed survey date. Legacy `_dateUnknown` crown outputs are still accepted for compatibility.
- `--exclude-stems` is critical for LHC — the Dec-9 OM (`lhc_09-12-25`) has severe misalignment and will corrupt the tracking graph if included.
- `--only-stems` is useful for fast iteration and debugging before committing to a full run.
- `--crowns-dir` with `run_summary.json` correctly resolves GPKG paths even when the detection step used a different naming convention.

---

### Step 1 — Crown Detection (Detectree2)

**Script:** `01_crown_detection.py`  
**Env:** `base`

Runs Detectree2 multi-threshold crown detection on each orthomosaic. **Skip this step if you already have `_multithreshold.gpkg` files** by passing `--crowns-dir` in Step 0.

```
python 01_crown_detection.py \
    --config <PATH>          [required] path to pipeline_config.json
    --device cpu|cuda        inference device (default: cpu)
    --threads N              CPU thread count (default: 6)
    --skip-existing          skip OMs that already have a valid GPKG (default: on)
    --no-skip-existing       force re-run even if GPKG exists
```

**Outputs:** `<output_dir>/01_detectree/crowns_multithreshold/{stem}_multithreshold.gpkg`

**Notes:**
- Each GPKG contains multiple GeoDataFrame layers, one per confidence threshold (e.g. `conf_0p15`, `conf_0p45`, `conf_0p65`).
- Detection is slow — allow several minutes per OM on CPU. Use `--device cuda` if a GPU is available.

---

### Step 2 — Crown Tracking + Consensus

**Script:** `02_crown_tracking.py`  
**Env:** `detectree`

The core step. Loads multi-threshold crowns for all OMs, aligns them across time, builds a tracking graph, assembles crown chains, and generates deduplicated consensus crowns.

```
python 02_crown_tracking.py \
    --config <PATH>                    [required] path to pipeline_config.json

    # Threshold selection
    --base-threshold-tag <TAG>         Crown layer used as the base population for tracking
                                       (default: conf_0p45)
                                       Use conf_0p15 for datasets with dense canopy / lower scores
                                       Use conf_0p65 for cleaner but sparser crown sets
    --align-threshold-tag <TAG>        Crown layer used for image alignment (default: conf_0p65)
                                       Higher threshold → fewer but more reliable anchor crowns for PCC

    # Alignment
    --align-method <METHOD>            Spatial alignment method (default: pcc_tiled)
                                       pcc_tiled  — phase correlation on image tiles (recommended)
                                       pcc        — whole-image phase correlation
                                       ecc        — enhanced correlation coefficient (slower)
                                       crowns     — geometry-based alignment

    # Graph / chain assembly
    --base-max-dist <FLOAT>            Max centroid distance (pixels) to consider two crowns a match
                                       (default: 30.0)
    --overlap-gate <FLOAT>             Min IoU overlap required to form a graph edge (default: 0.10)
    --min-base-similarity <FLOAT>      Min shape similarity score for a match (default: 0.30)
    --classify-mode <MODE>             Edge classifier mode: balanced or strict (default: balanced)
    --min-partial-len <INT>            Min chain length to qualify as a partial chain (default: 5)
    --min-partial-ratio <FLOAT>        Min one-to-one match ratio for partial chains (default: 0.9)
                                       Raise to 1.0 to require near-perfect temporal coverage

    # Deduplication
    --dedup-iou <FLOAT>                IoU threshold above which overlapping consensus crowns are merged
                                       (default: 0.75)
    --dedup-containment-buffer <FLOAT> Buffer (pixels) applied during containment dedup (default: 5.0)

    # Output control
    --skip-chain-viz                   Skip per-chain strip image output (significantly faster)
    --skip-consensus-viz               Skip per-consensus strip image output (significantly faster)
    --skip-diagnostics                 Skip all diagnostic PNG/CSV outputs:
                                         alignment_shifts.png, match_rates_by_pair.png,
                                         chain_length_distribution.png, consensus_overlay_om1_raw.png
    --skip-if-done                     Skip entirely if consensus_crowns_complete_all.gpkg already exists
```

**Outputs:**

| File | Description |
|---|---|
| `02_tracking/consensus_crowns_complete_all.gpkg` | Final deduplicated consensus crowns (main output) |
| `02_tracking/consensus_crowns_om1_phenology.geojson` | Consensus crowns in OM1 coordinate space |
| `02_tracking/consensus_crowns_summary.json` | Crown count, chain stats, dedup summary |
| `02_tracking/tracking_quality_metrics.json` | Graph quality metrics |
| `02_tracking/tracking_quality_report.txt` | Human-readable quality summary |
| `02_tracking/diagnostics/alignment_shifts.csv` | Per-OM dx/dy alignment shifts |
| `02_tracking/diagnostics/alignment_shifts.png` | Alignment shift visualisation |
| `02_tracking/diagnostics/match_rates_by_pair.png` | Match rate between consecutive OMs |
| `02_tracking/diagnostics/chain_length_distribution.png` | Histogram of chain lengths |
| `02_tracking/chain_viz/` | Chain strip images (if `--skip-chain-viz` not set) |
| `02_tracking/consensus_viz/` | Consensus strip images (if `--skip-consensus-viz` not set) |

**Tuning guide:**

| Goal | Adjustment |
|---|---|
| More crowns (denser/lower-confidence dataset) | `--base-threshold-tag conf_0p15` |
| Fewer, cleaner crowns | `--base-threshold-tag conf_0p65` |
| Better alignment in dense canopy | `--align-method pcc_tiled` (default) |
| Allow shorter chains (partial coverage) | Lower `--min-partial-len` (e.g. 3) |
| Stricter temporal coverage requirement | Raise `--min-partial-ratio` toward 1.0 |
| Less aggressive crown merging | Raise `--dedup-iou` (e.g. 0.85) |
| Fast debug run (no visualisations) | Add all three `--skip-*` flags |

---

### Step 3 — Phenology Analysis

**Script:** `03_phenology_analysis.py`  
**Env:** `detectree`

Extracts image patch features per crown per OM and computes deciduous scores and phenophase labels. Produces the canonical `tree_master_geojson.geojson`.

```
python 03_phenology_analysis.py \
    --config <PATH>              [required] path to pipeline_config.json
    --dataset-id <ID>            dataset identifier embedded in output (default: from run_name in config)

    # Feature extraction thresholds
    --veg-min <FLOAT>            minimum vegetation fraction to consider an OM observation valid
                                 (default: 0.45) — crowns below this are flagged as bad observations

    # Deciduous classification
    --ds-thresh <FLOAT>          deciduous score threshold — crowns scoring above this are labelled
                                 deciduous (default: 0.70)

    # Phenophase thresholds
    --on-thresh <FLOAT>          score above which a crown is in "leaf-on" state (default: 0.65)
    --off-thresh <FLOAT>         score below which a crown is in "leaf-off" state (default: 0.35)
                                 scores between off-thresh and on-thresh are labelled "transition"

    --skip-if-done               skip if tree_master_geojson.geojson already exists
```

**Outputs:**

| File | Description |
|---|---|
| `03_phenology/tree_master_geojson.geojson` | Canonical per-crown output with all phenology embedded |
| `03_phenology/phenology_features_raw.csv` | Raw per-crown per-OM feature table |
| `03_phenology/leafshed_tree_scores.csv` | Deciduous scores per crown |
| `03_phenology/leafshed_phenophase_by_om.csv` | Phenophase label per crown per OM |
| `03_phenology/leafshed_normalizers.json` | Normalisation parameters used |
| `03_phenology/leafshed_config.json` | Thresholds used in this run |

**Notes:**
- `RuntimeWarning: All-NaN slice encountered` messages are expected and non-fatal. They occur for crowns at the edge of some OMs that have no valid pixels at certain dates.
- The `--ds-thresh` default of 0.70 is calibrated for this dataset. Lower it slightly (e.g. 0.60) if you want to classify more crowns as deciduous; raise it for stricter classification.
- Phenophase labels (`leaf-on`, `leaf-off`, `transition`) are embedded per OM per crown in `tree_master_geojson.geojson`.

---

### Step 4 — Interactive HTML Viewer

**Script:** `04_interactive_viz.py`  
**Env:** `detectree`

Generates a standalone Leaflet-based HTML viewer. No server required — open `index.html` directly in a browser.

```
python 04_interactive_viz.py \
    --config <PATH>              [required] path to pipeline_config.json
    --underlay-om first|last|N   which OM to use as the map background (default: last)
                                 first — earliest OM; last — latest OM; N — 1-based index
    --max-base-px N              max pixel dimension for the base underlay PNG (default: 2600)
                                 lower to reduce file size; raise for sharper background
    --force-regen-crops          regenerate all crop PNGs even if they already exist
    --skip-if-done               skip if index.html already exists
```

**Outputs:**

| File | Description |
|---|---|
| `04_viewer/index.html` | Self-contained Leaflet viewer — open in any browser |
| `04_viewer/base_underlay_OM{N}_{stem}.png` | Base raster underlay image |
| `04_viewer/crowns_underlay_OM{N}_{stem}_pixels.geojson` | Crown polygons in pixel space |
| `04_viewer/manifest.json` | Viewer metadata (crown count, OM list, etc.) |
| `04_viewer/crops/crown_{NNNN}/OM{N}_{stem}.png` | Per-crown per-OM crop images |
| `04_viewer/phenology_overview.png` | Static matplotlib phenology summary chart |

---

## Shell Script Reference (`run_pipeline.sh`)

Orchestrates all steps, handles conda env switching, and passes arguments through to each script.

```bash
bash run_pipeline.sh \
    --om-dir <PATH>              [required] folder containing .tif orthomosaics

    --run-name <NAME>            human-readable run name (default: auto-generated)
    --output-dir <PATH>          output directory (default: <project_root>/output/<run_name>)
    --model-path <PATH>          path to .pth model (default: auto-discover)
    --exclude-stems <LIST>       comma-separated stems to exclude
    --crowns-dir <PATH>          use existing crowns (skip step 1)

    # Detection (step 1) options
    --tile-width N               tile width in metres (default: 25)
    --tile-height N              tile height in metres (default: 25)
    --tile-buffer N              tile buffer in metres (default: 15)
    --device cpu|cuda            inference device (default: cpu)
    --threads N                  CPU thread count (default: 6)
    --skip-existing              skip already-valid GPKGs (default: on)
    --no-skip-existing           force re-run detection

    # Tracking (step 2) options
    --align-method METHOD        alignment method (default: pcc_tiled)
    --align-threshold-tag TAG    alignment threshold layer (default: conf_0p65)
    --min-partial-len N          min partial chain length (default: 5)
    --min-partial-ratio R        min partial chain ratio (default: 0.9)
    --skip-chain-viz             skip chain strip images
    --skip-consensus-viz         skip consensus strip images

    # Viewer (step 4) options
    --underlay-om last|first|N   underlay OM choice (default: last)

    # Orchestration
    --steps 0,1,2,3,4            which steps to run (default: all)
    --base-env NAME              conda env for steps 0,1 (default: base)
    --tracking-env NAME          conda env for steps 2,3,4 (default: detectree)
```

---

## Known Dataset-Specific Settings

### LHC (8 OMs)

```bash
# Step 0
python 00_discover_oms.py \
    --om-dir input/input_om_lhc \
    --output-dir output/my_lhc_run \
    --crowns-dir output/detectree_om_lhc_multithreshold/crowns_multithreshold \
    --exclude-stems lhc_09-12-25 \
    --run-name my_lhc_run

# Step 2 (defaults work — only conf_0p15 exists in LHC GPKGs, tracker auto-selects it)
conda run -n detectree python 02_crown_tracking.py \
    --config output/my_lhc_run/pipeline_config.json \
    --skip-chain-viz --skip-consensus-viz --skip-diagnostics

# Steps 3 & 4
conda run -n detectree python 03_phenology_analysis.py --config output/my_lhc_run/pipeline_config.json
conda run -n detectree python 04_interactive_viz.py --config output/my_lhc_run/pipeline_config.json
```

Expected: ~87 consensus crowns, ~69% deciduous.

> ⚠️ **Always exclude `lhc_09-12-25`** (Dec-9 OM). It has severe spatial misalignment that corrupts the tracking graph.

### SIT (14 OMs)

```bash
# Step 0
python 00_discover_oms.py \
    --om-dir input/input_om_sit \
    --output-dir output/my_sit_run \
    --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
    --run-name my_sit_run

# Step 2 — must use conf_0p15; default conf_0p45 is too restrictive for this dataset
conda run -n detectree python 02_crown_tracking.py \
    --config output/my_sit_run/pipeline_config.json \
    --base-threshold-tag conf_0p15 \
    --skip-chain-viz --skip-consensus-viz --skip-diagnostics

# Steps 3 & 4
conda run -n detectree python 03_phenology_analysis.py --config output/my_sit_run/pipeline_config.json
conda run -n detectree python 04_interactive_viz.py --config output/my_sit_run/pipeline_config.json
```

Expected: ~131 consensus crowns, ~79% deciduous.

> **Why `conf_0p15` for SIT?** The default `conf_0p45` base layer yields ~96 crowns/OM in SIT (too restrictive). `conf_0p15` gives ~287 crowns/OM and produces 131 consensus crowns matching the reference notebook baseline.

---

## Output Directory Structure

After a full run, `output/<run_name>/` will contain:

```
pipeline_config.json          ← shared config, updated after each step

01_detectree/
  crowns_multithreshold/
    {stem}_multithreshold.gpkg  ← multi-threshold crown GeoPackages

02_tracking/
  consensus_crowns_complete_all.gpkg   ← main output: deduplicated consensus crowns
  consensus_crowns_om1_phenology.geojson
  consensus_crowns_summary.json
  tracking_quality_metrics.json
  tracking_quality_report.txt
  diagnostics/
    alignment_shifts.csv / .png
    match_rates_by_pair.png
    chain_length_distribution.png
    chain_breakdown.json

03_phenology/
  tree_master_geojson.geojson   ← canonical per-crown phenology output
  phenology_features_raw.csv
  leafshed_tree_scores.csv
  leafshed_phenophase_by_om.csv
  leafshed_normalizers.json
  leafshed_config.json

04_viewer/
  index.html                    ← open this in a browser
  manifest.json
  base_underlay_OM{N}_{stem}.png
  crowns_underlay_OM{N}_{stem}_pixels.geojson
  phenology_overview.png
  crops/
    crown_0001/
      OM01_{stem}.png
      OM02_{stem}.png
      ...
```

---

## Fast Debug / Iteration Workflow

When fixing bugs or tuning parameters, skip all visualisations and run only the tracking step on a small subset of OMs:

```bash
# 1. Discover only 3 OMs
python 00_discover_oms.py \
    --om-dir input/input_om_sit \
    --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
    --only-stems sit_03-08-25_dateNotConfirmed,sit_31-08-25_dateNotConfirmed,sit_20-11-25 \
    --output-dir output/debug_run \
    --run-name debug_run

# 2. Track with all viz disabled
conda run -n detectree python 02_crown_tracking.py \
    --config output/debug_run/pipeline_config.json \
    --base-threshold-tag conf_0p15 \
    --skip-chain-viz --skip-consensus-viz --skip-diagnostics
```

This runs in a fraction of the time of a full multi-OM run.

---

## Threshold Tag Reference

The GPKG files produced by Detectree2 contain multiple layers. Common tags:

| Tag | Confidence threshold | Crown density |
|---|---|---|
| `conf_0p15` | 0.15 | Densest — most crowns, includes lower-confidence detections |
| `conf_0p45` | 0.45 | Balanced — default for most datasets |
| `conf_0p65` | 0.65 | Sparse — high-confidence only, used for alignment |

Use `--base-threshold-tag` to control which layer is the source population for tracking.  
Use `--align-threshold-tag` to control which layer is used for image alignment (higher is better).

If a requested tag is not present in a GPKG, the tracker automatically falls back to the layer with the highest available threshold.

---

## `pipeline_config.json` Schema

All steps read and write this file. Key fields:

```json
{
  "run_name": "my_run",
  "created_at": "...",
  "project_root": "/path/to/project",
  "om_dir": "/path/to/orthomosaics",
  "output_dir": "/path/to/output/my_run",
  "model_path": "/path/to/model.pth",
    "om_stems": ["sit_03-08-25_dateNotConfirmed", "sit_20-11-25", ...],
  "exclude_stems": [],
  "pairs": [
        ["/path/to/crown.gpkg", "/path/to/ortho.tif", "sit_03-08-25_dateNotConfirmed"],
    ...
  ],
  "tile_width": 25,
  "tile_height": 25,
  "tile_buffer": 15,
  "detectree_dir": ".../01_detectree",
  "crowns_dir": ".../01_detectree/crowns_multithreshold",
  "tracking_dir": ".../02_tracking",
  "phenology_dir": ".../03_phenology",
  "viewer_dir": ".../04_viewer",
  "steps_completed": ["step_0", "step_2", "step_3", "step_4"]
}
```
