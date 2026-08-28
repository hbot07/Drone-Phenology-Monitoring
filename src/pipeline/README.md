# Drone Phenology Pipeline

This is the reusable pipeline for going from dated orthomosaics to tracked consensus crowns, crown-level phenology metrics, and a browser viewer.

Flow:

```text
orthomosaics (.tif)
  -> 00_discover_oms.py
  -> 01_crown_detection.py
  -> 02_crown_tracking.py
  -> 03_phenology_analysis.py
  -> 04a_cog_tiling.py          ← generates COGs, XYZ tile pyramids, manifest + info file
  -> 04b_interactive_viz.py     ← generates interactive HTML viewer
```

All steps share one `pipeline_config.json`, created by Step 0 and updated by later steps.

## Environments

Use two conda environments:

| Step | Environment | Purpose |
|---|---|---|
| 0 | `dpm-detectree` | discover inputs and write config |
| 1 | `dpm-detectree` | Detectree2 crown detection |
| 2 | `dpm-tracking` | crown alignment, graph tracking, consensus geometries |
| 3 | `dpm-tracking` | phenology metrics and labels |
| 4a | `dpm-tracking` | COG generation, XYZ tiling, tile manifest |
| 4b | `dpm-tracking` | standalone HTML viewer |

Create or update them from the repository root:

```bash
bash scripts/setup_dpm_detectree.sh
bash scripts/setup_dpm_tracking.sh
```

The conda YAML files live in `envs/`. Companion pip requirement files live in `requirements/`.

## Input Naming

The pipeline expects one orthomosaic GeoTIFF per site/date. File stems should contain enough information to sort dates correctly.

Recommended names:

```text
<site>_DD-MM-YY.tif
<site>_DD-MM-YY_dateNotConfirmed.tif
<site>_spot<id>_DD-MM-YY.tif
```

Examples:

```text
site_a_15-01-26.tif
site_a_29-01-26.tif
sv_spot1_10-05-26.tif
```

For older local datasets, the discovery step also understands `lhc_DD-MM-YY`, `sit_DD-MM-YY`, `sv_spotX_DD-MM-YY`, `sit_omN`, and legacy `odm_orthophoto_D_M_YY` stems.

## Configure Paths With `.env`

Copy the root example file:

```bash
cp .env.example .env
```

Set at least:

```bash
DPM_OM_DIR=input/input_om_sit
DPM_OUTPUT_DIR=output/example_sit_run
DPM_MODEL_PATH=input/detectree_models/250312_flexi.pth
DPM_STEPS=0,1,2,3,4a,4b
```

Then run:

```bash
bash src/pipeline/run_pipeline.sh
```

CLI flags override `.env`, so this also works:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir /path/to/clean_orthomosaics \
  --output-dir /path/to/output/my_run \
  --model-path /path/to/model.pth \
  --steps 0,1,2,3,4a,4b
```

If crown detection was already run, reuse the crowns and skip Step 1:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir /path/to/clean_orthomosaics \
  --crowns-dir /path/to/crowns_multithreshold \
  --output-dir /path/to/output/my_run \
  --steps 0,2,3,4a,4b
```

## Step 0: Discover Orthomosaics

Script: `00_discover_oms.py`

```bash
conda run -n dpm-detectree python src/pipeline/00_discover_oms.py \
  --om-dir /path/to/clean_orthomosaics \
  --output-dir output/my_run \
  --model-path input/detectree_models/250312_flexi.pth \
  --run-name my_run
```

Useful options:

- `--exclude-stems stem1,stem2`: remove bad dates from the series.
- `--only-stems stem1,stem2`: run a quick subset.
- `--crowns-dir /path/to/crowns_multithreshold`: reuse existing Detectree2 outputs.
- `--print-config`: print the full generated config.

Output:

```text
<output_dir>/pipeline_config.json
```

## Step 1: Detect Crowns With Detectree2

Script: `01_crown_detection.py`

```bash
conda run -n dpm-detectree python src/pipeline/01_crown_detection.py \
  --config output/my_run/pipeline_config.json \
  --device cpu \
  --threads 6
```

Output:

```text
<output_dir>/01_detectree/crowns_multithreshold/{stem}_multithreshold.gpkg
```

Each GPKG contains multiple layers, one per confidence threshold, such as `conf_0p15`, `conf_0p45`, and `conf_0p65`.

Use `--device cuda` only when the environment and machine support GPU inference.

## Step 2: Track Crowns And Build Consensus Geometries

Script: `02_crown_tracking.py`

```bash
conda run -n dpm-tracking python src/pipeline/02_crown_tracking.py \
  --config output/my_run/pipeline_config.json \
  --base-threshold-tag conf_0p45 \
  --align-threshold-tag conf_0p65 \
  --align-method pcc_tiled
```

Main ideas:

- Orthomosaics are aligned through phase correlation, usually `pcc_tiled`.
- Crowns are shifted using the same alignment offsets.
- Consecutive dates are matched using IoU, overlap, centroid distance, and shape similarity.
- A graph is built where nodes are dated crowns and edges connect likely same-tree crowns.
- Full and partial chains are converted into consensus crown geometries.
- Consensus crowns are deduplicated before phenology extraction.

Common tuning options:

| Goal | Option |
|---|---|
| Denser crown population | `--base-threshold-tag conf_0p15` |
| Cleaner, sparser crown population | `--base-threshold-tag conf_0p65` |
| Stronger alignment anchors | `--align-threshold-tag conf_0p65` |
| Allow shorter partial chains | lower `--min-partial-len` |
| Require stricter temporal coverage | raise `--min-partial-ratio` |
| Faster debug run | `--skip-chain-viz --skip-consensus-viz --skip-diagnostics` |

Main outputs:

```text
02_tracking/consensus_crowns_complete_all.gpkg
02_tracking/consensus_crowns_om1_phenology.geojson
02_tracking/tracking_quality_report.txt
02_tracking/tracking_quality_metrics.json
02_tracking/diagnostics/
```

## Step 3: Phenology Analysis

Script: `03_phenology_analysis.py`

```bash
conda run -n dpm-tracking python src/pipeline/03_phenology_analysis.py \
  --config output/my_run/pipeline_config.json
```

This crops each consensus crown from each orthomosaic and extracts features such as GCC, RCC, channel statistics, grayscale texture, Laplacian variance, vegetation fraction, robust date-normalized signals, deciduous scores, and leaf-on/leaf-off/transition labels.

Main outputs:

```text
03_phenology/tree_master_geojson.geojson
03_phenology/phenology_features_raw.csv
03_phenology/leafshed_tree_scores.csv
03_phenology/leafshed_phenophase_by_om.csv
```

## Step 4a: COG Tiling

Script: `04a_cog_tiling.py`

Must run before Step 4b. Converts all orthomosaics into Cloud-Optimised GeoTIFFs and generates
XYZ tile pyramids for efficient browser streaming.

```bash
conda run -n dpm-tracking python src/pipeline/04a_cog_tiling.py \
  --config output/my_run/pipeline_config.json \
  --underlay-om last
```

What it does:

- Reprojects each orthomosaic to EPSG:3857 (Web Mercator)
- Stretches bands to uint8 RGB with LZW compression and internal overviews (2×, 4×, 8×, 16×, ...)
- Generates XYZ tile pyramids at 256×256 px per tile up to `--max-cog-zoom` (default: z22)
- Writes `tile_manifest.json` consumed by Step 4b
- Writes `04a_OM_INFO.txt` with full per-OM and per-zoom statistics (see below)

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--underlay-om` | `last` | Which OM loads by default in the viewer (`first`, `last`, or an integer OM number) |
| `--tile-size` | `256` | Tile width and height in pixels |
| `--max-cog-zoom` | `22` | Maximum zoom level for the tile pyramid |
| `--force-regen-cogs` | off | Regenerate COGs and tiles even if they already exist on disk |
| `--skip-if-done` | off | Exit immediately if `tile_manifest.json` already exists |

Output layout:

```text
04_viewer/
├── cogs/
│   ├── OM01_<stem>.tif
│   ├── OM02_<stem>.tif
│   └── ...
├── tiles/
│   ├── OM01_<stem>/
│   │   ├── 13/
│   │   ├── 14/
│   │   └── ...  (one directory per zoom level, z10 → z22)
│   ├── OM02_<stem>/
│   └── ...
├── tile_manifest.json
└── 04a_OM_INFO.txt
```

### Info File (`04a_OM_INFO.txt`)

Auto-generated on every run at `04_viewer/04a_OM_INFO.txt`. Contains:

- Run metadata (timestamp, run name, paths)
- Summary (OM count, underlay OM, tile size, compression, total storage)
- Per-OM detail for every OM:
  - Geographic coverage (bounds in degrees, width and height in metres, area in km²)
  - COG file stats (path, size on disk, pixel dimensions, bands, dtype, native GSD, compression ratio vs raw)
  - Tile pyramid stats (zoom range, total tiles counted from disk, total size, average tile size)
  - Per-zoom breakdown table: tiles on disk, tiles visible in a 1080p viewport, GSD (cm/px), tile resolution (px/m), ground area covered per tile (m), quality label
  - Tile size analysis (ground coverage at min / mid / max zoom)
  - Before → after comparison (raw uncompressed estimate vs COG vs tiles, compression ratio, largest file the browser ever fetches vs full raster)
- Global before → after across all OMs combined
- Processing details (what each of the four sub-steps does)
- Zoom level reference table at the site's average latitude (z10 → z23)

### Libraries used in 04a

Already in `dpm-tracking`: `rasterio`, `rasterio.warp`, `rasterio.enums`, `rasterio.shutil`, `numpy`

May need installing separately:

| Library | When needed | Install |
|---|---|---|
| `mercantile` | Fallback tiler only (when `gdal2tiles` is not on PATH) | `pip install mercantile` |
| `imageio` | Fallback tiler only (writes PNG tiles) | `pip install imageio` |

External tool (not a Python library): `gdal2tiles` / `gdal2tiles.py` — called via `subprocess` as the primary tiler. Comes with GDAL (`conda install -c conda-forge gdal`). If found on PATH, `mercantile` and `imageio` are never used.

## Step 4b: Interactive Viewer

Script: `04b_interactive_viz.py`

Depends on Step 4a (`tile_manifest.json` must exist). Generates a standalone HTML viewer.

```bash
conda run -n dpm-tracking python src/pipeline/04b_interactive_viz.py \
  --config output/my_run/pipeline_config.json
```

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--skip-if-done` | off | Exit immediately if `index.html` already exists |

Output:

```text
04_viewer/index.html
```

Serve the `04_viewer/` directory with any HTTP server and open `index.html` in a browser:

```bash
cd output/my_run/04_viewer
python -m http.server 8000
# Open: http://localhost:8000/index.html
```

### Viewer Layout

The viewer uses a dual-panel layout:

- **Left panel (60%)**: Full orthomosaic map showing all crown polygons as white outlines. The main map is zoom-locked so the OM always covers at least 60% of the screen. Zoom and pan are free with no snap-back.
- **Right panel (40%)**: Crown-specific analysis, visible after clicking a crown.

### Crown Polygons

All crowns are drawn as white polygon outlines on the main map. Border weight increases on hover and on selection. There is no colour fill — outlines only.

### Right Panel (after selecting a crown)

- **Observation month label**: Shows the current OM stem and date.
- **Slider**: Scrubs through all OMs (OM01 → OMN). Dragging updates the large tile map.
- **Animate / Pause button**: Auto-cycles through all OMs at 800 ms per step. While animating, the time-series grid hides so the large tile map has full focus. The grid reappears when animation ends or is paused.
- **Reset button**: Jumps back to OM01.
- **Large tile map**: Square map showing the selected crown from the current OM. Fitted tightly to the crown bounds. No zoom or pan — view only.
- **Species annotation**: Free-text input saved to browser `localStorage` per crown. Persists across page reloads.
- **Time-series grid**: Single-column scrollable list of small square tile maps, one per OM. Each card is non-interactive (view only). The card for the leaf-on OM gets a green border; the card for the leaf-off OM gets a red border.

### Search and Filtering

- **Search bar**: Matches on crown ID or species annotation (substring).
- **Filter pills**: All / Deciduous / Evergreen / Uncertain — filters the crown overlay live.

### Other Features

- **Dark / light theme toggle**
- **Crown comparison panel**: Two independent slots (A and B), each with its own tile map and OM slider, for side-by-side crown comparison.
- **Draw tool**: Polygon and rectangle annotation on the main map, exportable as GeoJSON.

### Libraries used in 04b

Already in `dpm-tracking` (no extra installs): `rasterio`, `rasterio.warp`, `geopandas`, `pandas`, `shapely`, `matplotlib`, `numpy`

Internal pipeline module: `tree_tracking` (`TreeTrackingGraph`) — must be on the Python path.

JavaScript libraries loaded via CDN (no Python install): Leaflet.js 1.9.4, Leaflet.draw 1.0.4.

## Output Layout

After a full run:

```text
output/my_run/
  pipeline_config.json
  01_detectree/
    crowns_multithreshold/
  02_tracking/
    consensus_crowns_complete_all.gpkg
    diagnostics/
  03_phenology/
    tree_master_geojson.geojson
    phenology_features_raw.csv
    leafshed_tree_scores.csv
    leafshed_phenophase_by_om.csv
  04_viewer/
    index.html
    tile_manifest.json
    04a_OM_INFO.txt
    cogs/
      OM01_<stem>.tif
      OM02_<stem>.tif
      ...
    tiles/
      OM01_<stem>/
        13/
        14/
        ...
      OM02_<stem>/
      ...
```

## Fast Debug Workflow

Use a few orthomosaics, skip visualizations, and run only tracking:

```bash
conda run -n dpm-detectree python src/pipeline/00_discover_oms.py \
  --om-dir input/input_om_sit \
  --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
  --only-stems sit_03-08-25_dateNotConfirmed,sit_31-08-25_dateNotConfirmed,sit_20-11-25 \
  --output-dir output/debug_run \
  --run-name debug_run

conda run -n dpm-tracking python src/pipeline/02_crown_tracking.py \
  --config output/debug_run/pipeline_config.json \
  --base-threshold-tag conf_0p15 \
  --skip-chain-viz \
  --skip-consensus-viz \
  --skip-diagnostics

conda run -n dpm-tracking python src/pipeline/04a_cog_tiling.py \
  --config output/debug_run/pipeline_config.json \
  --underlay-om last

conda run -n dpm-tracking python src/pipeline/04b_interactive_viz.py \
  --config output/debug_run/pipeline_config.json
```

## Appendix: Local Dataset Settings

These are examples from the current IITD/Sanjay Van work. Treat them as starting points, not generic requirements.

LHC:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir input/input_om_lhc \
  --exclude-stems lhc_09-12-25 \
  --crowns-dir output/detectree_om_lhc_multithreshold/crowns_multithreshold \
  --steps 0,2,3,4a,4b
```

The `lhc_09-12-25` orthomosaic is excluded because it is badly misaligned relative to the rest of that local series.

SIT:

```bash
bash src/pipeline/run_pipeline.sh \
  --om-dir input/input_om_sit \
  --crowns-dir output/detectree_om_sit_multithreshold/crowns_multithreshold \
  --base-threshold-tag conf_0p15 \
  --steps 0,2,3,4a,4b
```

For that local SIT crown set, `conf_0p15` has been useful because higher thresholds are too sparse.

## Troubleshooting

- If Step 0 cannot find a model, set `DPM_MODEL_PATH` or pass `--model-path`.
- If Step 1 imports fail, check the `dpm-detectree` environment and Detectree2/Detectron2 installation.
- If Step 2 produces very short chains, inspect `02_tracking/diagnostics/alignment_shifts.csv` and the match-rate plots.
- If a single date breaks many chains, visually inspect that orthomosaic and consider excluding that stem.
- If phenology features are mostly missing, check whether crowns fall outside later orthomosaic bounds or whether rasters have nodata/black edges.
- If Step 4a raises a `UnicodeEncodeError` writing `04a_OM_INFO.txt`, the system locale does not support UTF-8. The info file is always written with `encoding='utf-8'` — check that your Python environment is not overriding this.
- If `index.html` opens but tiles do not load, make sure you are serving `04_viewer/` through an HTTP server (`python -m http.server 8000`) and not opening `index.html` directly as a `file://` URL. Browsers block local tile requests on `file://` origins.
- If the tile manifest is missing when running Step 4b, run Step 4a first.
- If tiles exist but look blurry at high zoom, the `--max-cog-zoom` ceiling may be set too low. Increase it (e.g. `--max-cog-zoom 22`) and rerun Step 4a with `--force-regen-cogs`.
