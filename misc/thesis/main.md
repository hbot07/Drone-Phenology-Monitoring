# Drone Phenology Monitoring: Technical Project Map

This document records the current technical understanding of the repository, the thesis/report material, the active pipeline, the older experiment notebooks, and the related UAV/satellite workflow described in the companion paper excerpt.

## 1. Core Research Problem

The project monitors individual tree phenology using repeated drone surveys over fixed sites. A drone is flown over the same area roughly every two weeks. The raw nadir images are processed into georeferenced orthomosaics. Tree crowns are segmented on each orthomosaic. The hard problem is then temporal identity: deciding which crown polygon in date `t + 1` is the same biological tree as a polygon in date `t`, despite georeferencing drift, photogrammetric distortions, lighting changes, leaf shedding, crown boundary changes, split/merge segmentation errors, and detection failures.

The final scientific object is not just a crown detection. It is a temporally stable tree identity with:

- a consensus crown geometry;
- per-date RGB crown crops;
- per-date phenology signals;
- deciduous/evergreen or leaf-on/leaf-off labels;
- field/QField species annotations when available;
- eventual satellite-scale feature extraction and classifier experiments.

The full conceptual chain is:

```text
site selection
  -> drone mission planning
  -> raw UAV images
  -> WebODM orthomosaic / DSM / DTM products
  -> Detectree2 crown polygons
  -> multi-threshold crown stores
  -> orthomosaic alignment
  -> graph-based temporal crown tracking
  -> chain extraction
  -> consensus crown geometry
  -> per-date crown crops
  -> phenology features and labels
  -> field/species validation
  -> satellite/GEE/DINO feature extraction
  -> supervised classifiers for species / phenology traits
```

The companion-paper context overlaps strongly with the repository up to crown delineation, QField validation, Google Earth visualisation, unsupervised/supervised crown embedding experiments, and satellite classifier experiments. The paper also mentions DTM + DSM canopy-height modelling; this repository currently has support and context for photogrammetric products, but the active pipeline does not yet compute canopy height from DTM/DSM.

## 2. Thesis and Report Material in `misc/thesis`

`main.md` and `main.tex` were empty before this write-up. The actual thesis content was in PDFs.

### `Drone Monitoring.pptx (1).pdf`

Early project presentation. It defines the initial pipeline:

- DJI Mini 4 Pro imagery.
- Map Pilot Pro flight planning.
- WebODM orthomosaic generation.
- DeepForest tree detection baseline.
- Segmentation attempts to reduce non-tree false positives.
- Detectree2 as the more suitable polygon-based detector.
- A first data-structure idea for tracking trees through detections.

Important technical settings reported here:

- altitude about 80 m;
- speed about 3 m/s;
- forward and side overlap about 80%;
- gimbal at near-nadir / 90 degrees;
- DJI metadata conversion from XMP to EXIF for better photogrammetry compatibility.

### `Drone_Phenology_Monitoring_Report_May_2025.pdf`

Formal early report. It is mostly about data acquisition and first detection/tracking baselines.

Main contents:

- Motivation: scalable tree-level phenology monitoring and satellite ground truth.
- Hardware/software: DJI Mini 4 Pro, Map Pilot Pro, WebODM, CVAT, DeepForest, Detectree2.
- Study areas: SAC, LHC front patch, SIT.
- DeepForest: bounding-box detector, high false positives, missed trees, no polygon boundaries.
- Detectree2: Mask R-CNN polygon segmentation on tiled orthomosaics.
- Cleaning: stitch tile predictions, simplify polygons, remove overlaps with `clean_crowns`.
- Evaluation: SIT CVAT polygon ground truth, IoU matching at threshold 0.5.
- Early tracking: Hungarian matching with IoU cost.

Reported detection metrics:

```text
GT crowns: 124
Predictions: 73
TP: 32
FP: 41
FN: 92
Precision: 43.8%
Recall: 25.8%
F1: 32.5%
Average IoU of true matches: 0.72
```

Interpretation: Detectree2 polygons are geometrically reasonable when correct, but recall is low and false positives remain.

### `presentation dpm.pdf`

Intermediate tracking presentation. It explains why Hungarian matching is insufficient:

- one-to-one assignment cannot represent splits and merges;
- local pairwise optimisation can block plausible secondary links;
- multi-date consistency is not represented.

It introduces the graph formulation:

- node = crown instance at one date;
- edge = plausible temporal match between consecutive dates;
- chain = path through the temporal graph;
- virtual node = placeholder for missing detection to preserve continuity.

Similarity is described as a weighted combination of spatial proximity, area ratio, and IoU.

### `Drone Phenology Monitoring End Term Presentation Nov 2025.pdf`

Presentation around the transition from simple detection to graph tracking.

Key points:

- Detectree2 tuning on IITD data.
- Tile/buffer/threshold experiments.
- Reported useful Detectree2 settings in that phase:
  - buffer around 30;
  - tile width/height around 45;
  - simplify tolerance 0.3;
  - confidence threshold around 0.7;
  - area threshold around 0.5.
- Repeated surveys every two weeks.
- SIT/LHC/SAC species motivations.
- Hungarian matching and case-based matching.
- Detection-score matrix idea: trees seen consistently across dates are likely reliable; inconsistent detections should be manually reviewed.

### `Drone_Phenology_Monitoring_Report_Nov_2025 (1).pdf`

Most complete older report. It expands the project from detection to robust multi-date tracking.

Important additions:

- IIT Delhi sites plus Sanjay Van dry-deciduous sites.
- Formal terminology: orthomosaic, crown, tree, chain, chain width, virtual node, IoU.
- Alignment methodology:
  - load crown GPKGs;
  - harmonise CRS;
  - extract centroids;
  - estimate translation vectors;
  - apply shifts;
  - evaluate with IoU.
- Graph-based multi-date tracking:
  - directed graph over consecutive orthomosaic layers;
  - edge features: IoU, overlap ratios, centroid distance, area ratios, containment flags;
  - conditional threshold matching.
- Similarity score:
  - spatial similarity from centroid distance;
  - area similarity from relative crown size;
  - IoU;
  - optional compactness/eccentricity.
- Chain typology:
  - full width-1 chains: reliable stable candidates;
  - full branching chains: split/merge or segmentation ambiguity;
  - partial/broken chains: detection failures or real phenological disappearance.
- Virtual nodes:
  - conceptual placeholders for missing crowns;
  - centroid/area interpolation;
  - separate algorithmic continuity from ecological absence.

This report mentions runtime and data organisation patterns that are now reflected in `src/pipeline`.

### `Drone Monitoring Mid Term Presentation - Feb 2026 (1).pdf`

Closest presentation to the current pipeline.

Key additions:

- Multi-threshold crown detection:
  - fixed threshold is brittle under illumination and seasonal changes;
  - save crown layers across confidence thresholds;
  - use lower thresholds when high-confidence matching fails.
- Phase Cross Correlation (PCC) alignment:
  - OM1 is fixed as spatial anchor;
  - FFT-based image registration estimates translation;
  - reported match-rate improvement and IoU recovery.
- Consensus crowns:
  - use a single polygon per tree for stable sampling across dates;
  - medoid, intersection core, and union-shrink discussed;
  - current code uses medoid by default.
- Partial chains:
  - full chains have best quality but limited coverage;
  - long partial chains can expand spatial coverage if filtered.
- Phenology metrics:
  - GCC, RCC;
  - HSV vegetation fraction;
  - GLCM-style texture in notebooks, gray entropy and Laplacian variance in current code;
  - per-observation QC.
- QField ground-truth marking for species and tree status.

### `Drone Phenology Monitoring Parth and Gaurav 2402 Major Presentation (1).pdf`

Recent overview presentation:

- reiterates multi-threshold detection;
- PCC alignment;
- consensus crowns from full and partial chains;
- split/merge lookahead idea;
- phenology feature extraction;
- QField validation;
- future CNN/non-tree filtering and user-adjustable confidence/tracking aggressiveness.

## 3. Active Pipeline in `src/pipeline`

`src/pipeline` is the current operational wrapper. It is not where all algorithmic code lives. It orchestrates the process and delegates tracking/phenology logic mainly to `src/flask_app_tracking/tree_tracking.py` and `src/flask_app_tracking/phenology_leafshed.py`.

### `src/pipeline/README.md`

Defines the canonical production flow:

```text
orthomosaics (.tif)
  -> Step 0: discover
  -> Step 1: crown detection
  -> Step 2: tracking + consensus
  -> Step 3: phenology
  -> Step 4: HTML viewer
```

All steps share `pipeline_config.json`. The README also documents important operational advice:

- LHC must exclude the bad Dec-9 orthomosaic (`lhc_09-12-25`) because severe misalignment corrupts the tracking graph.
- Use existing crown directories when Detectree2 was already run.
- Use `conf_0p45` as the default base population, `conf_0p65` for alignment anchors.
- Use `pcc_tiled` for alignment by default.

### `00_discover_oms.py`

Purpose: generate a run config.

Important implementation details:

- Finds project root by looking for `output/`, `src/`, and `input/`.
- Discovers `.tif` / `.TIF` orthomosaics.
- Parses dates from stems such as:
  - `lhc_DD-MM-YY`;
  - `sit_DD-MM-YY_dateNotConfirmed`;
  - `sv_spotX_DD-MM-YY`;
  - legacy `odm_orthophotoD_M_YY`.
- Handles placeholder date naming compatibility:
  - `_dateUnknown`;
  - `_dateNotConfirmed`.
- Supports `--exclude-stems` and `--only-stems`.
- Auto-discovers Detectree2 `.pth` models under `input/detectree_models`.
- Supports existing `--crowns-dir` and resolves crown GPKG paths from `run_summary.json` if available.
- Writes `pipeline_config.json` with:
  - `om_stems`;
  - `pairs` of `[gpkg_path, tif_path, stem]`;
  - `tile_width`, `tile_height`, `tile_buffer`;
  - directories for Detectree, tracking, phenology, viewer.

This is the root of reproducibility. Every later step should read paths and parameters from this config instead of re-discovering manually.

### `01_crown_detection.py`

Purpose: Detectree2 multi-threshold crown detection.

Inputs:

- `pipeline_config.json`;
- orthomosaic `.tif` files;
- Detectree2 model `.pth`.

Process:

1. Set thread env vars for Torch/NumPy backends.
2. Set PROJ data directory if pyproj cannot find it.
3. Build Detectree2 config using `setup_cfg(update_model=...)`.
4. For each orthomosaic:
   - tile with Detectree2 `tile_data`;
   - run `predict_on_data`;
   - project predictions to georeferenced GeoJSON using `project_to_geojson`;
   - stitch crown polygons using `stitch_crowns`;
   - simplify geometry;
   - run `clean_crowns` for each confidence threshold.

Outputs:

- one GPKG per OM: `{stem}_multithreshold.gpkg`;
- one layer per confidence threshold:
  - default thresholds are `0.15, 0.20, ..., 0.65`;
  - layer names are `conf_0p15`, `conf_0p20`, etc.;
- metadata JSON with layer counts;
- `run_summary.json`.

Important behaviour:

- `--skip-existing` validates GPKG layer presence and metadata before skipping.
- `_ensure_png_tile_dir` creates a PNG-only tile directory because Detectree2 prediction expects image tiles.
- `clean_bad_geojson_names` removes malformed `*_None.geojson` predictions.

This step implements the multi-threshold idea from the newer presentations: the project no longer treats one confidence threshold as the only crown universe.

### `02_crown_tracking.py`

Purpose: temporal tracking and consensus crown generation.

Inputs:

- multi-threshold GPKGs;
- original orthomosaic rasters;
- config from step 0/1.

Main sequence:

1. Instantiate `TreeTrackingGraph`.
2. Set `file_pairs` and OM IDs from config.
3. Load multi-threshold crown data.
4. Align OMs using `pcc_tiled` by default.
5. Build conditional graph.
6. Generate tracking quality report.
7. Assemble chains.
8. Run multithreshold gap-fill augmentation.
9. Select consensus source chains.
10. Generate medoid consensus crowns.
11. Deduplicate consensus crowns.
12. Export diagnostics and update config.

Important CLI parameters:

- `--base-threshold-tag`: base crown layer for tracking, default `conf_0p45`.
- `--align-threshold-tag`: cleaner crowns for alignment, default `conf_0p65`.
- `--align-method`: `pcc_tiled`, `pcc`, `ecc`, `crowns`.
- `--base-max-dist`: maximum centroid distance considered.
- `--overlap-gate`: threshold for overlap-based case counting.
- `--min-base-similarity`: similarity cutoff before edge selection.
- `--classify-mode`: `balanced` or `strict`.
- `--min-partial-len`: default 5.
- `--min-partial-ratio`: default 0.9.
- `--dedup-iou`: default 0.75.
- gap-fill parameters:
  - `--gapfill-min-threshold-tag conf_0p15`;
  - `--gapfill-max-centroid-dist 25.0`;
  - `--gapfill-min-iou 0.20`;
  - `--gapfill-duplicate-iou 0.70`.

Key outputs:

- `consensus_crowns_complete_all.gpkg`: main cleaned consensus crown product.
- `consensus_crowns_complete_all_raw.gpkg`: before deduplication.
- `consensus_crowns_om1_phenology.geojson`: OM1-space GeoJSON for phenology.
- `tracking_quality_report.txt`.
- `tracking_quality_metrics.json`.
- alignment CSV/plots.
- match-rate plots.
- chain length histograms.
- chain and consensus strip images if enabled.

### `03_phenology_analysis.py`

Purpose: extract per-crown, per-OM image features and compute leaf-shed/phenology labels.

Inputs:

- cleaned consensus crowns from step 2;
- all OM rasters;
- alignment shifts from step 2.

Important behaviour:

- Reuses saved alignment shifts from config, avoiding accidental re-registration.
- Applies inverse alignment when cropping from each raw OM.
- For each consensus crown and each OM, calls `TreeTrackingGraph.extract_patch_for_polygon`.
- Computes patch features through `phenology_leafshed.compute_patch_features`.
- Builds raw feature table.
- Computes deciduous scores and phenophases.
- Emits a canonical `tree_master_geojson.geojson`.

Raw feature columns include:

- `valid_pixel_fraction`;
- `shadow_fraction`;
- `veg_fraction_hsv`;
- `gcc_mean`;
- `rcc_mean`;
- `gray_entropy`;
- `laplacian_var`;
- `is_bad_observation`.

`tree_master_geojson.geojson` contains:

- top-level dataset metadata;
- OM list;
- phenology config;
- one feature per consensus crown;
- nested properties:
  - IDs;
  - tracking summary;
  - classification;
  - temporal summary;
  - observations per OM;
  - assets/crop paths;
  - alternate pixel geometries for viewer enrichment.

### `04_interactive_viz.py`

Purpose: create a standalone Leaflet viewer.

Inputs:

- cleaned consensus crowns;
- orthomosaics;
- crop geometry;
- optional phenology outputs.

Main outputs:

- `index.html`;
- base underlay PNG from selected OM;
- pixel-coordinate crown GeoJSON;
- crop PNGs for every crown and date;
- `manifest.json`;
- optional `phenology_overview.png`;
- enriched `tree_master_geojson.geojson`.

Technical detail:

- The viewer uses `L.CRS.Simple`, not geographic CRS directly.
- Crown geometries are transformed from map coordinates into downsampled image pixel coordinates.
- Aligned geometries are shifted back into the underlay OM raw coordinate frame before rendering.

### `run_pipeline.sh`

Shell orchestrator for steps 0-4.

Important operational details:

- Uses two conda envs by default:
  - `dpm-detectree` for steps 0-1;
  - `dpm-tracking` for steps 2-4.
- Captures `PIPELINE_CONFIG=` from step 0 output.
- Can run arbitrary subsets using `--steps`.
- Supports existing crown outputs via `--crowns-dir`.

## 4. Core Active Logic in `src/flask_app_tracking`

Despite the folder name, this is the core implementation used by `src/pipeline`.

### `tree_tracking.py`

This is the main algorithmic file.

#### Core class: `TreeTrackingGraph`

Important state:

- `file_pairs`: crown GPKG and orthomosaic raster per OM.
- `om_ids`: ordered temporal layers.
- `crowns_gdfs`: GeoDataFrames for each OM.
- `crown_attrs`: cached geometry attributes.
- `crown_images`: optional crop patches.
- `alignment_shifts`: per-OM translation.
- `alignment_transforms`: affine transforms.
- `multithreshold_layers`: available confidence layers per OM.
- `G`: NetworkX directed graph.
- `case_configs`: matching-case threshold presets.

#### Alignment methods

The code supports several alignment modes:

1. `pcc`
   - whole-overlap phase cross correlation.
   - Reads low-resolution grayscale previews.
   - Crops overlapping regions.
   - Uses `skimage.registration.phase_cross_correlation`.
   - Converts row/column shift into map-unit `dx, dy`.

2. `pcc_tiled`
   - robust local version of PCC.
   - Divides overlap into a grid.
   - Rejects low-texture tiles.
   - Rejects high-error phase-correlation estimates.
   - Rejects absurdly large shifts relative to overlap size.
   - Computes median `dx, dy` over inlier tiles.
   - Falls back to full PCC if not enough valid tiles.

3. `ecc`
   - OpenCV Enhanced Correlation Coefficient image registration.
   - Tries Euclidean, then translation.
   - More tolerant to some illumination/rotation changes but slower and less consistently used.

4. `crowns`
   - geometry-only fallback.
   - Uses high-confidence crown centroids.
   - Nearest-neighbour matches between consecutive OMs.
   - Uses robust median shift.

5. `orb_affine`
   - feature-based affine estimation.
   - Uses ORB keypoints and RANSAC.
   - Can represent affine transform beyond translation.
   - Present in code but not documented as the default pipeline method.

Alignment is cumulative through time:

```text
OM1 fixed
OM2 shift = shift(OM1 -> OM2)
OM3 shift = shift(OM1 -> OM2) + shift(OM2 -> OM3)
...
```

The transform is applied to crown geometries, not to the raster files. Later crop extraction inverts the transform so that an aligned consensus polygon can be sampled from each raw orthomosaic.

#### Crown attributes

For each polygon:

- centroid;
- area;
- perimeter;
- compactness = `4*pi*area/perimeter^2`;
- minimum rotated rectangle eccentricity = minor axis / major axis;
- aspect ratio;
- bounds;
- geometry.

These attributes are used in temporal matching.

#### Pairwise features

For candidate crown pair `(prev, curr)`:

- intersection area;
- union area;
- IoU = intersection / union;
- overlap relative to previous crown = intersection / previous area;
- overlap relative to current crown = intersection / current area;
- centroid distance;
- base weighted similarity;
- spatial similarity;
- area similarity;
- shape similarity;
- mean crown radius;
- raw area ratio;
- balanced area ratio = `min(area_prev, area_curr) / max(area_prev, area_curr)`;
- containment flags:
  - previous contains current;
  - current contains previous.

#### Weighted similarity

Default components:

```text
spatial similarity = max(0, 1 - centroid_dist / max_dist)
area similarity    = min(area1, area2) / max(area1, area2)
shape similarity   = average(compactness similarity, eccentricity similarity)
IoU similarity     = polygon IoU
```

Weighted default:

```text
0.4 spatial + 0.2 area + 0.2 shape + 0.2 IoU
```

Case configs override these weights.

#### Match-case classification

Candidate pairs are assigned to cases:

- `containment`: one polygon contains the other.
- `one_to_one`: enough mutual overlap and IoU, with centroid distance below mode-dependent threshold.
- `nearby`: weaker overlap but close centroid.
- `none`: discarded.

Modes:

- `strict`: requires higher overlap, unique overlap counts, and tighter distances.
- `balanced`: default practical middle ground.
- `lite`: looser fallback.

The active pipeline sets `make_strict_aligned_configs()` but `classify_mode` defaults to `balanced`.

#### Case-specific edge scoring

After case classification, the code filters candidates using a `MatchCaseConfig`.

Each case has:

- base similarity weights;
- final scoring weights;
- similarity threshold;
- minimum IoU;
- minimum overlap ratios;
- maximum centroid distance;
- whether multiple edges are allowed;
- max edges per source and target.

Strict aligned config:

- `one_to_one`:
  - stronger IoU weight;
  - min IoU 0.30;
  - centroid max 10 m;
  - threshold 0.40.
- `containment`:
  - min IoU 0.30;
  - min overlaps 0.30;
  - centroid max 15 m.
- `nearby`:
  - centroid and base similarity dominate;
  - min IoU 0.10;
  - centroid max 20 m.

This is why the algorithm is more technical than Hungarian matching: it deliberately preserves plausible split/merge/nearby links while still ranking them.

#### Graph construction

`build_graph_conditional`:

1. Adds every crown as a node `(om_id, crown_index)`.
2. For each consecutive OM pair:
   - enumerates candidate crown pairs;
   - rejects pairs beyond `base_max_dist`;
   - computes features;
   - counts overlap candidates per node;
   - classifies case;
   - optionally trims candidate lists per source/target;
   - selects candidates by case priority and score;
   - adds directed graph edges with full metrics.

Edge attributes include:

- similarity;
- method;
- case;
- overlap ratios;
- IoU;
- centroid distance;
- base/spatial/area/shape similarity.

#### Chain extraction

The graph is converted into chains by greedy successor selection:

- starts are nodes with in-degree 0;
- at each step, pick outgoing edge with maximum similarity;
- unvisited leftover nodes become singleton chains.

Chain categories:

- `full_width_1`: chain length equals number of OMs and no branching.
- `full_branching`: full temporal span but with branch ambiguity.
- `partial_long`: length >= 3 but not full.
- `partial_short`: length 2.
- `singleton`: length 1.

For branching chains, `extract_backbone_mixed` chooses a best path using edge-case quality:

```text
one_to_one > containment > nearby
```

then by similarity.

#### Multi-threshold gap fill

The code builds a cache of lower-threshold layers aligned into the same coordinate frame.

For partial chains:

- if the chain end has no outgoing edge, search the next OM lower-threshold layers;
- if the chain start has no incoming edge, search the previous OM lower-threshold layers;
- candidate must:
  - not duplicate an existing geometry above duplicate-IoU threshold;
  - be close enough by centroid;
  - have enough IoU with adjacent chain geometry.

When accepted:

- the lower-threshold crown is appended as an augmented node;
- `is_augmented = True`;
- gap-fill edge is added with method `gap_fill`.

This is the code implementation of the presentation idea: trees can disappear at a strict confidence layer but still exist in lower confidence layers.

#### Consensus crowns

Current consensus method is medoid.

For a chain of polygons, compute a score for each observed polygon:

```text
score_i = sum_j [
  w_centroid * normalized_centroid_distance(i,j)
  + w_iou * (1 - IoU(i,j))
  + w_area * (1 - area_ratio_similarity(i,j))
]
```

Default weights:

```text
w_centroid = 0.5
w_iou = 0.4
w_area = 0.1
```

The polygon with minimum score is selected. This avoids averaging polygons directly and keeps a real observed crown boundary.

#### Deduplication

`deduplicate_crowns` removes duplicate consensus polygons:

- fixes invalid geometries with `buffer(0)`;
- optionally filters small areas;
- sorts larger-area / longer-chain / higher-similarity crowns first;
- uses spatial index where available;
- drops near-duplicates when IoU exceeds threshold;
- optionally drops contained smaller crowns using a containment buffer.

This matters because branching and partial-chain inclusion can produce redundant consensus geometries.

#### Crop extraction

`extract_patch_for_polygon`:

- receives an aligned consensus polygon;
- gets OM transform;
- inverts alignment transform;
- maps polygon back to raw OM coordinates;
- uses `rasterio.mask.mask`;
- returns RGB array crop.

This is essential: the consensus polygon is stable in aligned space, but pixels must be sampled from each raw orthomosaic.

### `phenology_leafshed.py`

This file implements the current rule-based phenology classifier.

#### Patch QC config

```text
min_valid_pixel_fraction = 0.60
max_shadow_fraction = 0.55
min_laplacian_var = 25.0
min_valid_px = 20
```

An observation is bad if:

- too much NoData/invalid pixel area;
- too much shadow/dark area;
- too low Laplacian variance, indicating blurry or uninformative patch.

#### Vegetation mask config

HSV thresholds:

```text
h_min = 0.18
h_max = 0.48
s_min = 0.15
v_min = 0.12
```

In degrees, hue `0.18-0.48` is roughly green/yellow-green range. The output vegetation fraction is:

```text
vegetated valid pixels / all valid pixels
```

#### Patch features

For RGB patch:

- valid pixel fraction;
- shadow fraction;
- HSV vegetation fraction;
- GCC = `G / (R + G + B)`;
- RCC = `R / (R + G + B)`;
- grayscale entropy;
- Laplacian variance.

The current code does not compute GLCM features, though some presentations/notebooks mention them.

#### Time-series interpolation

Bad observations are set to NaN before scoring. Series are linearly interpolated across OM IDs, with edge filling.

#### Deciduousness score

For every chain:

- `A_veg`: amplitude of interpolated vegetation fraction.
- `A_gcc`: amplitude of interpolated GCC.
- `A_tex`: amplitude of interpolated Laplacian variance.
- `min_veg`: minimum interpolated vegetation fraction.

Global normalizers:

- `A90_veg`: 90th percentile of vegetation amplitude across crowns.
- `A90_gcc`: 90th percentile of GCC amplitude.
- `A90_tex`: 90th percentile of texture amplitude.

Scores:

```text
s_veg_amp = min(1, A_veg / A90_veg)
s_gcc_amp = min(1, A_gcc / A90_gcc)
s_tex     = min(1, A_tex / A90_tex)
s_depth   = max(0, (veg_min_threshold - min_veg) / veg_min_threshold)
```

Final deciduous score:

```text
DS =
  0.35 * s_veg_amp
  + 0.30 * s_depth
  + 0.25 * s_gcc_amp
  + 0.10 * s_tex
```

Default class threshold in the class is `0.85`, but pipeline step 3 default CLI uses `--ds-thresh 0.70`.

#### Phenophase labels

For deciduous crowns:

- min-max normalize interpolated vegetation series;
- `>= phenophase_on` -> `leaf_on`;
- `<= phenophase_off` -> `leaf_off`;
- otherwise `transitioning`.

For non-deciduous crowns:

- state is `stable`.

Event timing:

- `full_leaf_off_om`: trough OM;
- `leaf_off_start_om`: first OM after last leaf-on before trough;
- `leaf_on_return_om`: first leaf-on at/after trough.

#### Non-tree filtering

`apply_non_tree_thresholds` is present for older batch workflows. It flags likely non-tree artifacts based on low GCC/vegetation mean and low amplitudes. The main pipeline step 3 does not appear to apply this filter directly.

### `app.py` and `templates/index.html`

Older Flask UI:

- `/run_tracking`: instantiate `TreeTrackingGraph`, run conditional graph through backward-compatible `process_all_hungarian`.
- `/metrics`: return quality report.
- `/visualize_chain/<om_id>/<crown_id>`: Plotly chain view.
- `/orthomosaic_with_crowns`: Plotly orthomosaic crown overlay.

This is useful historically but not the current standalone viewer path.

## 5. Utility and Script Files

### `src/utility`

- `deepforestrasterio.py`: old DeepForest experiment. Runs pretrained DeepForest over an image folder, applies CLAHE preprocessing, draws bounding boxes and coordinate labels using rasterio geotransform.
- `segmentation.py`: Segment Anything / image segmentation experiment. Processes orthomosaic chunks; meant to isolate tree-like masks but not the current pipeline.
- `xmp_to_exif.py`: copies DJI XMP GPS metadata into EXIF fields for photogrammetry compatibility.
- `om_coordinates.py`: displays/prints raster map coordinates at grid points.
- `om_geographic_coordinates.py`: similar coordinate annotation with more precise labels.
- `view_coordinates.py`: extracts latitude/longitude from XMP metadata.
- `orthomosaic_size.py`: reports pixel dimensions, map resolution, real-world area.
- `run_leafshed_batch.py`: older batch phenology runner for existing tracking outputs; predates the unified `03_phenology_analysis.py`.
- `species_viewer_generator.py`: builds a species-aware viewer from master GeoJSON and viewer manifest; useful after QField/species annotation.
- `dump_notebook_text.py`: safe notebook-to-text dumper, skipping image MIME data.
- `overlap_numbered_crowns.ipynb`: makes a numbered crown overlay on an orthomosaic.

### `src/scripts`

- `run_detectree_other_area.py`: standalone Detectree2 runner for folders outside the normal pipeline. Uses fixed tile/buffer/conf settings and writes crown GPKGs plus overlay PNGs.
- `build_ground_visit_mapping.py`: maps QField ground-visit polygons to consensus viewer crown indices using greedy IoU matching. Writes a JS payload `window.GROUND_VISIT` for viewer integration.
- `01_harmonic_coefficients.py`: Google Earth Engine harmonic coefficient extraction for NDVI time-series phenology.
- `04_365_days_phenology.py`: reconstructs a 365-band daily phenology curve from harmonic coefficients in GEE.

Repo-level `scripts`:

- `extract_gee_embeddings.py`: GEE embedding export support.
- `upload_crowns_to_gcs.py`: cloud upload support for crown assets.
- `gee_embeddings_env.example`: environment variable template.

## 6. Modular Notebook Framework in `src/notebooks/organised/tracking`

This is a cleaner modular implementation of the tracking ideas. It is notebook-first and not currently the main pipeline import target, but it mirrors the production logic.

Files:

- `config.py`: `TrackingConfig` dataclass with thresholds, paths, gap-fill parameters, output dirs.
- `models.py`: dataclasses for match configs, file pairs, run artifacts, summaries, node IDs.
- `state.py`: `TrackingState`, a state container with graph, crown GDFs, attributes, paths, shifts.
- `io.py`: discover SIT-style pairs, list GPKG threshold layers, load layers, read patches.
- `geometry.py`: compute crown attributes and IoU.
- `alignment.py`: phase-correlation cumulative shift estimation.
- `matching.py`: weighted similarity, pair metrics, case classification, candidate scoring/selection, threshold presets.
- `cases.py`: exports strict/relaxed case configs.
- `graph_build.py`: graph construction from state and matching configs.
- `chains.py`: greedy chain extraction, categorisation, backbone extraction, partial-chain selection.
- `augmentation.py`: multi-threshold gap-fill functions.
- `multithreshold.py`: load multithreshold crown stores and build lower-threshold cache.
- `consensus.py`: medoid, intersection-core, union-shrink consensus methods.
- `diagnostics.py`: quality metrics, text reports, diagnostic figures.
- `viz.py`: chain panels, consensus panels, spatial maps.
- `pipeline.py`: notebook entry point wiring all modules together.
- `organised.ipynb`: notebook wrapper around this modular pipeline.

This folder is probably the best future direction if the goal is maintainability. The active pipeline still relies on the monolithic `TreeTrackingGraph`.

## 7. Satellite and Species Classifier Experiments

These files answer the downstream question: can coarser satellite data recover tree traits measured/validated at drone crown level?

### `src/notebooks/satellite/drone_phenology_rf_package`

This package prepares labels and runs local/GEE/STAC classifier experiments.

#### Data model

Main prepared outputs:

- `data/iitd_sv_crowns_master_wgs84.geojson`
- `data/crown_label_table.csv`

Label properties:

- `label_esd`: evergreen/semi-evergreen/deciduous multiclass.
- `label_deciduous`: deciduous vs rest.
- `label_acacia`: Acacia vs non-Acacia.
- `label_yellow_strict`: strict yellow-showy flowering.
- `label_yellow_broad`: broader yellow flowering.
- `label_red_showy`: red-showy flowering.
- `label_showy_flower`: any showy flowering.

Current label counts from outputs:

```text
label_esd:          75 evergreen, 152 semi-evergreen, 115 deciduous, 2870 ignored
label_deciduous:   227 non-deciduous, 115 deciduous, 2870 ignored
label_acacia:      328 non-acacia, 63 acacia, 2821 ignored
label_yellow:      40 strict positives / 100 broad positives
label_red_showy:   12 positives only, underpowered
label_showy:       63 positives
```

Large ignored counts happen because many crowns are unlabelled or ambiguous.

#### `python/00_generate_configs_from_traits.py`

Builds label configuration JSONs from a species trait CSV.

This encodes ecological knowledge into machine-readable label definitions:

- positive species;
- negative species;
- ignored ambiguous species;
- multiclass mappings.

#### `python/01_prepare_crowns.py`

Combines IITD and Sanjay Van master GeoJSONs into one WGS84 feature collection.

Important details:

- Input areas include `A1`-`A5`, `MITTAL`, `SIT`, `SV_S1`-`SV_S4`.
- Transforms geometries from EPSG:32643 to EPSG:4326.
- Cleans species aliases.
- Treats ambiguous/mixed labels conservatively.
- Applies label configs to create classifier labels.
- Writes species counts and label summary.

This is the bridge from QField/manual validation to satellite ML.

#### `python/01b_relabel.py`

Re-applies updated label configs to existing GeoJSON/CSV outputs without redoing geometry flattening.

#### `python/02_local_rf_from_gee_export.py`

Local baseline evaluator for exported GEE/Sentinel feature tables.

Core ideas:

- infer feature columns;
- make random / leave-area-out / leave-species-out splits;
- train/evaluate Random Forest;
- output confusion matrix, accuracy, balanced accuracy, macro-F1, feature importances.

#### `python/03_extract_sentinel2_stac_features.py`

Local Microsoft Planetary Computer STAC extractor for Sentinel-2 L2A.

Feature extraction:

- seasons:
  - winter;
  - premonsoon;
  - monsoon;
  - postmonsoon.
- optical bands:
  - B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
- indices:
  - NDVI;
  - GNDVI;
  - NDRE;
  - NDMI;
  - NBR;
  - EVI;
  - blue/green/red chromatic ratios;
  - yellow proxy.
- cloud/shadow masking uses SCL invalid classes.
- geometry modes:
  - crown polygon;
  - centroid point;
  - centroid buffer;
  - fractional crown.

For each season, multiple STAC items are queried, per-crown statistics are computed, and medians across items become features. It also adds amplitude features across seasons.

This directly addresses the drone-to-satellite scaling goal.

#### `python/04_binary_rf_threshold_sweep.py`

Binary RF threshold tuning. Chooses probability thresholds with cross-validation rather than default 0.5.

#### `python/05_make_weekly_report_assets.py`

Generates report assets:

- label count plots;
- species heatmaps;
- metrics plots;
- confusion matrices;
- feature importance plots;
- satellite/crown overlay figures;
- markdown report.

#### `python/06_model_sweep.py`

Runs multiple tabular classifiers:

- Random Forest balanced;
- deeper Random Forest;
- ExtraTrees;
- ExtraTrees + SelectKBest;
- logistic regression;
- RBF SVM;
- histogram gradient boosting.

Metrics:

- accuracy;
- balanced accuracy;
- macro-F1;
- confusion matrix.

For binary labels, it also performs threshold-CV and writes both default and tuned decisions.

#### `python/07_extract_sentinel1_stac_features.py`

Sentinel-1 SAR feature extraction. Computes radar features from VV/VH over seasonal/date windows. Useful because SAR can provide structure/moisture signals independent of clouds and optical colour.

#### `python/08_consolidate_round3_report.py`

Aggregates model sweep outputs, label summaries, feature summaries, and writes a consolidated report.

#### `python/09_signal_qc_plots.py`

Plots satellite signal traces, class summaries, area summaries, missingness, and spectral signatures.

#### `python/10_build_multiyear_s2_table.py`

Combines yearly Sentinel-2 tables into multi-year feature tables.

#### `python/11_add_temporal_features.py`

Adds derived multi-year temporal features such as amplitudes and cross-season/year differences.

#### `python/12_run_model_suite.py`

Orchestrates `06_model_sweep.py` over default labels and random/leave-area-out splits. It automatically chooses holdout areas that contain both classes.

#### `python/13_extract_janmay_stac_features.py`

Variant of Sentinel-2 extraction focused on January-May windows, likely for dry-season / flowering / pre-monsoon experiments.

#### `python/14_extract_embedding_features.py`

Extracts DINOv2 visual embeddings from Sentinel-2 RGB patches. This is the embedding lane mentioned in the companion paper context.

Concept:

- download Sentinel-2 RGB patches;
- crop around crown/buffer geometry;
- run a pretrained DINOv2 encoder;
- write embedding dimensions as tabular feature columns.

#### `python/15_pixel_level_classifier.py`

Pixel-level classification experiment. Simulates/uses pixels as training units, then aggregates pixel predictions to crown-level predictions.

#### `python/16_acacia_clustering_experiment.py`

Acacia clustering / crown vs pixel experiment. Evaluates whether Acacia detection improves using pixel-level or spatially clustered approaches.

#### `python/make_presentation_figures.py`

Creates presentation figures for the satellite/classifier story.

#### `outputs/recommended_experiments.csv`

Defines experiment priority:

1. ESD random split.
2. ESD leave-area-out.
3. Deciduous binary.
4. Acacia, especially Sanjay Van holdouts.
5. Yellow/showy flowering.
6. Red-showy exploratory only.

The split philosophy is important: random split is not enough; leave-area-out and leave-species-out are stronger tests of generalisation.

### `src/notebooks/satellite/embeddings`

Embedding-specific wrapper folder.

- `README.md`: explains DINO/GEE embedding lane.
- `run_embedding_experiment.py`: runs DINO extraction plus classifier suite.
- `run_gee_embedding_suite.py`: normalizes downloaded GEE embedding CSV column names and runs model suite, optionally training best classifiers.
- `train_best_classifiers.py`: trains full-data deployable models from best random-split rows.
- `summarize_results.py`: prints best rows from model sweep CSVs.

This is the most direct code overlap with the paper phrase "per crown feature embedding processing".

## 8. Notebook Hierarchy and Experiment Chronology

### Detection notebooks

- `src/notebook_archive/detectree.ipynb`: earliest Detectree2 install/model/tile/predict workflow.
- `detectree_new.ipynb`: crown detection class abstraction.
- `detectree_parth.ipynb`: adaptive/multi-threshold storage idea.
- `detectree_parth_10Mar26.ipynb`: multi-threshold Detectree2 workflow.
- `src/notebooks/detectree_parth_10Mar26.ipynb`: current SIT version.
- `src/notebooks/detectree_parth_LHC.ipynb`: current LHC version.

These led directly to `src/pipeline/01_crown_detection.py`.

### Early metrics / ground truth notebooks

- `metrics.ipynb`: Detectree2 detection metrics helper.
- `metrics ground truth.ipynb`: CVAT ground truth conversion and IoU matching.
- `ground truth processing.ipynb`: same crown detection evaluation path.

These support the May 2025 metrics.

### Hungarian and early tracking notebooks

- `hungarian.ipynb`: baseline one-to-one polygon matching.
- `crown_tracking.ipynb`: early `TreeTrackingGraph`.
- `tree_tracking_graph_demo.ipynb`: synthetic examples of graph cases.
- `crown_tracking_real_data_demo.ipynb`: real-data alignment/tracking demo.

These explain why the project moved away from Hungarian-only matching.

### Graph tracking evolution notebooks

Dated files show iterative development:

- `crown_tracking_4Nov.ipynb`
- `crown_tracking_10Nov.ipynb`
- `crown_tracking_15Oct.ipynb`
- `crown_tracking_6Jan26.ipynb`
- `crown_tracking_13Jan26 copy.ipynb`
- `crown_tracking_19Jan26.ipynb`
- `crown_tracking_21Jan26.ipynb`
- `crown_tracking_3Feb26.ipynb`
- `crown_tracking_4Feb26.ipynb`
- `crown_tracking_24Feb26.ipynb`

Main ideas developed here:

- alignment shifts;
- conditional matching;
- graph quality reports;
- chain visualisations;
- strict vs relaxed presets;
- chain length distributions;
- partial chains;
- quality metrics.

### Rotational shuffle and virtual-edge experiments

- `corwn_tracking shuffles.ipynb`: rotational shuffles, graph aggregation, virtual edges.
- The idea: test robustness by changing sequence order or cyclic paths; persistent edges under shuffles are more reliable.

This appears in reports/presentations, but is not central in the current production pipeline.

### Multithreshold + consensus notebooks

- `crown_tracking_10Mar26.ipynb`: pipeline-style multithreshold and consensus.
- `crown_tracking_10Mar26_lhc.ipynb`: LHC version.
- `crown_tracking_31Mar26.ipynb`: includes phenology QA.

These led directly to the current `src/pipeline` scripts.

### Phenology signal notebooks

- `Phenology_signals_10Mar26.ipynb`
- `Phenology_signals_18Feb26.ipynb`
- `Phenology_signals_25Mar26.ipynb`
- `Phenology_signals_LHC17Mar26.ipynb`

They combine graph tracking, consensus crowns, image crops, phenology features, ground-truth comparison, clustering, and visual audits.

These are the conceptual parent of `03_phenology_analysis.py` and `phenology_leafshed.py`.

### Leaf state notebook

- `LeavesFreshMatureOld.ipynb`: SIT-only fresh/mature/old leaf-state classifier.

It extracts crop PNG features, normalises by date, computes rule-based state scores, smooths per tree, and provides visual audits. This is distinct from the current binary deciduous/phenophase logic but could become a future richer phenology classifier.

### Alignment notebook

- `lhc_alignment copy.ipynb`: compares PCC and PCC-tiled alignment for LHC.

This directly informs the default `pcc_tiled` alignment in the current pipeline.

### Visualisation notebooks

- `overlay.ipynb`: simple raster + crown overlay.
- `visualisations.ipynb`: side-by-side crown detection comparison across OMs.
- `visualisations_presentation.ipynb`: slide figure generation.
- `output/end_sem_2501_images/viz.ipynb`: report figures for graph tracking and similarity.

### Satellite notebooks

- `sat_data.ipynb`: initial Sentinel feature exploration.
- `sat_data6May26.ipynb`: refined satellite phenology extraction.
- `sat_data_sentinel1.ipynb`: Sentinel-1 SAR feature extraction.

These are exploratory ancestors of the more formal `drone_phenology_rf_package`.

## 9. Integration with the Companion Paper Context

The pasted paper excerpt describes a workflow with these components:

1. Site selection using stratification.
2. UAV mission planning.
3. Photogrammetry into georeferenced orthomosaic.
4. Detectree2 crown delineation.
5. Crown-level feature embedding.
6. Manual/QField ground truth.
7. Google Earth species visualisation.
8. DTM + DSM canopy height model.
9. Clustering and spatial visualisation.

Mapping to this repository:

- Site selection:
  - implicit in IITD/Sanjay Van site organisation and docs;
  - not formalised in code.
- Mission planning:
  - documented in thesis PDFs and `misc/docs`;
  - not in `src`.
- Photogrammetry:
  - WebODM process documented in reports/docs;
  - outputs consumed as `.tif` orthomosaics.
- Detectree2:
  - active in `01_crown_detection.py`, detection notebooks, `run_detectree_other_area.py`.
- Crown polygon coordinate translation:
  - Detectree2 tile predictions are projected to GeoJSON;
  - `project_to_geojson`, rasterio transforms, CRS handling;
  - crown polygons stored in GPKG/GeoJSON.
- QField validation:
  - supported by `build_ground_visit_mapping.py`;
  - species labels consumed in `01_prepare_crowns.py`.
- Google Earth / GEE:
  - `drone_phenology_rf_package`;
  - GEE asset upload path;
  - GEE embeddings and satellite feature experiments.
- Feature embeddings:
  - `14_extract_embedding_features.py`;
  - `src/notebooks/satellite/embeddings`.
- Crown clustering:
  - `16_acacia_clustering_experiment.py`;
  - phenology clustering notebooks.
- DTM/DSM height:
  - mentioned in paper context;
  - not active in current pipeline;
  - could be added after WebODM by reading DSM and DTM rasters and computing `CHM = DSM - DTM`, then zonal stats per crown.

## 10. Key Technical Distinctions

### Detection vs tracking

Detectree2 answers:

```text
Where are likely crowns in one orthomosaic?
```

Tracking answers:

```text
Which detections across dates are the same biological tree?
```

The project's main originality is in tracking and consensus construction, not just detection.

### Crown instance vs tree identity

A Detectree2 polygon is a date-specific crown instance. A tree identity is a chain or graph path. A consensus crown is a stable spatial sampling geometry derived from that chain.

### Full chains vs partial chains

Full chains are highest confidence but low coverage. Partial chains increase coverage but must be filtered:

- length threshold;
- one-to-one edge ratio;
- average similarity;
- visual/diagnostic checks.

### Real phenology vs detection failure

A missing crown can mean:

- true canopy disappearance / severe leaf-off;
- detector confidence drop;
- shadow/lighting issue;
- orthomosaic distortion;
- over/under segmentation.

Multi-threshold gap fill and phenology QC are attempts to separate these cases.

### Drone phenology vs satellite classification

Drone pipeline:

- high spatial detail;
- individual trees visible;
- fine crown geometry;
- repeated local sites.

Satellite pipeline:

- coarse pixels;
- broader spatial/temporal coverage;
- crown-level extraction requires buffers/fractional geometry;
- supervised labels come from drone/QField/species data.

The project goal is to use drone outputs as high-quality ground truth for satellite-scale inference.

## 11. Current Main Outputs and Their Meaning

From a successful pipeline run:

- `pipeline_config.json`: run provenance and shared paths.
- `01_detectree/crowns_multithreshold/*.gpkg`: per-OM crown detections at multiple confidence thresholds.
- `02_tracking/consensus_crowns_complete_all.gpkg`: final deduplicated tree-level consensus crowns.
- `02_tracking/tracking_quality_metrics.json`: graph/chain quality diagnostics.
- `02_tracking/diagnostics/alignment_shifts.csv`: per-date spatial registration.
- `03_phenology/phenology_features_raw.csv`: one row per crown per OM.
- `03_phenology/leafshed_tree_scores.csv`: deciduousness score per crown.
- `03_phenology/leafshed_phenophase_by_om.csv`: leaf-on/off/transition labels per crown per OM.
- `03_phenology/tree_master_geojson.geojson`: canonical tree-level output.
- `04_viewer/index.html`: interactive crown time-series viewer.

## 12. Current Technical Limitations

1. `TreeTrackingGraph` is monolithic; the organised tracking framework is cleaner but not the active import path.
2. Alignment is mostly translation-based. ORB affine exists but is not the default.
3. Virtual nodes are conceptually described, but active pipeline currently uses lower-threshold gap-fill augmented real detections more than explicit synthetic virtual nodes.
4. Phenology classifier is rule/scoring based, not supervised against field leaf states.
5. GLCM features appear in notebooks/presentations, but current active `phenology_leafshed.py` uses entropy and Laplacian variance.
6. DTM/DSM canopy-height modelling is not yet integrated into the active pipeline.
7. QField/species data integration exists mostly in scripts/satellite package, not as a first-class pipeline step.
8. Random split satellite metrics can be overoptimistic; leave-area/species-out results are more scientifically meaningful.
9. Red-flower classification is underpowered because positives are very few.
10. Crown-level satellite extraction has a scale mismatch; centroid buffers and fractional crown weights are attempts to handle Sentinel pixel size.

## 13. Recommended Mental Model

The repository should be understood as four connected experiment tracks:

### Track A: UAV acquisition and orthomosaic generation

Mission planning, repeated flights, WebODM, georeferenced GeoTIFFs, optional DSM/DTM.

### Track B: Crown detection

Detectree2 on tiled orthomosaics, multi-threshold GPKGs, cleaning and validation.

### Track C: Crown tracking and drone phenology

Alignment, graph matching, chain extraction, gap fill, consensus crowns, crown crops, phenology scores, viewer.

### Track D: Field/satellite scale-up

QField validation, species labels, GEE/STAC/Sentinel/DINO features, supervised classifier sweeps, reports.

The current operational center of gravity is Track C, implemented by:

```text
src/pipeline/*.py
src/flask_app_tracking/tree_tracking.py
src/flask_app_tracking/phenology_leafshed.py
```

The downstream research frontier is Track D, implemented by:

```text
src/notebooks/satellite/drone_phenology_rf_package
src/notebooks/satellite/embeddings
scripts/extract_gee_embeddings.py
```
