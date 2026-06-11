# Paper Outline: Drone-Based Individual Tree Phenology Monitoring

Working framing:

> Repeated UAV RGB orthomosaics can support individual-tree phenology monitoring if per-date crown detections are converted into temporally stable tree identities through explicit alignment, graph-based temporal association, gap filling, and consensus crown construction.

This outline treats the **tree-identity-preserving temporal pipeline** as the central contribution. Satellite/species experiments are included as a proper part of the paper, but framed as **downstream ecological inference enabled by the drone-derived crown objects**, rather than as an independent main contribution.

---

## 1. Candidate Titles

### Recommended Main Title

**Tracking Individual Tree Phenology from Repeated UAV RGB Orthomosaics Using Graph-Based Crown Association and Cross-Scale Ecological Inference**

User selected this as the working title.

Why this works:

- "Tracking Individual Tree Phenology" immediately states the scientific object.
- "Repeated UAV RGB Orthomosaics" makes the data modality explicit.
- "Graph-Based Crown Association" names the central method.
- "Cross-Scale Ecological Inference" leaves room for satellite/species experiments without making them sound like the whole paper.

### Shorter Alternative

**From Crown Detection to Tree Identity: A UAV RGB Pipeline for Individual Tree Phenology and Cross-Scale Trait Mapping**

Why this works:

- Strong conceptual transition: detection is not the final goal; identity is.
- "Trait mapping" gives a defensible place for species/deciduous/flowering/satellite experiments.

### Thesis-Style Alternative

**Drone-Based Individual Tree Phenology Monitoring Through Temporal Crown Tracking, Consensus Geometry, and Satellite-Scale Inference**

Why this works:

- Slightly broader and more thesis-friendly.
- Makes all three major parts visible:
  1. drone tree tracking;
  2. consensus crowns and phenology;
  3. satellite/species downstream inference.

---

## 2. One-Sentence Thesis Statement

This work presents an end-to-end UAV RGB pipeline that transforms repeated orthomosaics into temporally stable individual-tree crown identities and crown-level phenology trajectories using multi-threshold crown delineation, residual orthomosaic alignment, graph-based temporal crown association, gap filling, consensus crown generation, and downstream species/trait inference using field labels and satellite-scale features.

---

## 3. Core Claims to Defend

These are the main paper claims. The writing should keep returning to them.

### Claim 1: Crown detection alone is not enough for phenology

Per-date crown polygons are unstable because of photogrammetric drift, illumination differences, seasonal canopy change, partial leaf shedding, shadows, and segmentation split/merge errors. Therefore, the central problem is not only detecting crowns, but preserving biological tree identity through time.

### Claim 2: Alignment is necessary before temporal crown association

Even georeferenced orthomosaics from repeated UAV flights show residual spatial offsets. A lightweight image-registration step, especially phase-cross-correlation-based translation alignment, improves spatial correspondence before matching crown polygons.

### Claim 3: Graph-based temporal association is more appropriate than strict one-to-one matching

Hungarian-style one-to-one matching is brittle when crowns split, merge, disappear, or reappear. A graph representation can preserve plausible temporal continuity under ambiguity and allows later extraction of reliable full chains and filtered partial chains.

### Claim 4: Consensus crowns are the correct unit for phenology sampling

Sampling from noisy per-date polygons produces inconsistent regions of interest. A consensus crown per tracked tree provides a stable spatial object for comparable color, vegetation, texture, and satellite feature extraction across dates.

### Claim 5: Drone-derived crown identities enable downstream ecological inference

Once stable individual-tree crown objects exist, they can support:

- RGB phenology time series;
- deciduous/evergreen and leaf-on/leaf-off scoring;
- field/QField species linkage;
- crown-level species/trait classifiers;
- Sentinel-2/Sentinel-1 feature extraction;
- DINO or visual embedding experiments;
- cross-scale analysis between high-resolution drone observations and coarser satellite signals.

This is how satellite/species experiments should enter the paper: not as side experiments, but as evidence that the crown identity layer becomes a reusable ecological data product.

---

## 4. Abstract Drafts

### Abstract Draft A: Paper-Style

Repeated UAV RGB surveys provide a practical route to monitoring tree phenology at individual-crown resolution, but converting orthomosaic time series into biological tree trajectories remains challenging. Per-date crown detections are affected by georeferencing drift, photogrammetric artifacts, changing illumination, seasonal canopy structure, and segmentation split-merge errors. We present an end-to-end pipeline that converts repeated UAV orthomosaics into temporally stable individual-tree crown identities and crown-level phenology trajectories. The workflow combines Detectree2-based multi-threshold crown delineation, residual orthomosaic alignment using phase-cross-correlation-based image registration, graph-based temporal association of crown polygons, gap filling using lower-confidence detections, and consensus crown construction for stable repeated sampling. For each tracked tree, the system extracts crown-level RGB and texture features across dates and derives interpretable phenology indicators such as green chromatic coordinate, red chromatic coordinate, vegetation fraction, shadow fraction, and leaf-on/leaf-off state. Field and QField annotations are linked to the resulting crown objects, enabling species-aware analysis. We further evaluate how these drone-derived crown identities support downstream ecological inference using Sentinel-2, Sentinel-1, and visual embedding features for species and phenological trait classification. The resulting framework reframes UAV crown delineation as a temporal identity problem and provides a practical bridge between high-resolution drone monitoring, field validation, and satellite-scale vegetation analysis.

### Abstract Draft B: More Conservative

Repeated UAV RGB orthomosaics can capture fine-scale seasonal changes in tree canopies, but individual-tree phenology monitoring requires stable tree identities across dates. This is difficult because crown detections vary with illumination, leaf state, orthomosaic reconstruction quality, residual spatial misalignment, and segmentation ambiguity. This thesis develops a UAV-based workflow for transforming repeated orthomosaics into individual-tree phenology records. The workflow detects crown polygons using Detectree2, stores detections across multiple confidence thresholds, aligns orthomosaics into a common frame, constructs a temporal graph of plausible crown correspondences, fills gaps using lower-confidence candidates, and generates consensus crown geometries for repeated feature extraction. Crown-level RGB and texture metrics are then used to characterize phenological change, including canopy greenness, reddishness, vegetation fraction, shadow contamination, and deciduousness. The tracked crown layer is also linked with field species labels and satellite-derived features to examine how drone-scale observations can support broader species and trait mapping. The work demonstrates that the key step in operational UAV phenology monitoring is the conversion of unstable per-date detections into persistent crown identities that can be sampled, validated, and analyzed across scales.

### Abstract Draft C: Thesis-Plus-Paper Hybrid

Monitoring tree phenology at the level of individual crowns can improve ecological interpretation of seasonal canopy change and provide high-resolution ground truth for satellite models. This thesis presents a drone-based system for individual tree phenology monitoring from repeated UAV RGB orthomosaics. The central challenge is temporal identity: crown polygons detected independently on each date do not directly correspond to stable biological trees because of orthomosaic drift, photogrammetric artifacts, changing illumination, leaf shedding, and segmentation split-merge errors. To address this, the system combines multi-threshold Detectree2 crown detection, phase-cross-correlation-based alignment, graph-based temporal crown association, missing-detection recovery, and medoid consensus crown generation. These stable crown objects are used to extract per-date crown crops and RGB/texture phenology features. The resulting trajectories support rule-based deciduousness and leaf-state scoring, manual validation, field species linkage, and interactive visualization. A downstream set of satellite and species experiments evaluates whether the drone-derived crown identities can act as training and validation units for Sentinel-2, Sentinel-1, and visual embedding classifiers. Together, the work shows how repeated drone surveys can be transformed from independent orthomosaics into a temporally structured, crown-level ecological monitoring dataset.

Recommended starting abstract: **Draft C** for thesis, then tighten later into **Draft A** for paper submission.

---

## 5. Proposed Paper Structure

## 1. Introduction

### 1.1 Motivation: Tree-Level Phenology Needs Individual Identity

Write:

- Phenology is a sensitive indicator of climate, water stress, deciduousness, flowering, and ecosystem change.
- Satellite phenology provides broad coverage but often mixes multiple trees/species within one pixel.
- Field phenology is accurate but difficult to scale.
- UAV imagery fills a scale gap: high spatial resolution, repeatable surveys, and crown-level observation.

Key transition:

> However, repeated high-resolution imagery does not automatically produce tree-level phenology records. The same tree must first be identified consistently across survey dates.

### 1.2 Technical Problem

Explain why independent crown detection per date is insufficient:

- drone GPS and orthomosaic georeferencing drift;
- photogrammetric distortion;
- illumination and shadow variation;
- seasonal canopy structure change;
- crown boundary instability;
- segmentation confidence variation;
- split and merge errors;
- missing detections in leaf-off or shaded states.

Then state that this makes phenology monitoring a temporal object identity problem.

### 1.3 Research Gap

Frame the gap carefully:

- Many works focus on individual tree crown delineation from UAV imagery.
- Many works extract vegetation indices or phenology from UAV images.
- Fewer works address the operational problem of preserving individual tree identities across a long sequence of repeated UAV orthomosaics under imperfect segmentation and residual misalignment.
- Even fewer connect the resulting crown identities to field labels and satellite-scale feature experiments.

Avoid saying "no one has done this" unless we verify literature very carefully. Say "this remains an operational bottleneck" or "less attention has been given to".

### 1.4 Contributions

Use a numbered contribution list.

Suggested contribution wording:

1. We develop an end-to-end UAV RGB workflow that converts repeated orthomosaics into temporally stable individual-tree crown identities.
2. We introduce a practical multi-threshold crown-detection and graph-association strategy for handling confidence variation, missing detections, and split-merge crown segmentation errors.
3. We incorporate residual orthomosaic alignment using phase-cross-correlation-based translation estimation to improve crown correspondence across dates.
4. We construct consensus crown geometries from temporal chains, enabling stable repeated sampling of crown-level RGB and texture phenology features.
5. We connect drone-derived crown identities with field species labels and satellite/embedding features to evaluate downstream species and phenological trait inference across spatial scales.

### 1.5 Paper Roadmap

One short paragraph:

- data and study sites;
- crown detection and tracking pipeline;
- phenology extraction;
- satellite/species inference;
- evaluation and discussion.

---

## 2. Related Work

This section should not be a generic literature dump. It should build the argument that the project lives at the intersection of crown delineation, multi-temporal object tracking, phenology, and cross-scale remote sensing.

### 2.1 UAV-Based Forest and Urban Tree Monitoring

Write about:

- UAVs as intermediate scale between field plots and satellites.
- Use of UAV RGB imagery for high-resolution canopy monitoring.
- Orthomosaic generation and repeat-flight workflows.
- Advantages: crown-level detail, flexible revisit, low cost.
- Limitations: local coverage, illumination sensitivity, photogrammetric artifacts.

Connect to project:

> This study uses UAV RGB orthomosaics as the primary repeated observation layer because they preserve individual crown structure while remaining feasible for repeated site-level monitoring.

### 2.2 Individual Tree Crown Detection and Delineation

Write about:

- Bounding-box methods such as DeepForest.
- Instance segmentation methods such as Mask R-CNN/Detectree2.
- Why polygon masks are more suitable than boxes for phenology sampling.
- Tiled inference for large orthomosaics.
- Confidence thresholds, overlap removal, polygon simplification.

Use our project trajectory:

- Early DeepForest baseline had too many false positives and boxes were too coarse.
- Detectree2 became the main detector because it outputs polygon crowns.
- But detection quality alone was not enough: low recall, false positives, and unstable date-to-date detections motivated temporal tracking.

### 2.3 Multi-Temporal Orthomosaic Alignment

Write about:

- Orthomosaics are georeferenced but residual offsets remain.
- Multi-temporal comparison requires co-registration.
- Registration can use image features, phase correlation, mutual information, control points, or affine/local warping.
- Project uses a pragmatic translation-based alignment because repeated flights over same site should mostly differ by residual shifts, and because crown tracking benefits from even coarse alignment.

Important nuance:

> Translation-only alignment is not a complete correction for all photogrammetric distortions. It is a residual alignment step used to reduce systematic offsets before crown association.

### 2.4 Multi-Object Tracking and Temporal Association in Remote Sensing

Write about:

- Tracking-by-detection: detect objects independently, then associate across time.
- Simple nearest-neighbor or Hungarian matching works when detections are stable and one-to-one.
- Ecological objects violate this assumption: trees can be partly missing, crowns overlap, segmentation can split/merge.
- Graph representations preserve multiple plausible links and allow chain extraction after candidate matching.

Connect:

> The proposed crown graph treats each detected crown polygon as a dated observation and constructs candidate temporal links using spatial overlap, distance, area consistency, and shape features.

### 2.5 UAV Phenology Metrics from RGB Imagery

Write about:

- RGB phenology indices: GCC, RCC, excess green, vegetation fraction.
- Phenocam-style logic adapted to crown crops.
- Challenges: shadows, illumination, exposure, view geometry, background pixels, mixed crowns.
- Importance of masking and stable regions of interest.

Connect:

> Consensus crowns make RGB phenology extraction more comparable because each tree is sampled from a stable spatial object across the sequence.

### 2.6 Species, Traits, and Cross-Scale Satellite Integration

This is where satellite/species becomes a proper section.

Write about:

- Crown-level field labels as a bridge between high-resolution UAV observations and satellite pixels.
- Sentinel-2 spectral/time-series features for vegetation phenology and species/trait mapping.
- Sentinel-1 backscatter as complementary structure/moisture information.
- Visual foundation/embedding features as an exploratory representation for crown appearance.
- Limitations from scale mismatch: one satellite pixel may contain many crowns/species.

Connect:

> In this work, satellite and embedding experiments are not treated as independent replacements for UAV monitoring. Instead, they test whether the drone-derived crown identity layer can serve as labeled ecological units for cross-scale trait inference.

---

## 3. Study Areas and Data

### 3.1 Study Sites

Current decision:

- The primary demonstration area is **IIT Delhi campus**.
- The main tracking/phenology sites are **LHC** and **SIT**.
- Current complete-data inventory reported by user:
  - LHC: 13 orthomosaics/dates available now.
  - SIT: 19 orthomosaics/dates available now.
- The currently inspected output folders are older partial runs:
  - LHC partial pipeline outputs use 8 or 9 orthomosaics.
  - SIT partial pipeline outputs use 14 orthomosaics.
- Final result tables should be regenerated from the latest 13-OM LHC and 19-OM SIT runs once available.

Write:

- LHC and SIT are treated as the primary repeated-UAV monitoring sites for demonstrating temporal crown tracking and crown-level phenology extraction.
- Other sites, if mentioned, should be framed as broader project context or auxiliary data sources unless their final outputs are included in the main results.
- The paper should clearly separate **available full dataset size** from **currently reported partial-run results**.

Need from user later:

- exact expanded names/descriptions for LHC and SIT;
- final date range for the 13 LHC and 19 SIT orthomosaics;
- whether SAC or Sanjay Van should appear only in background or in a secondary experiment.

### 3.2 UAV Data Acquisition

Write:

- DJI Mini 4 Pro.
- Mission planning using Map Pilot Pro.
- Repeated flights every roughly two weeks.
- Altitude approximately 50-80 m, often around 80-81 m.
- Speed approximately 3 m/s.
- Forward and side overlap approximately 80%.
- Near-nadir imagery.
- Reused flight paths for temporal consistency.

Need from user:

- exact flight count and dates;
- whether all sites used the same altitude/overlap;
- whether the 50-80 m range should be narrowed in final text.

### 3.3 Photogrammetric Processing

Write:

- Raw UAV images processed in WebODM.
- Outputs include georeferenced RGB orthomosaics.
- Orthomosaics are exported as GeoTIFFs.
- CRS and geospatial metadata are preserved for crown polygons and satellite linkage.

Important caution:

Do **not** include DSM/DTM/CHM, canopy-height modeling, or tree-height estimation as a claim of this work unless new results are added. That material came from overlapping companion-project context, not the current thesis contribution.

### 3.4 Field and Manual Labels

Write:

- Field-verified labels and QField validation were used to attach species/trait labels to crown objects.
- Labels include species, deciduous/evergreen behavior, possibly flowering traits or showy flower categories depending on data table.
- Google Earth/KML/KMZ visualizations supported inspection and communication.

Need from user:

- final label schema;
- number of labeled crowns;
- details of field verification protocol and who performed it;
- how QField annotations were joined back to crown geometries.

### 3.5 Satellite and Embedding Data

Write:

- Satellite/species work is still in progress and should currently be represented by headings and planned result slots.
- The most trusted current direction is the **Google Earth Engine embeddings** experiment.
- Sentinel-2 seasonal and harmonic features should remain as the main interpretable satellite baseline, while Google Earth Engine embeddings should be treated as the strongest current feature representation.
- Other Sentinel-1, random-forest, pixel-level, and visual-embedding experiments can remain in the method/result hierarchy, but detailed claims should wait until the final results are selected.
- Features will be linked to field-verified crown labels to test species/trait inference.
- The weekly meetings show that crown size versus satellite pixel size is not a side detail. It should be reported explicitly because many crowns are sub-pixel at Sentinel-2 resolution.

Need from user:

- final Google Earth Engine embedding workflow details;
- final classifier tasks and validation strategy;
- which non-GEE satellite experiments should remain as secondary comparisons.
- final crown-area filtering policy, if any, for satellite classifier tables.

---

## 4. Method

This is the technical heart. It should be written as a reproducible pipeline.

### 4.1 Overview

Start with a workflow figure.

Suggested figure:

```text
UAV images
  -> WebODM orthomosaics
  -> Detectree2 multi-threshold crown stores
  -> residual orthomosaic alignment
  -> graph-based crown association
  -> chain extraction and gap filling
  -> consensus crown generation
  -> crown crops and RGB phenology features
  -> field/species/satellite feature linkage
  -> phenology and trait inference
```

Write one paragraph explaining each block at high level.

### 4.2 Crown Detection with Detectree2

Include:

- Orthomosaics are tiled.
- Buffered tiling reduces edge artifacts.
- Detectree2 predicts polygon masks.
- Predictions are georeferenced.
- Polygons are cleaned/simplified/deduplicated.
- Detections are saved at multiple confidence thresholds.

Technical details to include if final values are stable:

- tile size around 40 m or 45 m depending on site;
- buffer around 30;
- simplify tolerance around 0.3;
- thresholds such as `conf_0p15` to `conf_0p65`;
- base tracking threshold such as `conf_0p45`;
- high-confidence alignment threshold such as `conf_0p65`.

Need from user:

- final canonical tile/buffer/threshold settings;
- whether settings vary by site;
- which exact settings produced final results.

### 4.3 Multi-Threshold Crown Store

This deserves its own subsection because it is one of the practical innovations.

Argument:

- A fixed detector threshold fails under seasonal and illumination shifts.
- High thresholds are precise but miss leaf-off or shadowed crowns.
- Low thresholds recover weak crowns but introduce false positives.
- Storing all thresholds lets tracking use a reliable base layer but recover missing dates from lower-confidence layers when supported by temporal context.

Write:

- define threshold layers;
- define base threshold;
- define fallback thresholds;
- explain how lower-confidence candidates are used only in constrained gap-filling contexts.

### 4.4 Residual Orthomosaic Alignment

Write:

- OM1 or first orthomosaic is fixed as reference.
- Downsampled grayscale previews are used for registration.
- Phase cross-correlation estimates image translation.
- Tiled PCC estimates local shifts and aggregates robustly.
- Resulting translation is applied to crown geometries before matching.

Frame honestly:

- This is not full bundle-adjustment or dense warping.
- It reduces dominant residual shift.
- Local distortions remain a limitation.

Possible equation:

Let `I_t` be orthomosaic at date `t`; estimate translation `delta_t = (dx_t, dy_t)` relative to reference or previous date; transform crown polygon `P_t` to aligned coordinates:

```text
P'_t = P_t + delta_t
```

If alignment is chained date-to-date, define whether `delta_t` is cumulative.

Need from user/code check:

- confirm final alignment is to OM1 directly or consecutive/cumulative;
- confirm `pcc_tiled` is the final method in current pipeline.

### 4.5 Temporal Crown Graph Construction

Define:

- Node: one crown polygon observation at date `t`.
- Edge: plausible same-tree association between crown at date `t` and crown at date `t + 1`.
- Graph: directed layered graph over dates.

Candidate features:

- IoU;
- overlap relative to previous crown;
- overlap relative to current crown;
- centroid distance;
- normalized centroid distance;
- area ratio;
- compactness similarity;
- eccentricity similarity;
- containment indicators.

Suggested scoring language:

> Candidate links are scored using a weighted combination of spatial overlap, centroid proximity, and shape/area consistency. Different geometric cases, such as direct overlap, containment, nearby non-overlap, and missing candidates, use different thresholds to avoid forcing a single global rule.

Keep this at the right level:

- enough detail to be reproducible;
- avoid dumping implementation internals unless they become equations/table.

### 4.6 Split, Merge, and Ambiguity Handling

Explain why graph tracking matters.

Cases:

- one-to-one crown continuation;
- split: one crown becomes multiple polygons;
- merge: multiple crowns become one polygon;
- missing detection;
- nearby candidate after small drift;
- false positive detection.

Describe:

- graph retains multiple plausible links;
- best backbone can be extracted for stable tree identity;
- branching chains can be flagged for manual review or lower confidence;
- full width-1 chains are highest-confidence identities;
- long partial chains are useful after filtering.

### 4.7 Chain Extraction and Gap Filling

Write:

- A temporal chain is a sequence of crown observations hypothesized to represent the same tree.
- Full chains cover all dates.
- Partial chains cover a subset.
- Broken chains can result from missed detections or real canopy invisibility.
- Gap filling searches lower-threshold detections near predicted/interpolated positions and accepts candidates only when spatially consistent.

Need from user:

- exact final rule for full vs partial chain inclusion;
- whether virtual nodes are implemented in final output or only conceptual;
- whether gap-filled lower-threshold detections are marked with provenance.

### 4.8 Consensus Crown Generation

This is a central method subsection.

Write:

- Detections in a chain vary by date.
- Direct per-date polygon sampling would mix changes in crown geometry with changes in color/phenology.
- A consensus crown creates a stable ROI for repeated sampling.

Implemented method:

- medoid consensus polygon.
- Selects the observed polygon with minimum total dissimilarity to others in the chain.
- Dissimilarity combines centroid distance, `1 - IoU`, and area inconsistency.

Suggested formula:

For polygons `P_i` in chain `C`, choose:

```text
P* = argmin_i sum_j D(P_i, P_j)
```

where:

```text
D(P_i, P_j) = w_d * centroid_distance(P_i, P_j)
            + w_iou * (1 - IoU(P_i, P_j))
            + w_a * (1 - area_similarity(P_i, P_j))
```

Then explain:

- medoid stays on a real observed polygon;
- avoids invalid geometry from repeated intersections/unions;
- robust enough for noisy time series;
- alternative consensus strategies include intersection core and union-shrink.

### 4.9 Crown Crop Extraction

Write:

- Consensus crowns are transformed back into each orthomosaic coordinate frame.
- Raster pixels under the crown polygon are cropped/masked.
- Crops are saved per tree/date for visual inspection and feature computation.

Important:

- If alignment transform is applied to detections, sampling from raw orthomosaic requires inverse transform.
- Mention this explicitly because it shows technical rigor.

### 4.10 RGB and Texture Phenology Features

Features:

- GCC: `G / (R + G + B)`;
- RCC: `R / (R + G + B)`;
- vegetation fraction from HSV thresholds;
- shadow fraction;
- valid pixel fraction;
- gray entropy;
- Laplacian variance.

Quality control:

- low valid-pixel fraction;
- high shadow fraction;
- very low sharpness/texture;
- possibly missing crop or low area.

Write:

> These features are not intended to replace field phenology observations. They provide interpretable image-derived proxies for canopy greenness, senescence/reddishness, shadow contamination, and texture/structural change.

### 4.11 Rule-Based Phenology State and Deciduousness Scoring

Write:

- Missing/bad observations are interpolated where appropriate.
- Seasonal amplitude is computed from vegetation fraction, GCC, and texture.
- Leaf-off depth is based on minimum vegetation fraction.
- Amplitudes are normalized across crowns.
- Weighted score classifies deciduous-like crowns.
- Per-date phenophase labels for deciduous crowns:
  - leaf-on;
  - leaf-off;
  - transitioning.

Be honest:

> The rule-based phenology score is an interpretable baseline designed for low-label settings. It should be treated as a proxy until validated against denser field phenology labels.

### 4.12 Field Label and Species Integration

This begins the bridge into satellite/species experiments.

Write:

- Consensus crown IDs become the primary keys for field labels.
- QField/manual labels are joined to crown geometries.
- Labels may include species, genus, deciduousness, flowering category, showy flower color, or other ecological traits.
- Outputs can be visualized in Google Earth or an HTML viewer for review.

Need from user:

- final label columns and definitions;
- whether to use "species classification", "trait classification", or both.

### 4.13 Satellite and Embedding Feature Extraction

Make this a full method subsection, not appendix.

Write:

- Drone-derived consensus crowns provide geospatial training units.
- Sentinel-2 features are extracted for crown locations or buffered crown geometries as an interpretable seasonal baseline.
- Sentinel-2 seasonal features include raw bands, vegetation indices, seasonal medians/amplitudes, and harmonic phenology coefficients where used.
- Sentinel-1 features provide complementary radar/time-series information.
- Temporal features are computed across seasonal windows.
- Google Earth Engine satellite embedding features provide a learned remote-sensing representation, currently the most trusted direction for the satellite/species section.
- Compare original crown geometry extraction with centroid-buffer extraction when that choice affects classifier behavior.
- Report crown area distributions against the 10 m Sentinel-2 pixel footprint to make scale mismatch explicit.
- DINO/visual crown embeddings can be described as secondary/exploratory if they are not part of the final classifier claim.

Potential extraction strategies:

- centroid sampling;
- buffered crown sampling;
- fractional crown-pixel overlap;
- annual or multi-year Sentinel-2 summaries;
- vegetation indices and raw bands;
- temporal aggregates;
- leave-area-out validation splits.

Need from user/code:

- which exact features are final;
- whether GEE or STAC is the final extraction source;
- which model family is final: random forest, threshold sweeps, pixel classifier, embeddings.
- whether original crown polygons or 20 m centroid buffers are the final extraction geometry for each task.

### 4.14 Species and Phenological Trait Classifiers

Write:

- Classification tasks:
  - Acacia vs non-Acacia;
  - deciduous vs rest;
  - showy flower vs rest;
  - red/yellow flowering categories;
  - ESD multiclass if defensible.
- Models:
  - random forest classifiers;
  - threshold sweeps;
  - leave-area-out validation;
  - random split as less strict baseline.

Important framing:

> These classifiers evaluate whether crown identities derived from UAV tracking can serve as labeled ecological units for broader trait mapping. They are not presented as a complete solution to species mapping under all canopy mixtures.

---

## 6. Evaluation and Results Plan

The paper needs evaluation that follows the pipeline logic.

Important current-results note:

The output folders inspected so far mostly correspond to **older partial runs**, not the final full LHC/SIT datasets. The current full available datasets are LHC = 13 OMs and SIT = 19 OMs, but the cleanest currently inspected pipeline outputs are:

- `output/lhc_pipeline_fixed`: LHC, 8 OMs, `pcc_tiled`, base threshold `conf_0p45`, alignment threshold `conf_0p65`.
- `output/sit_pipeline_fixed`: SIT, 14 OMs, `pcc_tiled`, base threshold `conf_0p15`, alignment threshold `conf_0p65`.

These numbers can be used as placeholders or historical partial-run evidence, but final thesis tables should be regenerated from the latest full-data runs.

### 6.0 How to Read Pipeline Output Folders

For a complete pipeline run, the folder has the following structure:

```text
output/<run_name>/
  pipeline_config.json
  02_tracking/
    consensus_crowns_complete_all.gpkg
    consensus_crowns_complete_all_raw.gpkg
    consensus_crowns_om1_phenology.geojson
    consensus_crowns_summary.json
    tracking_quality_metrics.json
    tracking_quality_report.txt
    diagnostics/
      alignment_shifts.csv
      chain_breakdown.json
      tracking_diagnostics_report.txt
      alignment_shifts.png
      match_rates_by_pair.png
      chain_length_distribution.png
      consensus_overlay_om1_raw.png
  03_phenology/
    tree_master_geojson.geojson
    phenology_features_raw.csv
    leafshed_tree_scores.csv
    leafshed_phenophase_by_om.csv
    leafshed_normalizers.json
    leafshed_config.json
  04_viewer/
    index.html
    manifest.json
    crops/
```

Main files for thesis tables:

- `pipeline_config.json`: tells which OMs, crown directory, thresholds, alignment method, and steps were used.
- `02_tracking/diagnostics/tracking_diagnostics_report.txt`: best human-readable summary of alignment shifts, graph metrics, chain counts, and consensus counts.
- `02_tracking/consensus_crowns_summary.json`: best machine-readable summary of tracking, chain breakdown, deduplication, and alignment parameters.
- `02_tracking/diagnostics/alignment_shifts.csv`: per-orthomosaic residual shift values.
- `03_phenology/leafshed_tree_scores.csv`: one row per consensus crown with deciduous score and crown-level classification.
- `03_phenology/leafshed_phenophase_by_om.csv`: one row per crown/date with phenophase label.
- `03_phenology/phenology_features_raw.csv`: one row per crown/date with raw RGB/texture/QC features.
- `03_phenology/tree_master_geojson.geojson`: canonical geospatial output combining crown geometry, tracking metadata, classification, temporal summary, and per-date observations.

Output hierarchy interpretation:

- `*_pipeline_fixed` folders are the best current source for thesis-style partial-run results because they are produced by the newer pipeline scripts and include configs, diagnostics, phenology, and viewer outputs.
- `*_pipeline_v2` folders are useful historical comparisons but used `crowns` alignment in the inspected runs, while the current recommended tracking pipeline uses `pcc_tiled`.
- `lhc_tracking_*` and `sit_tracking_*` folders are notebook-era or rerun outputs. They are useful for development history and sensitivity discussion, but should not be mixed with final numbers unless clearly labeled.
- Root-level `output/*.json`, `output/*.csv`, and visualization folders are older exploratory artifacts. They can support the project history but should not be the main source for final tables.

## 6.1 Crown Detection Evaluation

Report:

- manual ground-truth crown count;
- prediction count;
- TP/FP/FN;
- precision;
- recall;
- F1;
- mean IoU of matched crowns.

Known early SIT result:

```text
Ground truth crowns: 124
Predictions: 73
True positives: 32
False positives: 41
False negatives: 92
Precision: 43.8%
Recall: 25.8%
F1: 32.5%
Mean IoU of matched crowns: 0.72
```

How to interpret:

- The detector often gives geometrically reasonable masks when it detects correctly.
- Recall is limited.
- This supports the need for multi-threshold detection and temporal recovery.
- Do not oversell Detectree2 as solved crown delineation.

Potential table:

| Site | Date | GT crowns | Predictions | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIT | TBD | 124 | 73 | 0.438 | 0.258 | 0.325 | 0.72 |

Need:

- decide whether to include only this early evaluation or produce updated evaluation on final site.

## 6.2 Alignment Evaluation

Possible metrics:

- estimated pixel/meter shifts per date;
- crown IoU before vs after alignment for high-confidence anchors;
- centroid residual distance before vs after alignment;
- visual overlay before/after.

Recommended ablation:

```text
no alignment
whole-image PCC
tiled PCC
centroid median shift
ORB/ECC if available
```

If we only have qualitative results:

- show overlay figures;
- report representative shift values;
- describe observed improvement cautiously.

Need:

- final alignment logs/statistics from pipeline runs.

## 6.3 Tracking Evaluation

Best evaluation if possible:

- manually annotate a subset of crown identities across several dates;
- compare predicted chain associations to manual identities;
- report link precision/recall or identity consistency.

Possible metrics:

- link accuracy between consecutive dates;
- chain completeness;
- number of full chains;
- number of partial chains above length threshold;
- branch rate;
- gap-fill acceptance rate;
- number/percentage of consensus crowns retained after deduplication.

If no manual tracking benchmark:

- report proxy metrics and call them internal consistency metrics;
- add a limitation that full identity validation remains future work.

Suggested table:

| Site | Dates | Base threshold | Detections/date | Full chains | Partial chains | Gap-filled observations | Final consensus crowns |
|---|---:|---:|---:|---:|---:|---:|---:|

Current partial-run output inventory:

| Run | Site | OMs in run | Alignment | Base threshold | Full chains | Branching chains | Extracted backbones | Partial chains added | Raw consensus | Final consensus | Match rate | Avg chain length |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lhc_pipeline_fixed` | LHC | 8 | `pcc_tiled` | `conf_0p45` | 2 | 62 | 62 | 77 | 141 | 87 | 0.921 | 1.83 |
| `sit_pipeline_fixed` | SIT | 14 | `pcc_tiled` | `conf_0p15` | 7 | 44 | 44 | 189 | 240 | 131 | 1.026 | 2.37 |
| `sit_test_4om` | SIT debug subset | 4 | `pcc_tiled` | `conf_0p45` | 21 | 33 | 33 | 0 | 54 | 51 | 0.952 | 1.76 |

Interpretation:

- The fixed pipeline runs are the cleanest current summaries because they preserve `pipeline_config.json`, tracking diagnostics, chain breakdowns, consensus summaries, phenology CSVs, and viewer outputs in one directory.
- LHC `pipeline_v2` and SIT `pipeline_v2` used `crowns` alignment and are less aligned with the current recommended method; keep them as historical comparisons unless there is a reason to discuss alignment ablations.
- Older `lhc_tracking_*` and `sit_tracking_*` folders show the development history and reruns, but should not be mixed into final result tables unless explicitly labeled as earlier experiments.
- Match rate can exceed 1.0 because the graph permits many-to-many candidate links; it should be described as an internal graph connectivity/matching-rate diagnostic, not as a conventional accuracy score.

## 6.4 Consensus Crown Evaluation

Evaluate whether consensus crowns improve sampling stability.

Possible metrics:

- crop availability across dates;
- valid pixel fraction distribution;
- vegetation feature smoothness;
- reduced sudden jumps from segmentation area changes;
- visual examples comparing per-date polygons vs medoid consensus.

Recommended figure:

- one tree row:
  - date-wise crown detections;
  - consensus crown overlay;
  - crop sequence;
  - GCC/vegetation fraction time series.

## 6.5 Phenology Results

Report:

- example leaf-on, leaf-off, transition trajectories;
- deciduousness score distribution;
- map of deciduous/stable crowns;
- per-date phenophase map;
- representative crown crop panels.

Important:

- Separate image-derived phenology proxy from validated biological phenology.
- If QField/field validation exists, report agreement.
- If not, present as preliminary/heuristic with visual QA.

Possible table:

| Crown ID | Species | Deciduousness score | Leaf-off date | Leaf-on recovery date | QC notes |
|---|---|---:|---|---|---|

Current partial-run phenology inventory:

| Run | Site | OMs | Consensus crowns scored | Deciduous crowns | Feature records | Bad observations | Leaf-on obs | Leaf-off obs | Transition obs | Stable obs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lhc_pipeline_fixed` | LHC | 8 | 87 | 60 | 696 | 20 | 246 | 182 | 52 | 216 |
| `sit_pipeline_fixed` | SIT | 14 | 131 | 104 | 1834 | 89 | 568 | 603 | 285 | 378 |

Interpretation:

- These counts show that the current pipeline produces crown-level phenology trajectories and per-date phenophase labels at scale.
- Because the runs are partial relative to the latest available data, the paper should not present these as final biological findings.
- The rule-based deciduousness and phenophase labels should be framed as image-derived phenology proxies unless field phenology validation is added.

## 6.6 Field/Species Label Results

Report:

- number of labeled crowns;
- species distribution;
- label source distribution, with current framing that labels are field verified;
- QField-to-crown joining workflow;
- any manual cleaning or reconciliation between field labels and crown IDs.

Possible outputs:

- map of species labels;
- species x phenology summaries;
- examples where RGB time series differs by species/trait.

## 6.7 Satellite and Species Classifier Results

This should be a real results section.

Current decision:

- Keep this as a proper main-paper section.
- For now, write only headings and result slots.
- The strongest trusted direction is the Google Earth Engine embeddings experiment.
- Detailed metrics, task selection, and validation claims should be filled later after the satellite/species workflow stabilizes.

Suggested organization:

### 6.7.1 Satellite Feature Extraction Summary

Report:

- crown samples used;
- Google Earth Engine embedding source and temporal window;
- how crown geometries/points are linked to embedding features;
- cloud/missing-data handling if relevant;
- final feature dimensionality.
- original crown geometry versus buffered centroid extraction, if both are compared.

### 6.7.2 Classification Tasks

Candidate tasks to report if available:

- Acacia vs non-Acacia;
- deciduous vs rest;
- showy flower vs rest;
- red showy;
- yellow showy broad/strict;
- ESD multiclass.

Current stance:

- Keep these as headings/placeholders until the final GEE embedding results are chosen.
- Do not over-commit to all tasks being paper-ready.
- The weekly meetings indicate a usable label inventory for this section: clean species labels, ambiguous labels, ESD categories, Acacia labels, and flowering-color/showiness labels. Exact final counts should be regenerated from the current label table before writing the final paper.

### 6.7.3 Validation Strategy

Prioritize:

- leave-area-out validation as stronger evidence of spatial generalization;
- random split as an easier baseline, clearly labeled.
- leave-species-out validation can be reported when the question is trait generalization beyond seen species.

### 6.7.4 Model Performance

Report:

- accuracy;
- precision/recall/F1;
- ROC-AUC if binary and appropriate;
- confusion matrix;
- model comparison only if multiple feature sets are mature enough;
- feature importance only for interpretable models where applicable.
- Sentinel-2 seasonal/harmonic baselines should be separated from GEE embedding results so the paper does not mix interpretable baselines with the stronger learned representation.

### 6.7.5 Interpretation

Key argument:

> The satellite/species experiments demonstrate that the drone-tracked crown layer can act as a structured label and validation product for coarser remote-sensing models.

Avoid:

- claiming operational species mapping if validation is weak;
- claiming satellite pixels identify individual crowns directly;
- hiding spatial leakage if random splits perform much better than leave-area-out.

### 6.7.6 Crown Size and Satellite Pixel Scale

Report:

- crown area distribution by site;
- fraction of crowns smaller than, comparable to, or larger than a 10 m Sentinel-2 pixel;
- effect of area filtering on retained label count;
- why sub-pixel crowns make satellite trait inference harder, especially in dense or mixed canopies.

This can be a compact but important result because it explains why some satellite classifiers generalize poorly even when random-split performance looks acceptable.

---

## 7. Discussion

### 7.1 From Detection to Identity

Main interpretive point:

> The key methodological shift is from detecting crowns independently to constructing persistent tree identities.

Discuss:

- why crown detection metrics alone do not determine phenology success;
- how graph tracking and consensus crowns convert noisy detections into analyzable ecological units.

### 7.2 Why Multi-Threshold Detection Helped

Discuss:

- threshold instability across dates;
- high thresholds for precision;
- low thresholds for recovery;
- temporal context prevents low-threshold false positives from dominating.

### 7.3 Alignment as a Practical Requirement

Discuss:

- repeated UAV maps are not perfectly co-registered;
- even small residual shifts matter at crown scale;
- PCC alignment is lightweight and reproducible;
- local distortions remain.

### 7.4 Consensus Crowns and Stable Sampling

Discuss:

- consensus crowns make phenology time series less dependent on segmentation noise;
- medoid is simple, valid, and interpretable;
- intersection/union alternatives might be explored later.

### 7.5 Field, Species, and Satellite Integration

This should be substantial because user wants satellite/species properly included.

Discuss:

- individual-tree drone identities become training/validation units;
- species labels turn crown tracks into ecological objects;
- satellite features test whether high-resolution crown labels can inform broader mapping;
- scale mismatch remains a central limitation;
- strongest evidence should come from leave-area-out validation.
- GEE embeddings should be framed as a stronger learned representation, while Sentinel-2 seasonal/harmonic features provide an interpretable baseline.
- crown-area analysis should be used to explain why some sites/tasks are more realistic for satellite inference than others.

### 7.6 Operational Lessons

Discuss:

- repeated mission planning matters;
- orthomosaic quality controls downstream success;
- poor reconstructions cannot always be fixed later;
- manual validation remains important;
- interactive visualization is not just a demo but a QA tool.

### 7.7 Limitations

Be direct:

- crown detection recall is limited;
- orthomosaic distortions and shadows remain;
- translation alignment does not correct all local deformation;
- graph tracking needs stronger manual identity validation;
- rule-based phenology is a proxy unless validated;
- species/satellite classifiers depend on label quality and may suffer spatial leakage or scale mismatch;
- results may not generalize across forest types without retuning.

### 7.8 Future Work

Include:

- manual temporal identity benchmark;
- stronger phenology field validation;
- learned temporal association model;
- non-tree and shadow filtering;
- active learning for uncertain chains;
- improved satellite fusion and spatially blocked validation;
- species-aware or phenology-aware tracking priors.

---

## 8. Conclusion

Draft conclusion paragraph:

This work demonstrates that individual-tree phenology monitoring from repeated UAV RGB orthomosaics depends on preserving tree identity through time. By combining multi-threshold crown segmentation, residual orthomosaic alignment, graph-based temporal association, gap filling, consensus crown generation, and crown-level feature extraction, the proposed pipeline converts independent orthomosaics into a structured dataset of tracked tree crowns and phenology trajectories. The resulting crown identity layer supports visual inspection, field validation, species linkage, and downstream satellite/embedding classifiers for ecological trait inference. The study therefore reframes UAV crown delineation as part of a larger temporal monitoring problem and provides a practical foundation for linking fine-scale drone observations with broader remote-sensing analyses.

---

## 9. Figure Plan

### Figure 1: Overall Workflow

Show:

```text
UAV survey -> WebODM orthomosaic -> Detectree2 crowns -> alignment -> graph tracking -> consensus crowns -> phenology -> field/satellite inference
```

Purpose:

- Gives professor/reviewer the whole paper in one figure.

### Figure 2: Study Sites and Data Timeline

Show:

- map of IITD/Sanjay Van sites;
- orthomosaic dates;
- which dates/sites are used for main tracking.

### Figure 3: Crown Detection and Multi-Threshold Outputs

Show:

- orthomosaic tile;
- detected polygons at high/medium/low thresholds;
- example of missed crown recovered at lower threshold.

### Figure 4: Alignment Before/After

Show:

- same crown overlays before alignment;
- after PCC/tiled PCC alignment;
- vector shift or residual plot.

### Figure 5: Graph-Based Crown Tracking

Show:

- nodes by date;
- candidate edges;
- split/merge example;
- extracted chain.

### Figure 6: Consensus Crown Construction

Show:

- per-date polygons for same tree;
- medoid/consensus polygon;
- resulting crop sequence.

### Figure 7: Phenology Time Series

Show:

- crown crops across dates;
- GCC/RCC/vegetation fraction line plot;
- leaf-on/leaf-off/transition labels.

### Figure 8: Species and Satellite Inference

Show:

- consensus crown labels;
- satellite feature extraction buffer/pixel overlap;
- classifier workflow;
- feature importance or confusion matrix.

---

## 10. Table Plan

### Table 1: Study Sites and Data

Columns:

- site;
- area;
- number of orthomosaics;
- date range;
- flight altitude;
- main use in paper;
- labels available.

### Table 2: Crown Detection Evaluation

Columns:

- site/date;
- GT crowns;
- predicted crowns;
- TP;
- FP;
- FN;
- precision;
- recall;
- F1;
- mean IoU.

### Table 3: Tracking and Consensus Summary

Columns:

- site;
- dates;
- base threshold;
- total detections;
- full chains;
- partial chains retained;
- gap-filled observations;
- final consensus crowns.

### Table 4: Phenology Feature Definitions

Columns:

- feature;
- formula;
- interpretation;
- QC sensitivity.

### Table 5: Species/Satellite Classification Tasks

Columns:

- task;
- labels/classes;
- features;
- model;
- validation split;
- primary metric;
- result.

---

## 11. What Goes in Main Paper vs Supplement

Since this is a master's thesis framed as a paper, the thesis can be wider than a strict 8-page paper. Recommended split:

### Main Body

Keep these in the main paper:

- UAV acquisition and orthomosaic generation.
- Detectree2 crown segmentation.
- multi-threshold detection.
- residual alignment.
- graph-based tracking.
- consensus crowns.
- phenology features and rule-based scoring.
- species label integration.
- satellite/species classifier results.

### Supplement / Appendix

Move here if too long:

- full code/config tables;
- all threshold sensitivity results;
- all classifier config JSONs;
- weaker exploratory embedding experiments;
- detailed notebook history;
- failed methods like early DeepForest, unless used as motivation.

Important:

Satellite/species should stay in the main body, but only the strongest classifier tasks and cleanest validation should be emphasized. Weaker or exploratory variants can go to supplement.

---

## 12. Decisions We Need Before Writing Full Draft

These are the current decisions and remaining open points before polished prose.

### Decision 1: Final Title

Resolved working title:

**Tracking Individual Tree Phenology from Repeated UAV RGB Orthomosaics Using Graph-Based Crown Association and Cross-Scale Ecological Inference**

### Decision 2: Main Study Site

Resolved:

- Primary study area: IIT Delhi campus.
- Primary tracking/phenology demonstration sites: LHC and SIT.
- Current full-data availability:
  - LHC: 13 orthomosaics/dates.
  - SIT: 19 orthomosaics/dates.
- Existing output summaries inspected so far are partial older runs and should be replaced by full-run results later.

### Decision 3: Final Satellite/Species Scope

Partly resolved:

- Satellite/species work should remain in the main paper, but detailed claims should wait.
- The most trusted current direction is the Google Earth Engine embeddings experiment.
- For now, keep satellite/species as headings and planned result slots.

Still need:

- final embedding feature source;
- final classifier tasks;
- final validation split;
- final metrics.

### Decision 4: Validation Strength

Partly resolved:

- A small manually validated tracking subset will be done later.
- For the current outline stage, keep tracking validation as a planned results subsection.
- Once validation is done, include manual link/identity metrics and visual examples in the results.

### Decision 5: Phenology Label Strength

Current drafting stance:

- Rule-based leaf state has not yet been validated against field observations.
- Do not foreground this as a weakness in the paper draft.
- For now, write the section neutrally as image-derived phenology state estimation.
- If time permits, add field validation results later.

### Decision 6: CHM/Height

Resolved:

- Exclude CHM/tree-height estimation from the main claim and method.
- Do not present DTM/DSM/CHM as our work unless new results are added later.
- This came from the overlapping companion-project context, not the current thesis spine.

---

## 13. Immediate Writing Plan

Recommended order:

1. Lock title and central contribution.
2. Fill Table 1 with exact LHC/SIT date ranges and 13/19 OM lists.
3. Run or ingest final full-data LHC/SIT tracking outputs.
4. Replace current partial-run result tables with final full-run numbers.
5. Keep satellite/species headings in main paper; fill after GEE embeddings work stabilizes.
6. Write Introduction and Contributions.
7. Write Method as pipeline.
8. Write Results around figures/tables.
9. Write Discussion honestly around limitations and cross-scale value.

---

## 14. Questions for User

These are the most important questions to answer next.

1. What are the exact 13 LHC orthomosaic filenames/dates in the final run?
2. What are the exact 19 SIT orthomosaic filenames/dates in the final run?
3. Which output folder will contain the latest full-data LHC/SIT results once you share them?
4. Do we have or want a small manually checked tracking-validation subset?
5. What exact field-verified label schema should the species section use?
6. What is the final Google Earth Engine embeddings workflow and which classifier result should become the main satellite/species result?
