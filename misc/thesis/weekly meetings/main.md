# Weekly Meeting Presentation Digest

This document summarizes the extracted text from the weekly meeting presentations in chronological order. The goal is not to reproduce every slide, but to record the technical ideas, decisions, experiments, failures, and thesis-relevant context that emerged through the project.

Main timeline in one sentence:

The project evolved from generating georeferenced UAV orthomosaics and testing DeepForest detections, to Detectree2 polygon crown delineation, to graph-based temporal crown association with alignment and multi-threshold recovery, then to consensus crown phenology extraction, field-verified species linkage, and satellite/Google Earth Engine embedding experiments for cross-scale ecological inference.

## High-Level Evolution

- January-February 2025: Orthomosaic generation, GPS/UTM/EXIF issues, DeepForest baselines, and first tree-tracking data-structure concepts.
- March-April 2025: Shift from bounding boxes toward polygon segmentation with Detectree2; first evaluation metrics, Hungarian matching, and split/merge case thinking.
- May-July 2025: Recognition that per-date detections are unstable; exploration of IoU/containment logic, crown merging, affine/translation alignment, and missing-crown handling.
- August-December 2025: Graph-based tracking becomes central; similarity scoring, conditional edge cases, Detectree2 threshold tuning, multi-threshold crown ideas, pruning, and split/merge successor selection are developed.
- January-March 2026: Consensus crowns, phase cross-correlation alignment, partial-chain consensus, phenology metrics, deciduousness scoring, manual error taxonomy, and LHC/SIT analysis mature.
- April-June 2026: Interactive viewer, pipeline formalization, master GeoJSON, field/QField integration, satellite harmonic/embedding experiments, species/trait classifiers, and Sanjay Van/IITD cross-site label work enter the thesis scope.

## 2025-01-28: First Good Orthomosaic and DeepForest Coordinate Problem

Source: [2025-01-28_01_28_Jan_2025.md](<extracted text markdown/2025-01-28_01_28_Jan_2025.md>)

Important points:

- The team finally generated a usable orthomosaic after experimenting with WebODM.
- A key practical finding was that very small image sets, fewer than about 25 images, did not produce good orthomosaics; the successful run used 67 images.
- The early crown-detection baseline was DeepForest.
- The immediate technical blocker was extracting geospatial coordinates for DeepForest-detected tree boxes.
- The project was still working with relative image coordinates rather than usable latitude/longitude or projected coordinates.
- There was an early need for GPU/HPC access because DeepForest and coordinate extraction libraries were difficult to run on local hardware.

Thesis relevance:

- This establishes the photogrammetric foundation of the project.
- It shows that the first technical layer was not machine learning, but making repeatable georeferenced UAV products.
- It also motivates why later geospatial consistency, CRS handling, and alignment became important.

## 2025-02-04: XMP to EXIF, UTM Coordinates, and WebODM Georeferencing

Source: [2025-02-04_01_4_Feb_2025.md](<extracted text markdown/2025-02-04_01_4_Feb_2025.md>)

Important points:

- GPS extraction from drone shots and orthomosaics became functional.
- DJI Mini 4 Pro images stored coordinate metadata in XMP rather than standard EXIF fields.
- Scripts were created for viewing and transforming metadata formats.
- XMP-to-EXIF conversion was needed before uploading to WebODM so it could correctly read image coordinates.
- Orthomosaic outputs were interpretable in QGIS.
- The coordinate system was identified as WGS84 UTM 43N; projected UTM coordinates needed conversion to geographic latitude/longitude when required.
- `pyproj` was identified as the automation route for CRS conversion.
- WebODM runtime was already seen as a bottleneck.

Thesis relevance:

- Supports a methods subsection on photogrammetric processing and geospatial coordinate handling.
- Explains why all downstream crown polygons are treated as geospatial objects, not just image masks.
- The XMP-to-EXIF issue is a practical reproducibility detail for DJI Mini 4 Pro workflows.

## 2025-02-11: DeepForest Baseline, First Tracking Data Structure, and Alignment Concerns

Source: [2025-02-11_01_11_Feb_2025.md](<extracted text markdown/2025-02-11_01_11_Feb_2025.md>)

Important points:

- DeepForest detection was tuned but still missed trees and detected non-tree objects.
- Tree boxes were annotated with coordinates.
- A tree-monitoring data structure was planned.
- The first matching idea was nearest-neighbor matching using geospatial coordinates.
- Misalignment between repeated UAV images was already anticipated.
- ORB/SIFT/SURF keypoint matching, RANSAC, homography, and affine transforms were discussed as possible alignment strategies.
- Buildings and roads were proposed as more stable alignment features than trees, because seasonal tree changes could hinder alignment.
- YOLOv8 was considered as another detection option.

Thesis relevance:

- This is the first explicit appearance of the temporal identity problem.
- The project already understood that repeated surveys require matching the same tree across runs.
- It also shows why simple nearest-neighbor matching was only an initial baseline.

## 2025-02-18: Detectron2, Segmentation Models, and Two-Stage Canopy Extraction

Source: [2025-02-18_01_18_Feb_2025.md](<extracted text markdown/2025-02-18_01_18_Feb_2025.md>)

Important points:

- The team studied Detectron2, Mask R-CNN-style object detection, FPN, RPN, and ROI heads.
- A two-stage approach was proposed:
  - semantic segmentation for tree/canopy patches;
  - DeepForest or another detector inside those patches.
- UNet, DeepLab, Mask R-CNN, DeepLabV3+, and PointRend were reviewed.
- The motivation was reducing DeepForest false positives and improving crown boundary extraction.
- The team recognized that crown size variation requires multi-scale feature handling.

Thesis relevance:

- This supports the related-work pathway from generic object detection to instance segmentation.
- It explains the technical rationale for moving away from bounding-box DeepForest toward polygon-based crown delineation.
- It also helps justify Detectree2 as a Mask R-CNN/Detectron2-based method.

## 2025-03-04: Tiling, OOM Problems, Detectree2 Discovery, and TreeTracker

Source: [2025-03-04_01_4_Mar_2025.md](<extracted text markdown/2025-03-04_01_4_Mar_2025.md>)

Important points:

- Large orthomosaics, around 10742 x 11102 pixels, caused GPU memory and image-size warnings.
- Tiling was adopted as the solution for high-resolution orthomosaic processing.
- Tile boundary issues were identified early:
  - cut-off crowns;
  - duplicate detections;
  - need for overlap/buffer and post-processing.
- Detectree2 was identified as a tree-specific polygon segmentation library.
- Detectree2's workflow was understood as tiling, model training/tuning, evaluation, and full-region prediction.
- The early `TreeTracker` concept used DeepForest boxes, geospatial centers, scores, labels, persistence via pickle, and Euclidean distance matching.

Thesis relevance:

- Tiling and buffering should be described as core engineering requirements, not incidental implementation details.
- The limitations of DeepForest and generic segmentation provide strong motivation for Detectree2.
- This is also the first point where stateful tracking becomes a concrete software object.

## 2025-03-11: First Detectree2 Outputs and GeoPackage Crowns

Source: [2025-03-11_01_11_Mar_2025.md](<extracted text markdown/2025-03-11_01_11_Mar_2025.md>)

Important points:

- A new orthomosaic dated 2025-03-06 was processed.
- Detectree2 was run using a pretrained model, `230103_randresize_full.pth`.
- The large orthomosaic was split into 45 x 45 tiles with a 20-pixel buffer.
- Tile predictions were converted into georeferenced GeoJSON outputs.
- Overlapping crowns were filtered using confidence and IoU logic.
- Processed crowns were saved as GeoPackage (`.gpkg`) files.
- Next steps were integrating Detectree2 into the data structure and validating tracking across the two available orthomosaics.

Thesis relevance:

- This marks the transition from box detections to georeferenced crown polygons.
- It anchors the pipeline's crown store outputs: GeoJSON/GPKG polygons.
- The tile/buffer/dedup pattern becomes part of the final method.

## 2025-03-18: SAM Exploration and First Crown-Matching Algorithms

Source: [2025-03-18_01_18_Mar_2025.md](<extracted text markdown/2025-03-18_01_18_Mar_2025.md>)

Important points:

- SAM was implemented and produced promising segmentation.
- The team explored taking SAM output into Detectree2 and overlaying results on GeoTIFF imagery.
- The goal was to assign persistent IDs to detected crowns for phenology tracking.
- Euclidean distance matching was considered using crown coordinates, radius, and area.
- IoU-based overlap matching was proposed, with IoU > 0.75 as a possible same-tree rule.
- A hybrid approach was proposed: high-confidence IoU matching first, then Euclidean/crown-attribute matching for remaining detections.
- Site profiling on campus was discussed.
- GPU workstation access became available.

Thesis relevance:

- This is a clean conceptual predecessor to the final graph edge features: IoU, centroid distance, area, and shape.
- The date also shows that persistent tree IDs were already the real target, not just segmentation.

## 2025-03-25: LHC Drone Runs, Campus Site Expansion, and First Multi-Threshold/Alignment Lessons

Sources:

- [2025-03-25_01_25_Mar_2025_duplicate.md](<extracted text markdown/2025-03-25_01_25_Mar_2025_duplicate.md>)
- [2025-03-25_02_25_Mar_2025.md](<extracted text markdown/2025-03-25_02_25_Mar_2025.md>)

Important points:

- LHC patch drone runs were underway.
- Taxila/Nalanda waypoint issues were encountered because of DJI flight/landing-zone restrictions.
- Detectree confidence layers from `conf_0p15` to `conf_0p65` were already being explored.
- A critical bug was identified: alphabetic sorting put OM10 before OM1, corrupting chronological alignment. This was fixed using explicit chronological ordering.
- LHC 2025-12-09 was identified as highly misaligned.
- Global phase cross-correlation could fail across large gaps.
- Tiled PCC was proposed:
  - split overlap into tiles;
  - run PCC per tile;
  - reject high-error or too-large-shift tiles;
  - aggregate shifts using the median.
- Leaf-shed classifier definitions and tree/non-tree rules started appearing.
- GCC/vegetation amplitude rules were used to identify non-tree objects.
- A PR/instructions contribution related to DJI Mini 4 Pro was mentioned.

Thesis relevance:

- Multi-threshold crown storage did not appear suddenly; it emerged from early threshold experiments.
- The chronological sorting bug is a useful cautionary implementation lesson.
- Tiled PCC became a key final alignment method and should be treated as a methodological contribution.
- The 2025-12-09 bad LHC orthomosaic explains why later LHC runs exclude that date.

## 2025-04-08: Regular Drone Surveys and Hungarian Matching Plan

Source: [2025-04-08_01_8_Apr_2025.md](<extracted text markdown/2025-04-08_01_8_Apr_2025.md>)

Important points:

- Drone shots of LHC, SAC, and SIT were collected on 2025-04-03.
- The project moved toward regular weekly drone shots to capture tree features.
- Detectree results were shown on LHC and SIT patches.
- The next work item was implementing the Hungarian algorithm and metrics.

Thesis relevance:

- Marks the transition from isolated experimentation to repeated monitoring.
- Establishes LHC and SIT as recurring analysis sites.
- Hungarian matching becomes the main baseline to compare against graph-based tracking.

## 2025-04-15: Detection Metrics and First Quantitative Crown Evaluation

Source: [2025-04-15_01_15_Apr_2025.md](<extracted text markdown/2025-04-15_01_15_Apr_2025.md>)

Important points:

- Hungarian matching was introduced as the assignment baseline.
- Initial cost matrix used centroid distance and crown radius terms.
- Ground truth processing was implemented by converting annotations into the same coordinate space as the orthomosaic.
- Crown-detection metrics at IoU 0.5 were reported:
  - true positives: 32;
  - false positives: 41;
  - false negatives: 92;
  - precision: 0.438;
  - recall: 0.258;
  - F1: 0.325;
  - omission error: 74.19 percent;
  - commission error: 56.16 percent;
  - average matched IoU: 0.724.
- Interpretation: Detectree2 predictions were spatially accurate when correct, but many trees were missed and many predictions were spurious.
- Literature comparison included Detectree2/Mask R-CNN F1 around 0.64 in other settings and lower omission/commission errors in other multi-season studies.

Thesis relevance:

- These are the core early detection-evaluation numbers already present in reports.
- The high average IoU but low recall is important: correct masks can be good, but per-date detection is incomplete.
- This motivates temporal recovery, multi-threshold layers, and consensus crowns.

## 2025-04-22: Hungarian Cost Metric, Detectree2 Tuning, and Grid Search

Source: [2025-04-22_01_22_Apr_2025.md](<extracted text markdown/2025-04-22_01_22_Apr_2025.md>)

Important points:

- SIT and LHC orthomosaics were processed.
- Hungarian matching moved from Euclidean distance toward IoU cost, `1 - IoU`.
- A hybrid metric was proposed: IoU plus Euclidean/centroid distance.
- Smaller tile sizes, higher confidence, and IoU thresholds improved detection metrics:
  - TP increased from 32 to 47;
  - FN decreased from 92 to 77;
  - F1 increased from about 32.5 percent to about 41.1 percent;
  - mean matched IoU stayed around 0.73.
- Urban model performance with small tile size around 50 m was poor.
- Systematic grid search over Detectree2 parameters was proposed, with caching to avoid rerunning heavy prediction stages.

Thesis relevance:

- This supports an ablation/results narrative around detector parameter sensitivity.
- It explains why a single fixed Detectree2 configuration is brittle across sites/dates.
- It also previews the final use of both IoU and centroid distance in graph edges.

## 2025-04-29: Matching Case Taxonomy and Detection Banks

Source: [2025-04-29_01_29_Apr_2025.md](<extracted text markdown/2025-04-29_01_29_Apr_2025.md>)

Important points:

- A detailed case taxonomy for matching was developed:
  - exact one-to-one;
  - one-to-one partial;
  - containment base-to-target;
  - containment target-to-base;
  - one-to-many;
  - many-to-one;
  - unmatched base;
  - unmatched target.
- Each case had an ID-assignment strategy.
- One-to-many and many-to-one were explicitly treated as split/merge phenomena.
- Unmatched base crowns were kept as missing in that run; unmatched target crowns were temporary IDs promoted only if seen later.
- A "detection bank" idea was proposed: run Detectree multiple times with different thresholds, tile sizes, strides, seeds, or augmentations, then pool detections per date.
- Hybrid matching with IoU plus centroid distance was found more reliable than IoU alone, because faraway crowns can have misleading overlap.

Thesis relevance:

- This is the conceptual origin of the final graph case classifier.
- It strongly supports the claim that one-to-one assignment is insufficient.
- It also foreshadows multi-threshold crown stores.

## 2025-05-13: Prediction Score Matrix and Low-IoU Problem

Sources:

- [2025-05-13_01_13_May_2025_duplicate.md](<extracted text markdown/2025-05-13_01_13_May_2025_duplicate.md>)
- [2025-05-13_02_13_may_2025.md](<extracted text markdown/2025-05-13_02_13_may_2025.md>)

Important points:

- The team worked on a prediction score matrix over SIT orthomosaics.
- Containment-base problems were partially resolved visually.
- IoU thresholds had to be lowered to around 0.2 because maximum IoUs were often below 0.5.
- This revealed that crown polygons can visually correspond but still have low IoU due to boundary/area changes.
- Concave hull ideas were explored for boundary construction.

Thesis relevance:

- This justifies why final matching cannot rely only on high IoU.
- Low IoU despite visual correspondence becomes a central reason to combine overlap fractions, centroid distance, and area/shape features.

## 2025-05-21: Prediction Matrix Rejected and Sequential Consistency Idea

Source: [2025-05-21_01_21_may_2025.md](<extracted text markdown/2025-05-21_01_21_may_2025.md>)

Important points:

- The prediction-vs-score matrix idea was deemed infeasible.
- Reason: crown predictions were inconsistent across orthomosaics, and almost every prediction looked unique, so scores were near 1 and not discriminative.
- Many-to-one logic was refined using unary unions and area constraints.
- Overlap removal within the same orthomosaic was formalized:
  - remove contained fragments;
  - merge duplicates with high IoU;
  - clip partial overlaps where one polygon strongly covers another.
- A sequential tracking idea emerged: process OM1 and OM2, save the resulting crown set, then compare that persistent set to OM3, and so on.

Thesis relevance:

- This is a key argument against naive score matrices and toward graph/chain tracking.
- The current pipeline inherits the idea that per-date crown detections need cleaning and temporal context.

## 2025-06-16: DSM/SAM Literature and RGB-Only Constraint

Source: [2025-06-16_01_16Jun25.md](<extracted text markdown/2025-06-16_01_16Jun25.md>)

Important points:

- A paper using SAM plus DSM/elevation data was reviewed.
- DSM was understood as a surface-height model that could improve segmentation.
- BalSAM/RSPrompter-style DSM fusion was considered.
- The team recognized that the project mostly had RGB imagery, not LiDAR or a mature DSM workflow.
- Two options were discussed:
  - re-fly with structured nadir plus oblique imagery to generate DSM;
  - continue with RGB-only Detectree2.
- A feature data structure per crown was proposed, including mean color, leaf density, and species information.

Thesis relevance:

- This explains why CHM/DSM should not be claimed as a core contribution.
- It can be mentioned only as reviewed/considered background, not as implemented thesis work.
- It reinforces that the final pipeline is RGB-only.

## 2025-06-24: Affine Alignment and Crown Completion Logic

Source: [2025-06-24_01_24Jun25.md](<extracted text markdown/2025-06-24_01_24Jun25.md>)

Important points:

- Objective: align OM3 and OM4 crown sets and resolve missing crowns/geometric mismatches.
- CRS unification was performed.
- Nearest-neighbor centroid matches within 10 m were used to compute a rigid affine transform.
- Alignment improved IoU from around 30 percent to around 72-73 percent.
- Missing crowns from one OM were copied into the other with an `origin` label and inverse transform as needed.
- Difference masks `A - B` and `B - A` were used to visualize incomplete overlaps or shape shifts.

Thesis relevance:

- This is an important predecessor to residual alignment and gap filling.
- It shows that alignment improved crown correspondence dramatically.
- It also shows why alignment and temporal crown recovery are coupled.

## 2025-07-01: Median Translation and Outlier-Rejected Alignment

Source: [2025-07-01_01_1July25.md](<extracted text markdown/2025-07-01_01_1July25.md>)

Important points:

- The team compared mean translation, median translation, and outlier-filtered median translation.
- Mean translation was sensitive to bad matches.
- Median translation improved IoU from about 71 percent to about 80 percent.
- Removing the top 15 percent largest shift outliers and taking the median improved IoU further to about 81.34 percent.
- Post-alignment `A - B` and `B - A` difference analysis was used to reason about under/over-detected crown regions.

Thesis relevance:

- Supports robust aggregation language in the alignment method.
- Explains why median/inlier shifts are preferable to raw mean shifts.

## 2025-07-15: Pairwise Crown Merging and Multi-Temporal Alignment Failure Mode

Source: [2025-07-15_01_15July25.md](<extracted text markdown/2025-07-15_01_15July25.md>)

Important points:

- Crowns from OM3 and OM4 were merged if IoU exceeded 0.7.
- Difference geometry was selected using area constraints.
- Unmatched crowns were restored into the other crown set to preserve all trees.
- A key multi-temporal failure was identified:
  - OM4 aligned to OM3 is no longer in native coordinates;
  - aligning OM5 to already-transformed OM4 introduces inconsistencies.

Thesis relevance:

- This motivates careful coordinate-frame management.
- It also supports the later strategy of saving alignment shifts and reversing them for crop extraction.

## 2025-08-05: Graph-Based Tracking Formulation

Source: [2025-08-05_01_5Aug25.md](<extracted text markdown/2025-08-05_01_5Aug25.md>)

Important points:

- The project explicitly became graph-based tracking of individual tree crowns across multiple orthomosaics.
- Temporal monitoring needed to handle:
  - one-to-one matches;
  - one-to-many splits;
  - many-to-one merges;
  - disappearances;
  - appearances;
  - ambiguous overlaps.
- IoU matrices were used to expose split/merge structure.
- Backward edges and skip edges such as OM1 to OM3 were considered for cases where a crown is missed in OM2.
- The team worried that cycles and backward links could violate temporal causality or create ambiguous object histories.

Thesis relevance:

- This is the conceptual birth of the graph-tracking contribution.
- The final paper should frame tracking-by-detection with graph association as a response to these ecological/segmentation cases.

## 2025-08-12: Transition Statistics and Graph Diagnostics

Source: [2025-08-12_01_12Aug25.md](<extracted text markdown/2025-08-12_01_12Aug25.md>)

Important points:

- Transition summaries were calculated for consecutive OM pairs.
- Metrics included:
  - one-to-one matches;
  - splits;
  - merges;
  - disappearances;
  - appearances;
  - ambiguous overlaps;
  - total crowns per OM;
  - total edges;
  - mean IoU;
  - in/out degree.
- Example transitions showed increasing crowns and many split/merge/appearance events.

Thesis relevance:

- These metrics are useful for a tracking diagnostics table.
- They show that the graph was not just conceptual; it was used to quantify temporal association complexity.

## 2025-09-02: Low IoU in Crowded Regions and Clickable Visualization

Source: [2025-09-02_01_2Sep25.md](<extracted text markdown/2025-09-02_01_2Sep25.md>)

Important points:

- The graph was clarified to track crowns, not hidden "trees inside crowns."
- Crowded regions caused multiple red crowns to overlap a blue crown while IoU remained low.
- Multiple IoU threshold regimes were proposed:
  - high IoU one-to-one;
  - moderate IoU split;
  - weaker overlaps requiring other logic.
- Clickable visualization was introduced to inspect tracking cases.

Thesis relevance:

- Supports the need for visualization as QA, not just presentation.
- Reinforces why IoU alone fails in dense canopy or changing crown boundaries.

## 2025-09-12: Similarity Score Beyond IoU

Source: [2025-09-12_01_12Sep25.md](<extracted text markdown/2025-09-12_01_12Sep25.md>)

Important points:

- Wrong matches in crowded regions motivated a weighted similarity score.
- Features included:
  - spatial similarity from centroid distance;
  - area similarity;
  - compactness;
  - eccentricity;
  - shape similarity.
- Tracking statistics were reported:
  - total detected trees: 626;
  - total edges: 476;
  - average chain length: 4.17;
  - maximum chain length: 5;
  - 80 length-5 trees.
- Too many C6 crowns occurred when IoU threshold was too low.
- Conclusion: rely on a combined similarity score, not only IoU.
- A web app for tracking visualization was shown.

Thesis relevance:

- Directly maps to the final graph edge features.
- The combined similarity score should be a central method component.

## 2025-09-23: Threshold Sweeps, Graph Components, and Parameter Consistency

Source: [2025-09-23_01_23Sep25.md](<extracted text markdown/2025-09-23_01_23Sep25.md>)

Important points:

- A threshold sweep over similarity scores showed edge counts collapsing as the threshold increased.
- At threshold 0.7, there were 798 edges from 626 nodes; at lower thresholds, tens of thousands of edges appeared.
- Large graph components with diameter up to 29 indicated spurious weak-edge propagation.
- Proposed fixes:
  - prune connected components based on edge weights;
  - keep only strongest k edges to the next OM;
  - iteratively remove edges until component diameter is acceptable.
- Detectree2 parameter consistency was proposed:
  - keep tile size, buffer, min/max crown size, and simplification fixed;
  - adjust confidence and merging thresholds per mosaic to stabilize crown counts.
- Crown detection should be integrated into the pipeline for joint detection/matching evaluation.

Thesis relevance:

- Shows why graph pruning and thresholding matter.
- Explains why low thresholds can overconnect graphs.
- Provides a strong methodological argument for controlling detector parameters across time.

## 2025-10-07: Conditional Threshold Matching

Source: [2025-10-07_01_7Oct25.md](<extracted text markdown/2025-10-07_01_7Oct25.md>)

Important points:

- Conditional threshold matching was formalized.
- Candidate crown pairs within `base_max_dist` were assigned metrics:
  - IoU;
  - overlap ratios;
  - centroid distance;
  - area ratio;
  - containment flags;
  - baseline similarity.
- Low-signal pairs were dropped with `min_base_similarity` and `overlap_gate`.
- Edges were classified as:
  - containment;
  - one-to-one;
  - split;
  - merge;
  - partial overlap;
  - proximity;
  - none.
- Cases were processed in priority order.
- Initial result had low total edges but low maximum graph diameter, indicating conservative matching.

Thesis relevance:

- This is very close to the current `TreeTrackingGraph` method.
- Case-specific thresholds and scoring should be included in methodology.

## 2025-10-21: Shuffle Connectivity, Virtual Edges, and Failure of Over-Complex Graphs

Source: [2025-10-21_01_21Oct25.md](<extracted text markdown/2025-10-21_01_21Oct25.md>)

Important points:

- Rotational shuffles were tested: run matching on cyclic permutations of orthomosaic order and vote on edges.
- This performed poorly, with very low match rates and maximum chain length only 2 or 3.
- Virtual edges between non-consecutive OMs were tested.
- Virtual edges caused error propagation and allowed edges with similarity below intended thresholds.
- Future idea shifted back toward simpler high-confidence matching and Detectree2 parameter tuning.

Thesis relevance:

- Important negative result: more complex graph connectivity was not always better.
- Supports the final conservative design: consecutive-date graph plus controlled gap filling, rather than arbitrary virtual links.

## 2025-10-28: Detectree2 Flexi Model, Adaptive Thresholding, and Multi-Threshold Idea

Source: [2025-10-28_01_28Oct25.md](<extracted text markdown/2025-10-28_01_28Oct25.md>)

Important points:

- Rotational/mountain-shaped graphs were explored for crown growth and recession patterns.
- Detectree2 stitching and cleaning were documented:
  - concatenate tile detections;
  - remove duplicate overlapping crowns;
  - drop low-confidence polygons;
  - simplify geometries.
- Geometry-related knobs should remain consistent across dates:
  - simplify tolerance;
  - IoU merge threshold;
  - tile size/buffer.
- Confidence threshold can vary per orthomosaic to stabilize detection counts.
- New `250312_flexi.pth` model was found more consistent than earlier `urban_cambridge` weights.
- Confidence sweeps showed that crown count could be controlled by varying confidence threshold.
- Median crown-count threshold selection was tested.
- Key future idea:
  - store detections at multiple confidence thresholds with each orthomosaic;
  - during tracking, fall back to lower confidence if a node has no outgoing edge;
  - later drop chains with low average confidence if needed.

Thesis relevance:

- This is the direct origin of multi-threshold crown stores.
- The flexi model and threshold sweeps are important for the detection-method narrative.

## 2025-11-11: Detectree2 Mechanics, Chain Typology, and Pruning Rules

Source: [2025-11-11_01_11_Nov_2025.md](<extracted text markdown/2025-11-11_01_11_Nov_2025.md>)

Important points:

- Detectree2 was explained as Mask R-CNN/Detectron2 instance segmentation.
- Useful features for crown detection were identified:
  - color;
  - texture;
  - shape/edge;
  - scale/size;
  - spatial context;
  - shadow/lighting.
- Chain patterns were visually analyzed:
  - full chain width 1: reliable, often evergreen;
  - smaller chains: deciduous or shadow/edge cases;
  - full chain with width > 1: split/merge ambiguity.
- Lower confidence thresholds recovered some longer chains but did not fix edge/shadow failures.
- The `250312_flexi.pth` model reduced branching and produced more consistent crown sizes.
- Pruning rules:
  - if one candidate has similarity > 0.5, select it;
  - for overlaps within OM, select larger crown when overlap fraction is high.
- Results after pruning:
  - 21 full width-1 length-5 chains;
  - 14 full branching length-5 chains;
  - 33 partial length 3-4 chains;
  - 39 partial length-2 chains.

Thesis relevance:

- Chain typology should appear in the method/results.
- The interpretation that full width-1 chains are high confidence, while branching/smaller chains need filtering, remains central.

## 2025-12-09: Split/Merge Successor Selection and Shedding Logic

Source: [2025-12-09_01_9_dec_2025.md](<extracted text markdown/2025-12-09_01_9_dec_2025.md>)

Important points:

- Split and merge cases were handled with structured successor logic.
- Split logic:
  - identify one-to-many nodes;
  - select top successor candidates by similarity;
  - use forward similarity from those successors to choose continuity.
- Merge logic:
  - compare merging candidates with their previous/back node;
  - choose the candidate with highest similarity.
- Containment rules used:
  - contained geometry;
  - area balance;
  - IoU;
  - centroid proximity;
  - overlap fractions.
- A shedding event was defined when:
  - previous crown contains current crown;
  - high overlap of current inside previous;
  - area drops substantially.
- In shedding, the smaller contained polygon may be preferred as true successor to avoid jumping to a nearby larger crown.

Thesis relevance:

- This links tracking with phenology: crown shrinkage can be a biological/seasonal event, not only segmentation error.
- The paper can mention that split/merge and containment rules are designed to preserve identity under both segmentation ambiguity and canopy-state change.

## 2026-01-07: Lookahead Split Voting and Google Earth Visualization

Source: [2026-01-07_01_7_Jan_26.md](<extracted text markdown/2026-01-07_01_7_Jan_26.md>)

Important points:

- Split branch finalization used lookahead over a window of future OMs.
- For each candidate successor branch, the algorithm counted whether later OMs supported persistent split structure or collapse to single continuity.
- `split_support` and `single_support` were compared.
- Crowns and orthomosaic outputs were overlaid in Google Earth.
- Google Earth was considered for ground truthing.

Thesis relevance:

- Lookahead split voting is useful to mention as explored split/merge logic.
- Google Earth/QField visualization connects to field validation and species labeling.

## 2026-01-14: External GPS/RTK Consideration

Source: [2026-01-14_01_14_Jan_26.md](<extracted text markdown/2026-01-14_01_14_Jan_26.md>)

Important points:

- External GPS hardware and RTK GNSS were investigated.
- Phone GNSS was estimated at 3-10 m; non-RTK external GNSS at 0.3-1 m; RTK at 1-3 cm.
- RTK modules were expensive and difficult to source.
- Affordable Android-compatible GPS receivers offered only around 2 m accuracy, not enough improvement to justify major effort.

Thesis relevance:

- Supports the practical need for image/orthomosaic alignment instead of relying on field GPS precision.
- Can be mentioned in background if discussing why residual alignment is necessary.

## 2026-01-21: Consensus Crowns and Phase Cross-Correlation Alignment

Source: [2026-01-21_01_21_Jan_26.md](<extracted text markdown/2026-01-21_01_21_Jan_26.md>)

Important points:

- Consensus crown requirements were formalized:
  - too large includes ground/other trees;
  - too small loses true crown pixels, especially where deciduous changes appear first.
- Possible consensus geometries:
  - medoid;
  - intersection core;
  - union shrink.
- Misalignment caused all consensus methods to degrade.
- Crown-based alignment was judged too sensitive to detection quality.
- Phase cross-correlation on grayscale orthomosaic pixels was introduced.
- PCC estimates translation through Fourier phase and is robust to brightness changes.
- Cumulative shifts across consecutive OMs were applied.
- For most OMs shifts were within around +/-2 m, but one OM had around +9 m, +3 m.
- PCC improved OM4 to OM5 tracking and produced 9 more full chains.
- Tracking improved:
  - edges: 260 to 269;
  - match rate: 0.818 to 0.846;
  - full chains: 23 to 32;
  - singletons: 128 to 101.

Thesis relevance:

- This is one of the most important method-development dates.
- It justifies consensus crowns and phase-correlation alignment as paired contributions.

## 2026-02-04: PCC Formalization, Ground-Truth Comparison, and Data Storage

Source: [2026-02-04_01_4_Feb_2026.md](<extracted text markdown/2026-02-04_01_4_Feb_2026.md>)

Important points:

- PCC was formalized:
  - pure translation estimation;
  - low-resolution grayscale orthomosaics;
  - Fourier transforms differ by linear phase ramp;
  - inverse normalized cross-power spectrum peak gives translation.
- Consensus crowns were compared to ground truth.
- Error examples:
  - 5 cases where tracked trees were not in ground truth;
  - 1 split case;
  - 1 merge case with 3 ground-truth crowns merged into one tracking crown.
- IIT Delhi SIT and LHC data were uploaded to CSE Filer.

Thesis relevance:

- Provides language for explaining PCC in the method.
- Ground-truth comparison/error taxonomy should inform the planned manual validation section.

## 2026-02-11: Manual Phenology Labels and Error Taxonomy

Source: [2026-02-11_01_11_Feb_2026.md](<extracted text markdown/2026-02-11_01_11_Feb_2026.md>)

Important points:

- Manual condition categories were defined:
  - dull green;
  - green leaves bloom;
  - green leaves starts;
  - leaf shedding;
  - yellow leaves;
  - wrong detection;
  - flowering.
- Full chains were annotated with these statuses across seven OMs.
- Reference images were collected for label policy.
- Temporal interpretation:
  - early OMs mixed conditions;
  - middle OMs peak healthy growth;
  - late OMs showed some stress/dull leaves.
- Consensus crown error taxonomy:
  - missing detection;
  - false positive;
  - split error;
  - merge error;
  - poor overlap;
  - boundary error.
- Future work included canopy gap measurement, greenness labeling policy, and QField/Google Earth marking format.

Thesis relevance:

- This supports manual validation and phenology label definitions.
- The error taxonomy should be reused for the tracking-validation subsection.

## 2026-02-17: Phenology Metrics, Texture, QC, and Flower Detection Literature

Source: [2026-02-17_01_17_Feb_2026.md](<extracted text markdown/2026-02-17_01_17_Feb_2026.md>)

Important points:

- Canopy gap and variation heatmaps were explored using grayscale conversion, Gaussian smoothing, and normalization.
- Formal RGB phenology features appeared:
  - GCC;
  - RCC;
  - vegetation fraction.
- Texture/heterogeneity features were motivated:
  - gray standard deviation;
  - smoothed gray statistics;
  - gray entropy;
  - energy.
- QC features:
  - valid pixel fraction;
  - shadow fraction;
  - Laplacian variance for blur.
- Within-date z-score normalization was proposed to reduce illumination/camera/exposure effects.
- Flower detection literature was reviewed:
  - explicit flower-pixel thresholding;
  - flower indices;
  - correlation with field assessments.

Thesis relevance:

- This is the basis for the phenology feature extraction method.
- QC and normalization are important for making RGB phenology defensible.

## 2026-02-25: Multi-Threshold Tracking, Partial Chains, and Phenology Clustering

Source: [2026-02-25_01_25_Feb_2026.md](<extracted text markdown/2026-02-25_01_25_Feb_2026.md>)

Important points:

- Relaxed matching produced more edges and high match rates, but not many more full chains.
- Missing trees between consecutive pairs were not the same trees each time, so full-chain coverage stayed low.
- Multi-threshold crowns were developed:
  - originally used lower thresholds only for gap filling;
  - then used all threshold layers in graph construction, selecting best candidates by similarity;
  - min threshold was lowered to 0.15.
- This produced 35 full chains across 7 SIT orthomosaics, +15 over the previous approach.
- A structured metadata format for multi-threshold crowns was defined.
- `MultiThresholdTreeTracker` extended `TreeTrackingGraph`.
- The base threshold was fixed at `conf_0p45` for OM1 instead of adaptive median-count selection.
- Partial-chain consensus was introduced:
  - full 7-date chains were not necessary;
  - shorter high-quality chains could generate consensus crowns;
  - consensus crops can still be extracted for missing dates.
- Result: 79 consensus crowns, +44.
- Partial-chain quality checks included all one-to-one edges and similarity > 0.65.
- Reverse alignment was used for extracting image patches from raw orthomosaics.
- Ground-truth comparison at IoU 0.3:
  - TP: 41;
  - missing detection: 68;
  - false positive: 20;
  - split error: 9;
  - merge error: 7;
  - poor overlap: 3;
  - boundary error: 3.
- Phenology clustering used 5 metrics over 7 OMs, normalized and concatenated into time-series vectors, then clustered using Ward linkage.
- LHC work started in parallel to SIT.

Thesis relevance:

- This is the strongest origin date for final multi-threshold and partial-chain consensus methods.
- The reverse-alignment step is a key technical detail for crop extraction.
- Phenology clustering can be treated as exploratory analysis or a secondary results figure.

## 2026-03-11: LHC Tracking and Leaf-Shed Scoring

Source: [2026-03-11_01_11_March_2026.md](<extracted text markdown/2026-03-11_01_11_March_2026.md>)

Important points:

- LHC tracking results were reported:
  - match rates: 1->2 = 0.559, 2->3 = 0.810, 3->4 = 0.830;
  - high-quality chains: 16;
  - smaller chains length >= 3: 14;
  - average chain length: 2.02;
  - overall match rate: 0.748.
- Leaf-shed/no-leaf-shed signals were formalized:
  - continuous deciduousness score in [0, 1];
  - binary deciduous vs evergreen;
  - per-date phenophase label.
- Signals:
  - vegetation fraction as primary leaf-presence signal;
  - GCC as secondary greenness signal;
  - Laplacian variance as texture/structure support.
- Component scores:
  - vegetation amplitude;
  - depth of minimum vegetation;
  - GCC amplitude;
  - texture amplitude.
- Deciduousness score used weighted combination:
  - 0.35 vegetation amplitude;
  - 0.30 depth;
  - 0.25 GCC amplitude;
  - 0.10 texture.
- Initial deciduous threshold was 0.40 in this deck.
- Phenophase:
  - leaf-on above normalized vegetation 0.65;
  - leaf-off below 0.35;
  - transitioning intermediate;
  - stable where no active transition is indicated.
- Organised code was introduced.

Thesis relevance:

- This is the clearest source for the leaf-shed scoring method.
- Current code may use different thresholds, so final paper should state the code-run values, but the conceptual structure is stable.

## 2026-03-17: Formal Phenology Metrics and Flowering Indices

Source: [2026-03-17_01_17_March_2026.md](<extracted text markdown/2026-03-17_01_17_March_2026.md>)

Important points:

- Phenology metrics were formalized around:
  - pixel validity and masks;
  - GCC, RCC, ExG;
  - HSV vegetation and shadow fractions;
  - grayscale texture;
  - normalization;
  - temporal descriptors;
  - time series.
- Literature review found no standard crown-level "deciduousness index"; indirect amplitude-based greenness thresholds are more common.
- LHC 2025-12-09 misalignment was again noted as a major issue.
- Flower detection discussion included:
  - pigment-specific indices such as NDYI and EBI;
  - GLCM texture to separate petal clusters from leaf glint;
  - bloom coverage percentage per crown.
- NDYI was described for yellow flowers.
- GLCM energy was proposed as a texture discriminator for petals versus vegetation/branches.
- SIT 50 m and 80 m Semal examples were compared.

Thesis relevance:

- Supports a phenology-feature table.
- Flowering should be included as part of broader trait inference, but probably not the central validated result unless final results are strong.

## 2026-04-01: Ground Visit Missing Trees, Consensus Deduplication, and Flower Logic

Source: [2026-04-01_01_1_Apr_2026.md](<extracted text markdown/2026-04-01_01_1_Apr_2026.md>)

Important points:

- Ground-visit missing trees were analyzed.
- Some trees missed earlier became detectable when more orthomosaics were added.
- Some large/small trees remained undetected if Detectree2 failed across most OMs.
- Running on 14 OMs with `min_partial_len=5` produced many overlapping consensus crowns.
- Increasing `min_partial_len` reduced overlaps but lost crowns.
- Consensus-level deduplication was introduced:
  - IoU removal;
  - containment removal;
  - containment buffer for almost-contained crowns.
- Finalized parameters in this deck:
  - `DEDUP_IOU_THRESHOLD = 0.85`;
  - `DEDUP_CONTAINMENT_BUFFER = 5.0`.
- Flower detection:
  - RGB thresholds could detect flowering vs non-flowering canopies;
  - red detection confused yellow flowers;
  - VARI > 0.12 was used to identify green and bypass flower logic;
  - hierarchical priority tested yellow first using NDYI;
  - Semal/red required low GCC (< 0.28) to avoid yellow false positives;
  - crown subdivided into 10 x 10 cells, with >45 percent agreement for final classification.

Thesis relevance:

- Deduplication is necessary in the consensus-crown method.
- Ground-visit analysis shows why adding more OMs improves recovery.
- Flower logic can be framed as exploratory crown-level trait extraction.

## 2026-04-08: Interactive Viewer and Satellite Harmonic Bridge

Source: [2026-04-08_01_8_Apr_2026.md](<extracted text markdown/2026-04-08_01_8_Apr_2026.md>)

Important points:

- A static clickable orthomosaic time-series viewer was built.
- Viewer output includes:
  - `index.html`;
  - base underlay PNG;
  - crown pixel GeoJSON;
  - crop images per crown/date.
- Viewer was hosted with `python http.server`.
- LHC leaf-shed classifier was shown.
- Separate labelers/classifiers for LHC and SIT were considered.
- Mayank's satellite scripts were reviewed:
  - HLS/NDVI time series;
  - cloud/shadow/snow/water masking;
  - 16-day median composites;
  - harmonic regression;
  - reconstructed 365-day NDVI curves.
- Future bridge:
  - align OM1 to satellite image;
  - sample satellite data at consensus crown centroids;
  - use harmonic coefficients and sampled biweekly curves.

Thesis relevance:

- Interactive visualization should be described as a QA/validation tool.
- This marks the first concrete satellite bridge through consensus crowns.

## 2026-04-15 and 2026-04-22: GEE Setup, Sentinel-2 Harmonics, and Satellite Phenology

Sources:

- [2026-04-15_01_15_April_2026.md](<extracted text markdown/2026-04-15_01_15_April_2026.md>)
- [2026-04-22_01_22_Apr_2026.md](<extracted text markdown/2026-04-22_01_22_Apr_2026.md>)

Important points:

- Google Earth Engine project `satellite-phenology` was set up.
- HLS 30 m data was replaced with Sentinel-2 SR 10 m data.
- QA60 masking and 20 percent cloud prefilter were used.
- Precise SIT and LHC bounding boxes from QGIS:
  - SIT: `[77.190011007, 28.544543935, 77.192435230, 28.546753777]`;
  - LHC: `[77.191979006, 28.542881228, 77.193985263, 28.544842795]`.
- Harmonic model:
  - `NDVI(t) = a0 + a1*t + a2*sin(2*pi*t) + a3*cos(2*pi*t) + a4*sin(4*pi*t) + a5*cos(4*pi*t)`.
- 365-day phenology script identifies leaf-shed trough and flowering peak.
- Drone survey dates are annotated onto satellite-derived curves.
- Harmonic coefficients were planned as model inputs.
- Drone results were intended as labels for satellite machine learning.
- Viewer and ground visit annotations were included.
- Future work included LeavesFresh, LeavesMature, LeavesOld classifiers.

Thesis relevance:

- This provides the first concrete satellite method subsection before the later embedding work.
- For the final paper, harmonics may be a subsection under cross-scale experiments or an earlier baseline before GEE embeddings.

## 2026-05-06: Satellite-Drone Phenology Paper Review and Flower Regression Framing

Source: [2026-05-06_01_6_May_2026.md](<extracted text markdown/2026-05-06_01_6_May_2026.md>)

Important points:

- Reviewed a workflow where drone images label coarse satellite pixels, then satellite time series predict fine-grained phenological signals.
- Example task: Random Forest regression predicting flower coverage percentage per satellite pixel.
- Drone labels were derived from very high-resolution RGB orthomosaics by classifying pixels into Flower, Green Vegetation, Soil, Road, Wood, Shadow.
- Satellite features included GNDVI and EBI time series at phenologically meaningful dates.
- Features included:
  - raw spectral values at key dates;
  - change-vector magnitudes/directions;
  - delta features.
- Important phenological dates:
  - baseline;
  - start;
  - target date;
  - peak flowering;
  - end/recovery.
- Vector B/re-greening was used to avoid mistaking drought/fire for flowering.

Thesis relevance:

- Gives a conceptual framework for cross-scale inference.
- Even if not directly implemented, it supports the argument that drone-derived crown labels can supervise satellite-scale models.

## 2026-05-13: Master GeoJSON Format

Source: [2026-05-13_01_13_May_2026.md](<extracted text markdown/2026-05-13_01_13_May_2026.md>)

Important points:

- Master GeoJSON format was discussed.
- A simpler GeoJSON format was also considered.
- Although extracted text is sparse, this aligns with the current `tree_master_geojson.geojson` pipeline output.

Thesis relevance:

- The master GeoJSON is important as the final data product:
  - geometry;
  - tracking metadata;
  - phenology observations;
  - field labels;
  - assets/crop paths.

## 2026-05-19: Final Pipeline Formalization and Dataset Inventory

Source: [2026-05-19_01_19_May_2026.md](<extracted text markdown/2026-05-19_01_19_May_2026.md>)

Important points:

- Sanjay Van data was added to the phenology mapping sheet.
- New species added included Kanju, Prosopis juliflora, Subabool, and Teak.
- Existing species included Amaltas, Bamboo, Banyan, Karanj, Native acacia, Neem, Peepal.
- The final pipeline was formalized:
  - Step 0: discover OMs and write `pipeline_config.json`;
  - Step 1: Detectree2 multi-threshold crown detection;
  - Step 2: crown tracking to `consensus_crowns_complete_all.gpkg`;
  - Step 3: phenology analysis to `tree_master_geojson.geojson`;
  - Step 4: interactive viewer.
- Step 2 details:
  - PCC alignment on `conf_0p65`;
  - graph build;
  - full chains;
  - branching backbones;
  - partial chains with length >= 5 and one-to-one ratio >= 0.9;
  - deduplication.
- Step 3:
  - extract patch;
  - compute vegetation indices;
  - normalize per crown across time;
  - score leaf-shed intensity and assign phenophase.
- Dataset inventory:
  - LHC: 13 OMs;
  - SIT: 19 OMs;
  - Sanjay Van: 8 OMs in spots 1-3, 6 OMs in spot 4.
- CSE Filer scripts and Docker/NFS/rsync transfer were used for data movement.

Thesis relevance:

- This date should anchor the final pipeline method.
- It confirms current LHC/SIT full-data counts and Sanjay Van availability.

## 2026-05-27: Species/Trait Labels and Sentinel-2 Classifier Baselines

Source: [2026-05-27_01_27_May_2026.md](<extracted text markdown/2026-05-27_01_27_May_2026.md>)

Important points:

- Clean species-labeled crowns: 372.
- Major species counts included:
  - Prosopis juliflora: 53;
  - Neem: 51;
  - Ashok: 29;
  - Amaltas: 25;
  - Peepal: 24;
  - Pilkhan: 23.
- Ambiguous/unknown labels: 46 crowns.
- Classifier configs included:
  - evergreen/semi-evergreen/deciduous multiclass;
  - deciduous vs rest;
  - Acacia vs non-Acacia;
  - yellow showy strict;
  - yellow broad;
  - showy flower vs rest;
  - red/orange showy.
- Label class distributions were reported:
  - ESD: evergreen 75, semi-evergreen 152, deciduous 115;
  - deciduous: 115 positive, 227 not deciduous;
  - Acacia: 65 positive, 326 non-Acacia;
  - yellow strict: 40 positive;
  - yellow broad: 100 positive;
  - red showy: 12 positive;
  - showy flower: 63 positive.
- Sentinel-2 features:
  - bands B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12;
  - indices NDVI, GNDVI, NDRE, NDMI, NBR, EVI, visible greenness, redness.
- Temporal features:
  - winter/dry;
  - pre-monsoon;
  - monsoon;
  - post-monsoon;
  - seasonal medians;
  - amplitude features.
- Baseline GEE ESD random split:
  - train accuracy 99.2 percent;
  - test accuracy about 58 percent.
- Leave-area-out accuracy dropped strongly:
  - SIT: 36 percent;
  - A3: 29 percent;
  - A4: 34 percent.
- Leave-species-out also performed poorly:
  - Amaltas: 12 percent;
  - Neem: 20 percent;
  - Ashok: 14 percent.

Thesis relevance:

- This is the key baseline satellite/species experiment before embeddings.
- It shows the danger of random split overconfidence and the importance of spatially blocked validation.
- The final paper should include satellite/species as main body, but carefully frame random split vs spatial generalization.

## 2026-06-03 and 2026-06-10: Sanjay Van Augmentation, Crown Size Limits, and GEE Embeddings

Sources:

- [2026-06-03_01_3_Jun_2026.md](<extracted text markdown/2026-06-03_01_3_Jun_2026.md>)
- [2026-06-10_01_10_Jun_2026.md](<extracted text markdown/2026-06-10_01_10_Jun_2026.md>)

Important points:

- Sanjay Van clustering augmentation for Acacia/non-Acacia:
  - 108 SV crowns had both cluster-derived and ground-truth labels;
  - 2,071 remaining SV crowns were cluster-labeled;
  - 362 GT-only crowns were available.
- Pseudo-label agreement:
  - 89/108 = 82.4 percent overall agreement;
  - non-Acacia recall 89.1 percent;
  - Acacia recall 77.4 percent;
  - 14 Acacia crowns were mislabeled as non-Acacia.
- Augmented labels did not clearly improve models:
  - baseline ExtraTrees balanced accuracy 0.701, F1 0.707;
  - augmented ExtraTrees balanced accuracy 0.670, F1 0.556;
  - augmented RF balanced accuracy 0.700, F1 0.543;
  - baseline RF balanced accuracy 0.636, F1 0.595.
- Crown-size analysis:
  - mean crown area 45.5 m2;
  - median 28.1 m2;
  - one Sentinel-2 pixel is 100 m2.
- A1-A5 areas had larger crowns, median around 55-65 m2, with 21-33 percent above one pixel.
- Sanjay Van S1-S3 were almost entirely sub-pixel:
  - S1 median 21.8 m2, 1.2 percent above 100 m2;
  - S2 median 21.3 m2, 0.7 percent above 100 m2;
  - S3 median 21.6 m2, 0.5 percent above 100 m2.
- Area filtering removed many labels:
  - filtering at 20 m2 loses 31 percent of crowns;
  - 50 m2 loses 74 percent;
  - 100 m2 loses 91 percent.
- Pixel-level classifiers had weak crown-level performance:
  - mean crown-level accuracy about 50.04 percent;
  - OOB accuracy 69.20 percent.
- Jan-May features were added on top of seasonal features.
- Google Earth Engine Satellite Embedding V1:
  - 2024 annual embedding;
  - 64-dimensional vector A00-A63;
  - tested original crown polygon and 20 m centroid buffer.
- GEE embeddings performed strongly:
  - yellow broad random split around 0.805 with 20 m buffer and 0.795 original;
  - yellow strict random split around 0.764 with 20 m buffer;
  - showy flower random split around 0.631-0.699 depending on geometry;
  - Acacia random split around 0.939 with 20 m buffer;
  - Acacia original geometry accuracy 0.924, balanced accuracy 0.934, macro F1 0.880.
- Holdouts for Acacia were mostly around 0.65-0.78.
- User guides were documented:
  - collecting drone imagery;
  - building orthomosaic in WebODM;
  - running Detectree2;
  - orthomosaic printout/crown IDs;
  - QField crown annotation.

Thesis relevance:

- These are the strongest satellite/species results in the weekly decks.
- The crown-size/sub-pixel analysis is crucial: it explains why individual drone crowns are not directly resolved by Sentinel-2 pixels.
- GEE embeddings should be emphasized over weak pixel-level and pseudo-label augmentation results.
- Acacia and yellow/showy phenology appear to be the strongest candidate classifier tasks.

## Thesis/Paper Implications from Weekly Meetings

### Main Contribution Confirmed

The weekly decks strongly support the selected framing:

> Tracking individual tree phenology requires converting unstable per-date UAV crown detections into temporally stable crown identities through alignment, graph association, gap recovery, consensus crowns, and crown-level feature extraction.

### Strongest Method Story

The project's method should be written as a sequence of hard-earned design choices:

1. WebODM orthomosaics and CRS handling made repeated UAV surveys geospatially usable.
2. DeepForest was useful as a baseline but too coarse because bounding boxes and false positives were not enough for crown phenology.
3. Detectree2 gave georeferenced polygon crowns, but per-date detection was incomplete and unstable.
4. Hungarian matching was a useful baseline but failed for split/merge/missing detections.
5. Graph-based association with case-specific edge scoring was developed to preserve ambiguity while extracting reliable chains.
6. Alignment moved from crown-centroid/affine approaches to image-based phase cross-correlation, then tiled PCC for robustness.
7. Multi-threshold crown stores were introduced because fixed confidence thresholds either missed crowns or produced noisy detections.
8. Full chains alone were too restrictive, so high-quality partial chains were used for consensus crown construction.
9. Consensus crowns became stable ROIs for extracting RGB, texture, QC, and phenology features.
10. Master GeoJSON and viewer outputs made the pipeline inspectable and field-linkable.
11. Field-verified labels and GEE embeddings extended the crown identity layer into species/trait and satellite-scale inference.

### Results to Prioritize Later

When final full runs are available, the paper should prioritize:

- detection evaluation table using ground truth;
- alignment before/after metrics or visual overlays;
- tracking chain table for LHC and SIT full runs;
- manual tracking-validation subset;
- consensus crown count and deduplication summary;
- phenology feature examples and maps;
- field-verified label statistics;
- GEE embedding classifier results, especially Acacia and yellow/showy traits;
- spatial holdout results where available.

### Results to Treat Carefully

- Pseudo-label augmentation in Sanjay Van has useful analysis but may not improve classifier performance.
- Pixel-level classifiers are weak at crown level in the current slides.
- Random-split classifier results can be optimistic; spatial holdouts are more honest.
- DSM/CHM/tree height was reviewed but not implemented as the main project.
- Flower detection logic is promising but should be included only if results are solid enough.

### Suggested Paper Updates

- Include a short "Tracking-by-detection development" framing in the Method or Discussion: DeepForest/Hungarian were baselines, graph association is the resolved method.
- Add an explicit "Manual tracking validation subset" result subsection.
- Add "Interactive visualization and QA" under Method or Results.
- In satellite/species, separate:
  - Sentinel-2 seasonal-feature baselines;
  - GEE embedding experiments;
  - spatial generalization/holdout evaluation.
- Add a short discussion of crown size versus Sentinel-2 pixel size to explain scale mismatch.

