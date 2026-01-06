# Comprehensive Notebook Analysis for Report
**Drone-Based Tree Phenology Monitoring Project (July–November 2025)**

This document synthesizes key findings from all notebooks to support report writing.

---

## 1. Study Sites and Datasets

### Orthomosaics Analyzed
- **5 Orthomosaics** labeled OM1 through OM5 (`sit_om1.tif` through `sit_om5.tif`)
- Located in `input/input_om/` directory
- Site: **SIT (Student Information Technology) Front Area** at IIT Delhi campus
- Temporal coverage: Multiple drone flights across July–November 2025

### Crown Detection Dataset
- Crown polygons stored as GeoPackages (`.gpkg` files) in `output/input_crowns/`
- Files named `OM1.gpkg` through `OM5.gpkg`
- **Total crowns detected**: 408 unique crown detections across all 5 OMs
- Breakdown by OM:
  - OM1: 82 crowns
  - OM2: 78 crowns  
  - OM3: 78 crowns
  - OM4: 78 crowns
  - OM5: 80 crowns

---

## 2. Tree Crown Detection Pipeline (detectree_parth.ipynb)

### Detectree2 Model
- **Model used**: `250312_flexi.pth` (pre-trained Detectron2-based model)
- Located in `input/detectree_models/`
- Framework: Detectron2 with Mask R-CNN architecture
- Task: Instance segmentation of tree crowns from RGB orthomosaics

### Detection Workflow
1. **Tiling**: Orthomosaics divided into 45×45m tiles with 20m buffer for edge handling
2. **Inference**: Detectron2 DefaultPredictor runs on each tile
3. **Reprojection**: Tile-based predictions converted to geo-referenced GeoJSON
4. **Stitching**: Individual tile predictions merged into single layer per OM
5. **Cleaning**: 
   - Invalid geometries removed
   - Geometry simplification (tolerance = 0.3m)
   - Overlapping crowns resolved using IoU threshold (0.7) and confidence filtering

### Adaptive Confidence Thresholding
**Challenge**: Different orthomosaics produce varying crown counts depending on detection confidence threshold.

**Solution**: Adaptive per-OM thresholding
- **Confidence sweep**: Test thresholds [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
- **Target metric**: Global median crown count across all OM×confidence combinations
- **Selection rule**: For each OM, choose threshold yielding count closest to global median
- **Result**: Balanced crown counts across orthomosaics (avg ~81.6 crowns per OM)

**Fixed parameters**:
- IoU threshold for overlap removal: 0.7
- Geometry simplification: 0.3m tolerance

---

## 3. Graph-Based Crown Tracking Framework

### 3.1 Conceptual Foundation (tree_tracking_graph_demo.ipynb)

**Core Idea**: Represent tree tracking as a directed graph
- **Nodes**: Individual crown detections (OM_id, crown_index)
- **Edges**: Temporal correspondences between crowns across OMs
- **Edge attributes**: IoU, similarity scores, overlap ratios, centroid distances

**Tracking Cases Demonstrated**:
1. **1-to-1 match**: Same tree detected in consecutive OMs with high overlap
2. **1-to-many split**: One crown in OM_t splits into multiple crowns in OM_{t+1} (over-segmentation)
3. **Many-to-1 merge**: Multiple crowns in OM_t merge into one in OM_{t+1} (under-segmentation)
4. **Disappear**: Crown detected in OM_t but not OM_{t+1} (deciduous leaf-off, detection failure)
5. **Appear**: New crown in OM_{t+1} not present in OM_t (new detection, seasonal appearance)
6. **Ambiguous**: Multiple plausible matches with similar scores

**Forward-only vs Bidirectional Tracking**:
- **Forward-only**: Edges only from OM_i → OM_j where i < j (preserves temporal causality, acyclic)
- **Bidirectional**: Allows edges OM_i ↔ OM_j (can introduce cycles, higher ambiguity but may recover missed detections)
- **Project choice**: Forward-only consecutive matching (OM_i → OM_{i+1})

---

### 3.2 TreeTrackingGraph Class Architecture

**Core Components**:
- **File discovery**: Automatic pairing of crown GeoPackages with orthomosaic TIFFs by numeric ID
- **Crown attributes computation**: For each crown, compute:
  - Centroid (x, y coordinates)
  - Area (m²)
  - Perimeter (m)
  - Compactness: 4πA/P²
  - Eccentricity: minor_axis/major_axis from minimum rotated rectangle
  - Aspect ratio from bounding box
  - Bounds (minx, miny, maxx, maxy)

- **Match graph building**: NetworkX directed graph with crown nodes and match edges
- **Image extraction** (optional): RGB patches extracted from orthomosaics for each crown using rasterio mask

---

## 4. Spatial Alignment Across Orthomosaics

### 4.1 Problem: Georeferencing Drift
- Consecutive orthomosaics show **positional misalignment** (translation errors)
- Caused by: GPS drift, lack of ground control points, photogrammetry inaccuracies
- **Impact**: Reduced IoU between corresponding crowns, false negatives in matching

### 4.2 Alignment Solution (crown_tracking_4Nov, crown_tracking_10Nov)

**Consecutive Alignment Approach**:
1. **Reference OM**: OM1 treated as fixed reference (no shift)
2. **Pairwise alignment**: Each OM_i aligned to OM_{i-1} (already aligned)
3. **Overlap region identification**: Compute spatial intersection bounding box between consecutive OMs
4. **Centroid-based shift estimation**:
   - Select all crowns within overlap region for both OMs
   - Compute mean centroid positions for each OM
   - **Shift vector**: Δx = mean_ref_x - mean_curr_x, Δy = mean_ref_y - mean_curr_y
5. **Outlier rejection**: Use 90th percentile threshold on shift vector magnitudes, reject outliers
6. **Apply shift**: Translate all crowns in OM_i by (Δx, Δy) using shapely.affinity.translate

**Key Parameters**:
- Distance threshold for nearest-neighbor matching: 50m (relaxed to 100m if insufficient matches)
- Minimum matches required: 5

**Observed Alignment Shifts** (10Nov results):
- OM1 → OM2: ~10-30m typical shift
- OM2 → OM3: ~15-40m
- OM3 → OM4: ~20-50m  
- OM4 → OM5: ~25-60m
(Exact values depend on run, but cumulative drift increases with OM index)

**Critical Implementation Detail**: 
- Images extracted from **ORIGINAL** geometries BEFORE alignment
- Geometries shifted for matching, but images remain associated with original crown positions
- Ensures image-geometry correspondence is preserved

---

## 5. Multi-Feature Similarity Scoring

### 5.1 Pairwise Crown Metrics

For each crown pair (prev_crown, curr_crown), compute:

1. **IoU (Intersection over Union)**:
   ```
   IoU = intersection_area / union_area
   ```

2. **Overlap ratios**:
   ```
   overlap_prev = intersection_area / prev_crown_area
   overlap_curr = intersection_area / curr_crown_area
   ```

3. **Centroid distance**: Euclidean distance between crown centroids (meters)

4. **Spatial similarity**:
   ```
   spatial_sim = max(0, 1 - centroid_dist / max_dist_threshold)
   ```

5. **Area similarity**:
   ```
   area_sim = min(area_prev, area_curr) / max(area_prev, area_curr)
   ```

6. **Shape similarity**: Average of compactness and eccentricity similarities
   ```
   compactness_sim = 1 - |compactness_prev - compactness_curr|
   eccentricity_sim = 1 - |eccentricity_prev - eccentricity_curr|
   shape_sim = (compactness_sim + eccentricity_sim) / 2
   ```

7. **Containment flags**: Boolean flags for prev_contains_curr or curr_contains_prev

### 5.2 Weighted Base Similarity

Combine components with configurable weights:
```
base_similarity = w_spatial * spatial_sim + 
                  w_area * area_sim + 
                  w_shape * shape_sim + 
                  w_iou * IoU
```

**Example weights** (one_to_one case):
- w_spatial: 0.45
- w_area: 0.20
- w_shape: 0.15
- w_iou: 0.20

### 5.3 Final Match Score

Combine base similarity with overlap/centroid factors:
```
score = w_base * base_similarity +
        w_iou * IoU +
        w_overlap_prev * overlap_prev +
        w_overlap_curr * overlap_curr +
        w_centroid * centroid_factor
```

Where:
```
centroid_factor = 1 - min(1, centroid_dist / (mean_radius * 3))
```

---

## 6. Conditional Threshold Matching Strategy

### 6.1 Three Matching Cases

**Case 1: one_to_one**
- **Conditions**:
  - Exactly 1 overlapping candidate in both directions (prev_count=1, curr_count=1)
  - High overlap: overlap_prev ≥ 0.72, overlap_curr ≥ 0.72 (strict) OR ≥ 0.30 (relaxed)
  - Strong IoU: IoU ≥ 0.40 (strict) OR ≥ 0.10 (relaxed)
  - Mutual best match (highest score for both prev and curr)
- **Interpretation**: Clear 1-1 correspondence, high confidence
- **Weights**: Emphasize base similarity and IoU
- **Selection**: Mutual best matching enabled, max 1 edge per node

**Case 2: containment**
- **Conditions**:
  - Geometric containment: prev_contains_curr OR curr_contains_prev
  - Very high overlap: overlap_prev ≥ 0.82, overlap_curr ≥ 0.82
- **Interpretation**: Crown boundary changed significantly but spatial coincidence strong (segmentation artifact or real growth/shrinkage)
- **Weights**: Higher weight on overlaps, lower on spatial distance
- **Selection**: Greedy by score, max 1 edge per node

**Case 3: nearby** (relaxed matching only)
- **Conditions**:
  - Close proximity: centroid_dist < 30m
  - Minimal overlap: overlap_prev ≥ overlap_gate OR overlap_curr ≥ overlap_gate
- **Interpretation**: Crowns are close spatially but poor shape match (detection inconsistency, large shape changes)
- **Weights**: Heavily favor spatial similarity (0.70-0.85)
- **Selection**: Greedy by score, max 1 edge per node

**Case 4: none**
- All other pairs: No edge created

### 6.2 Parameter Presets

**Strict Preset** (15Oct, best precision):
```python
base_max_dist = 75.0  # Max centroid distance (m)
overlap_gate = 0.48   # Min overlap to count as overlapping
min_base_similarity = 0.35  # Min base_similarity threshold

one_to_one:
  similarity_threshold = 0.82
  min_iou = 0.40
  min_overlap_prev = 0.72
  min_overlap_curr = 0.72
  max_centroid_dist = 45.0
  mutual_best = True

containment:
  similarity_threshold = 0.74
  min_overlap_prev = 0.82
  min_overlap_curr = 0.82
  max_centroid_dist = 60.0

# Only one_to_one and containment cases enabled
```

**Ultra-Relaxed Preset** (10Nov, best recall):
```python
base_max_dist = 200.0  # Very permissive distance
overlap_gate = 0.05    # Minimal overlap requirement
min_base_similarity = 0.05

one_to_one:
  similarity_threshold = 0.25
  min_iou = 0.01
  min_overlap_prev = 0.10
  min_overlap_curr = 0.10
  max_centroid_dist = 50.0
  max_edges_per_prev = 3
  max_edges_per_curr = 3

containment:
  similarity_threshold = 0.25
  min_overlap_prev = 0.25
  min_overlap_curr = 0.25
  max_centroid_dist = 150.0

nearby:
  similarity_threshold = 0.20
  min_overlap_prev = 0.05
  min_overlap_curr = 0.05
  max_centroid_dist = 200.0
  # Spatial weight 0.85, almost purely distance-based

# All three cases enabled
```

### 6.3 Selection Algorithm

**Pipeline**:
1. **Generate candidates**: All crown pairs with centroid_dist < base_max_dist
2. **Early filtering**: Reject if base_similarity < min_base_similarity AND IoU < overlap_gate
3. **Compute overlap counts**: For each crown, count how many overlapping partners it has (overlap ≥ overlap_gate)
4. **Classify cases**: Assign each candidate to one_to_one, containment, nearby, or none
5. **Filter by case**: Apply case-specific thresholds (min_iou, min_overlap, max_centroid_dist, similarity_threshold)
6. **Prioritize cases**: Process in order [one_to_one, containment, nearby]
7. **Select per case**:
   - If mutual_best: Select only pairs that are mutual best matches
   - Else: Greedy selection by score (highest first)
   - Enforce max_edges_per_prev and max_edges_per_curr constraints
8. **Track used nodes**: Once a node has an edge, prevent further edges if allow_multiple=False

---

## 7. Tracking Results and Quality Metrics

### 7.1 Strict Preset Results (15Oct)
```
Total Trees Detected: 626
Total Tracking Edges: 2
Overall Match Rate: 0.004 (0.4%)

Match Rates by OM Pair:
- OM1→OM2: 1/80 (1.3%)
- OM2→OM3: 0/116 (0.0%)
- OM3→OM4: 0/130 (0.0%)
- OM4→OM5: 1/150 (0.7%)

Chain Length Distribution:
- Length 1: 622 trees (99.4%)
- Length 2: 2 trees (0.3%)

Edge Selection:
- one_to_one: 2 selected (100%)
- containment: 0 selected (0%)
```

**Interpretation**: Extremely conservative, very few matches. Strict thresholds ensure high precision but sacrifice recall. Most crowns tracked as singletons.

### 7.2 Ultra-Relaxed Preset Results (10Nov)
```
Total Trees Detected: 408
Total Tracking Edges: 201
Overall Match Rate: 0.632 (63.2%)

Match Rates by OM Pair:
- OM1→OM2: 57/82 (69.5%)
- OM2→OM3: 53/78 (67.9%)
- OM3→OM4: 50/78 (64.1%)
- OM4→OM5: 41/80 (51.3%)

Chain Length Distribution:
- Length 1: 114 trees (27.9%)
- Length 2: 39 trees (9.6%)
- Length 3: 21 trees (5.1%)
- Length 4: 12 trees (2.9%)
- Length 5: 21 trees (5.1%) ← Full-length chains!

Average Chain Length: 1.97
Median Chain Length: 1.0
Maximum Chain Length: 5

Edge Selection by Case:
- one_to_one: 122 / 122 candidates (100%)
- nearby: 74 / 162 candidates (45.7%)
- containment: 5 / 8 candidates (62.5%)
```

**Interpretation**: Much better recall, sacrifices some precision. Successfully tracks 21 crowns across all 5 OMs (full temporal chains). Match rate gradually degrades with increasing OM gap (OM4→OM5 worst at 51.3%).

### 7.3 Aligned vs Unaligned Comparison
- **Alignment benefit**: Improves match rates by ~5-15% depending on OM pair
- **Greatest improvement**: Mid-sequence pairs (OM2→OM3, OM3→OM4) where drift is moderate
- **Limited benefit**: First pair (OM1→OM2) has good overlap even unaligned
- **Cumulative drift**: Alignment can't fully compensate for large accumulated drift by OM4→OM5

---

## 8. Chain Typology

### 8.1 Chain Definitions
- **Chain**: A path in the directed graph from a node with in_degree=0 to a node with out_degree=0
- **Chain length**: Number of nodes in the chain (ranges from 1 to num_OMs)
- **Chain width**: Maximum number of edges entering/leaving any node in chain (width=1 means strict linear chain)

### 8.2 Chain Categories (10Nov Analysis)

**Full Chains (Length = 5)**:
- **Width-1 full chains**: 21 trees tracked linearly across all 5 OMs with no branching
  - **Phenology interpretation**: Likely evergreen trees with stable canopy, good detection consistency
  - **Confidence**: High - continuous presence indicates real tree
  
- **Branching full chains**: Not observed in 10Nov run (mutual_best prevents branching)
  - Would indicate: split/merge events or detection ambiguities
  
**Partial Chains (Length 3-4)**: 33 trees (21 + 12)
- **Possible reasons**:
  1. Deciduous trees: Leaf-off periods cause detection failures (appear/disappear)
  2. Detection sensitivity: Some crowns missed in certain OMs due to lighting, shadow, occlusion
  3. Segmentation inconsistency: Crown boundaries shift enough to break match thresholds
  4. New/removed trees: Real ecological changes (new plantings, removed trees)

**Short Chains (Length 2)**: 39 trees
- Detected in only 2 consecutive OMs
- **Interpretation**: Either ephemeral detections (false positives) or trees on edge of study area

**Singletons (Length 1)**: 114 trees (27.9%)
- No temporal matches found
- **Possible causes**:
  1. Unique to one OM (new planting, transient detection)
  2. Poor match quality (shape changes, misalignment residuals)
  3. Crown merged/split across OMs
  4. False positive detections

### 8.3 Branching Patterns (Not Observed with Strict/Ultra-Relaxed Presets)
- **1→2 splits**: One crown in OM_t → two crowns in OM_{t+1}
  - Could indicate over-segmentation OR real canopy splitting
- **2→1 merges**: Two crowns in OM_t → one crown in OM_{t+1}
  - Could indicate under-segmentation OR real canopy overlap increase
- **Pruning rules** (planned but not implemented):
  - Use ground truth or area thresholds to decide which branch is "correct"
  - Example: If one branch is much larger, it's likely the true crown

---

## 9. Advanced Matching Strategies (Explored but Not Used in Final Pipeline)

### 9.1 Rotational Shuffles (Voting-Based Edge Selection)
**Idea**: Build graphs for multiple random temporal orderings of OMs, aggregate edges that appear in ≥K orderings

**Algorithm**:
1. For N shuffles (e.g., N=8):
   - Random permutation of OM order
   - Build graph for that order
   - Track which edges appear
2. **Vote aggregation**: edge_count[(u,v)] = sum of appearances across shuffles
3. **Selection**: Keep edges with count ≥ threshold (e.g., ≥50% of shuffles)

**Benefit**: Reduces false positives from order-dependent artifacts

**Drawback**: Computationally expensive (N × original cost), can still miss true matches

**Status**: Implemented in code (crown_tracking_15Oct), tested with `rotational_agg_*` output files, NOT used for final results

### 9.2 Virtual All-Pairs Matching
**Idea**: Instead of only consecutive OM pairs (OM_i→OM_{i+1}), allow edges between ANY pair OM_i→OM_j where i < j

**Gap penalty**:
```
gap = j - i
w_gap = exp(-α * (gap - 1))
final_score = similarity_score * w_gap
```
Discourages long jumps (OM1→OM5) in favor of shorter gaps

**Benefit**: Can recover crowns that failed detection in intermediate OMs (OM1→OM2 miss, but OM1→OM3 succeeds)

**Drawback**: 
- Much larger candidate set: O(N²) pairs instead of O(N)
- Higher ambiguity risk
- Global 1-1 constraint harder to enforce

**Status**: Implemented (`build_graph_virtual_allpairs`), tested with `virtual_allpairs_enhanced_*` outputs, NOT used for final results

---

## 10. Implementation and Engineering

### 10.1 Software Stack
- **Python 3.10+**
- **Core libraries**:
  - `geopandas` (0.14+): Geometric operations on crown polygons
  - `shapely` (2.0+): Geometry manipulation (contains, intersects, union, translate)
  - `rasterio` (1.3+): Raster I/O and masking for image extraction
  - `networkx` (3.1+): Graph data structure and algorithms
  - `numpy`, `pandas`: Numerical/tabular operations
  - `matplotlib`, `seaborn`: Visualization
- **Detectree2 dependencies**:
  - `detectron2` (0.6): Mask R-CNN backbone
  - `torch` (2.0+): PyTorch for deep learning inference

### 10.2 Compute Environment
- **Workstation**: Ubuntu 22.04 (challenges with SSH access noted in notebooks)
- **Local development**: macOS (Jupyter notebooks via VS Code)
- **Memory**: Large orthomosaics (multi-GB TIFFs) require ≥16GB RAM
- **Runtime**: 
  - Detection (Detectree2): ~30-60 min per OM (depending on size)
  - Alignment: <1 min
  - Graph building (ultra-relaxed): ~2-5 min for 5 OMs
  - Graph building (strict): <1 min (fewer candidates)

### 10.3 Data Management
- **Input directory structure**:
  ```
  input/
    input_om/           # Orthomosaic TIFFs
      sit_om1.tif
      sit_om2.tif
      ...
    detectree_models/   # Pre-trained weights
      250312_flexi.pth
  ```
- **Output directory structure**:
  ```
  output/
    input_crowns/       # Crown GeoPackages from detection
      OM1.gpkg
      OM2.gpkg
      ...
    [preset_name]_quality_report.txt
    [preset_name]_quality_metrics.json
    [preset_name]_complexity_report.txt
    [preset_name]_complexity_metrics.json
    *.png               # Visualizations
  ```

### 10.4 Archiving and Versioning
- **Output file naming convention**: `[preset]_[date/descriptor]_[metric_type].[ext]`
  - Example: `crown_tracking_10Nov_quality_metrics.json`
  - Example: `strict_15oct_complexity_report.txt`
- **Version control**: Git repository tracking code and notebooks (NOT tracking large TIFFs/GPKGs)
- **Metadata**: JSON metric files contain full parameter configurations for reproducibility

---

## 11. Challenges and Limitations

### 11.1 Geometric Challenges
1. **Orthomosaic misalignment**: GPS drift causes 10-60m shifts between OMs
   - Partial mitigation via consecutive alignment
   - Residual errors still impact matching
   
2. **Lens distortion**: Radial distortion at image edges not fully corrected
   - Affects crown shapes in periphery of orthomosaics
   
3. **Ground control points**: Insufficient GCPs lead to global positioning errors
   - Professional photogrammetry would require more GCPs

### 11.2 Detection Failures
1. **Lighting variation**: Shadows, sun angle differences across flight dates
   - Can cause same tree to have different RGB appearance
   
2. **Seasonal changes**: Deciduous trees with partial/full leaf-off
   - Detectree2 trained on leafy crowns, may miss bare branches
   
3. **Crown occlusion**: Overlapping canopies from multiple trees
   - Leads to under-segmentation (merged crowns)
   
4. **Small crowns**: Young trees or shrubs near detection size limit
   - Often missed or filtered out in cleaning step

### 11.3 Matching Trade-offs
1. **Precision vs Recall**: 
   - Strict thresholds → few false matches but many missed matches
   - Relaxed thresholds → more true matches but also more false matches
   
2. **Crowded regions**: High tree density areas have many overlapping candidates
   - Ambiguous matches harder to resolve
   - Need for spatial context beyond pairwise matching
   
3. **Temporal gap sensitivity**: 
   - OM1→OM2 typically easier than OM4→OM5
   - Cumulative drift, seasonal changes, detection inconsistencies compound

4. **Parameter sensitivity**: 
   - Small changes in overlap_gate or similarity_threshold drastically affect results
   - No single "optimal" parameter set for all use cases

### 11.4 Computational Constraints
1. **Candidate explosion**: All-pairs matching would be O(N²) in number of crowns
   - Spatial indexing helps but still expensive for large areas
   
2. **Memory for large graphs**: 
   - NetworkX graph with 400+ nodes and rich edge attributes can consume GBs
   - Visualization of full graph becomes impractical

### 11.5 Ground Truth Limitations
- **No field validation**: Crowns not manually verified against ground surveys
- **No species labels**: Cannot validate phenology patterns with known tree species
- **No temporal validation**: Don't know which chains represent same physical tree vs detection artifacts

---

## 12. Key Lessons Learned

### 12.1 Importance of Alignment
- Even modest positional shifts (10-20m) can break matching for crowns with poor IoU
- Consecutive alignment is computationally cheap and significantly improves recall
- More sophisticated alignment (rotation, scale) may help but not yet explored

### 12.2 Case-Based Matching
- Single global similarity threshold is insufficient
- Different matching scenarios require different criteria:
  - High-quality one_to_one matches: Strict thresholds
  - Containment cases: Emphasize overlap over centroid distance
  - Fallback nearby matches: Spatial proximity dominates
- Explicit case classification makes algorithm interpretable and tunable

### 12.3 Hyperparameter Tuning
- Optimal parameters depend on:
  - Alignment quality
  - Detection consistency across OMs
  - Crown density/overlap in scene
  - Desired precision/recall trade-off
- Parameter sweeps are essential (e.g., threshold_07, threshold_07_gated, threshold_07_topk3 experiments)
- No "one size fits all" solution

### 12.4 Graph Topology as Diagnostic
- **Degree distributions** reveal matching quality:
  - Most nodes with out_degree ≤ 1: Good 1-1 tracking
  - High in_degree nodes: Potential merge artifacts
  - High out_degree nodes: Potential split artifacts
- **Connected component sizes** indicate tracking continuity:
  - Many small components: Poor temporal matching
  - Large components: Successful long-term tracking
- **Chain length distribution** most intuitive metric for end-users

### 12.5 Visualization is Critical
- Overlaying aligned crowns across OMs reveals residual misalignment
- Plotting chains with extracted RGB images confirms phenological patterns
- Interactive HTML maps (Folium) useful for spatial exploration but not created in final pipeline

### 12.6 Iterative Development
- Progression from **demo** (simulated data) → **strict** (ultra-conservative) → **relaxed** (balanced) → **ultra-relaxed** (maximize recall)
- Each version informed by failures of previous:
  - Strict preset showed baseline was too conservative
  - Alignment added to address misalignment
  - Nearby case added for poor-shape-match crowns
  - Ultra-relaxed tested limits of recall maximization

---

## 13. Next Steps and Future Enhancements (Not Yet Implemented)

### 13.1 Algorithmic Improvements
1. **Multi-date context**: Use information from multiple OMs (not just pairwise) to disambiguate matches
   - Example: OM1→OM2 uncertain, but OM1→OM2→OM3 chain has high consistency → favor that path
   
2. **Spectral features**: Beyond RGB, use vegetation indices (NDVI, EVI) if multispectral data available
   - More robust to illumination changes
   
3. **Shape descriptors**: Add Fourier descriptors, Hu moments for rotation-invariant shape matching
   
4. **Learned embeddings**: Train neural network to embed crown features, use cosine similarity
   - Data-driven alternative to hand-crafted features

### 13.2 Virtual Nodes for Broken Chains
- Insert "virtual nodes" when crown disappears for 1-2 OMs then reappears
- Example: OM1 → [virtual_OM2] → OM3 → OM4 → OM5
- Allows tracking deciduous trees through leaf-off periods
- Requires confidence thresholds and spatial consistency checks

### 13.3 Ground Truth and Validation
1. **Field surveys**: GPS-tag select trees, verify crown boundaries and species
2. **Manual annotation**: Create ground-truth chains for sample trees
3. **Metric validation**: Compute precision/recall against ground truth

### 13.4 Scalability
1. **Spatial indexing**: R-tree for crown pairs to avoid all-pairs comparison
2. **Distributed processing**: Parallelize detection and matching across tiles/OMs
3. **Database backend**: Store crown graphs in PostgreSQL/PostGIS for query efficiency

### 13.5 Phenology Analysis
1. **Feature extraction from chains**:
   - Crown area time series → growth/shrinkage patterns
   - RGB color histograms → greenness/senescence proxy
   - Chain length distribution by location → evergreen vs deciduous zones
   
2. **Statistical modeling**:
   - Fit phenological curves (e.g., logistic growth models)
   - Cluster trees by temporal signature
   - Correlate with weather data (temperature, rainfall)

---

## 14. Key Metrics Summary Table

| Metric | Strict (15Oct) | Ultra-Relaxed (10Nov) |
|--------|----------------|----------------------|
| Total crowns | 626 | 408 |
| Total edges | 2 | 201 |
| Overall match rate | 0.4% | 63.2% |
| Full-length chains | 0 | 21 |
| Avg chain length | 1.00 | 1.97 |
| Max chain length | 2 | 5 |
| one_to_one edges | 2 | 122 |
| containment edges | 0 | 5 |
| nearby edges | N/A | 74 |

**Conclusion**: Ultra-relaxed preset successfully tracks ~5% of trees across all 5 orthomosaics, with ~64% of crowns matched between at least 2 consecutive OMs. This is a significant improvement over baseline strict matching.

---

## 15. Code Structure Overview

### Key Classes and Functions

**TreeTrackingGraph** (`crown_tracking_10Nov.ipynb`, `crown_tracking_15Oct.ipynb`, `crown_tracking_4Nov.ipynb`):
- `__init__()`: Initialize tracker, discover files
- `discover_files()`: Auto-pair crown GeoPackages with orthomosaic TIFFs
- `load_data(load_images, align, reference_om_id)`: Load crowns, optionally extract images and align
- `align_to_reference(reference_om_id)`: Compute and apply alignment shifts
- `_compute_crown_attributes(geometry)`: Extract geometric features from polygon
- `_compute_pair_metrics(prev_attrs, curr_attrs, max_dist)`: Compute all similarity metrics for crown pair
- `_classify_match_case()`: Assign case label (one_to_one, containment, nearby, none)
- `_score_candidate()`: Compute weighted final score from base_similarity and features
- `_select_candidates_by_case()`: Prioritize and select edges by case with constraints
- `build_graph_conditional()`: Main graph building pipeline
- `quality_report()`: Generate match rates, chain lengths, edge selection stats
- `complexity_report()`: Graph topology metrics (degree dists, components, diameter)
- `_extract_all_chains()`: Find all temporal chains in graph
- `_greedy_chain(start_node)`: Follow best-successor path from start node

**Utility Functions**:
- `consecutive_alignment(tracker, distance_threshold, min_matches)`: Nearest-neighbor consecutive alignment
- `visualize_chain_with_extracted_images()`: Plot chain polygons + RGB images
- `categorize_chains(tracker)`: Group chains by length and width
- `find_branching_cases(tracker, min_chain_length)`: Identify split/merge nodes

**Configuration Dataclass**:
- `MatchCaseConfig`: Stores all parameters for one matching case (thresholds, weights, selection rules)

---

## 16. Output Files Explained

### Quality Reports
- **Format**: Plain text, human-readable summary
- **Contents**:
  - Total nodes, edges
  - Overall match rate
  - Match rates by consecutive OM pair
  - Chain length distribution
  - Edge selection by case (counts, ratios)
- **Example**: `crown_tracking_10Nov_quality_report.txt`

### Quality Metrics JSON
- **Format**: JSON with nested dictionaries
- **Contents**: Same as text report but machine-readable
- **Example**: `crown_tracking_10Nov_quality_metrics.json`

### Complexity Reports
- **Format**: Plain text
- **Contents**:
  - Average in/out degrees
  - Degree distributions (histograms)
  - Zero-degree node counts
  - Weakly/strongly connected component counts and sizes
  - Graph diameters (longest shortest path in each component)
- **Example**: `strict_15oct_complexity_report.txt`

### Complexity Metrics JSON
- **Format**: JSON
- **Contents**: Same as complexity text report, machine-readable
- **Example**: `strict_15oct_complexity_metrics.json`

### Visualizations (PNG)
- `chain_length_distribution.png`: Bar chart of chain lengths
- `match_rates_by_pair.png`: Bar chart of match rates by OM pair
- `degree_distributions.png`: Histograms of in/out degrees
- `all_5_oms_overlay_aligned.png`: All OM crowns overlaid after alignment
- `[preset]_chain_examples.png`: Sample visualizations of tracked chains with images

### Other Outputs
- `edge_counts_by_threshold.json`: Results from threshold sweeps
- `all_om_chosen_thresholds.csv`: Adaptive threshold selection results from detectree_parth
- `interactive_tracking_map.html`: Folium map for spatial exploration (if generated)

---

## 17. Notebook-Specific Insights

### tree_tracking_graph_demo.ipynb
- **Purpose**: Pedagogical demonstration with simulated data
- **Key contribution**: Visual explanation of all tracking cases with minimal examples
- **Not used for real data**: All examples are synthetic polygons

### detectree_parth.ipynb
- **Purpose**: Crown detection pipeline from orthomosaics
- **Key contribution**: Adaptive per-OM confidence thresholding
- **Output**: Crown GeoPackages in `output/input_crowns/`

### crown_tracking_15Oct.ipynb
- **Purpose**: Strict high-precision tracking baseline
- **Key contribution**: Establishes minimal false-positive rate
- **Finding**: 0.4% match rate too conservative for practical use

### crown_tracking_4Nov.ipynb
- **Purpose**: Add spatial alignment and test aligned vs unaligned
- **Key contribution**: Shows alignment improves match rates by ~5-15%
- **Finding**: Alignment helps but not sufficient alone

### crown_tracking_10Nov.ipynb
- **Purpose**: Ultra-relaxed preset for maximum recall
- **Key contribution**: Achieves 63% match rate and 21 full-length chains
- **Finding**: Best balance of precision/recall found so far
- **Innovation**: Image extraction from original geometries BEFORE alignment (preserves correspondence)

### crown_tracking_graph_new.ipynb
- **Purpose**: Earlier exploration of graph-based approach
- **Status**: Preliminary, superseded by later notebooks

### crown_tracking.ipynb, crown_tracking refined.ipynb, crown_tracking_real_data_demo.ipynb, crown_tracking_good matches only.ipynb
- **Purpose**: Intermediate development versions
- **Status**: Exploratory, not used for final results

---

## 18. Terminology and Notation

### Abbreviations
- **OM**: Orthomosaic
- **IoU**: Intersection over Union
- **GCP**: Ground Control Point
- **NDVI**: Normalized Difference Vegetation Index

### Node Notation
- `(om_id, crown_index)`: Tuple representing unique crown
  - Example: `(1, 5)` = 6th crown (0-indexed) in OM1
- Node ID strings: `"OM1_5"` (for visualization)

### Edge Attributes
- `similarity`: Final weighted score
- `iou`: Intersection over Union
- `overlap_prev`, `overlap_curr`: Overlap fractions
- `centroid_distance`: Euclidean distance (meters)
- `base_similarity`: Weighted combination of spatial/area/shape/iou
- `spatial_similarity`, `area_similarity`, `shape_similarity`: Components
- `case`: Match case label (one_to_one, containment, nearby)
- `method`: Matching method used (e.g., 'conditional', 'shuffle_agg')

### Chain Terminology
- **Chain**: Directed path through graph from source (in_degree=0) to sink (out_degree=0)
- **Full chain**: Length equals number of OMs (spans all time points)
- **Partial chain**: Length < number of OMs (some time points missing)
- **Width-1 chain**: No node has in_degree > 1 or out_degree > 1 (linear)
- **Branching chain**: At least one node with in_degree > 1 or out_degree > 1

---

## Conclusion

This project successfully developed a **graph-based crown tracking framework** that:
1. Detects tree crowns from drone orthomosaics using Detectree2
2. Applies adaptive confidence thresholding for balanced detection across dates
3. Aligns orthomosaics to reduce positional drift
4. Matches crowns across time using multi-feature similarity scoring
5. Handles diverse matching cases (one_to_one, containment, nearby)
6. Tracks 21 trees across all 5 orthomosaics (full temporal chains)
7. Achieves 63.2% overall match rate with ultra-relaxed parameters

**Key innovation**: Case-based conditional matching with explicit thresholds per case type, enabling interpretable and tunable tracking.

**Limitation**: Lack of ground truth prevents quantitative precision/recall validation; trade-offs between false positives and false negatives are qualitatively assessed from visual inspection and metric distributions.

**Future work**: Virtual nodes for broken chains, learned embeddings, field validation, and phenology feature extraction from tracked chains.

This comprehensive analysis provides all necessary context for writing the technical report covering methods, algorithms, parameters, results, challenges, and lessons learned.
