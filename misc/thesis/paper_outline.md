# Paper Outline: Drone-Based Individual Tree Phenology Monitoring

Working framing:

> Repeated UAV RGB orthomosaics can support individual-tree phenology monitoring if per-date crown detections are converted into temporally stable tree identities through explicit alignment, graph-based temporal association, gap filling, and consensus crown construction.

This outline treats the **tree-identity-preserving temporal pipeline** as the central contribution. Satellite/species experiments should be held for a separate short paper once that workflow and its results are complete. The appendix for this paper should instead document the datasets, dates, labels, and reproducibility metadata behind the UAV tracking/phenology analysis.

---

## Working Context for Future LLM Sessions (READ THIS FIRST)

This block is the single up-to-date brief for anyone (human or LLM) continuing
the paper. It captures the project, the decisions, the verified code facts, the
data, the supervisor's instructions, the author's style preferences, and what is
still blocked. If something here conflicts with older prose lower in this file,
**this block and the heading skeleton win.**

### 0.1 What the paper is

A master's thesis framed as a paper. One sentence: *an end-to-end pipeline that
turns repeated UAV RGB orthomosaics of the same site into temporally stable
individual-tree crowns and crown-level phenology, using Detectree2
multi-threshold detection, tiled phase-cross-correlation alignment, graph-based
crown association, medoid consensus crowns, and RGB/texture phenology features,
with field species labels joined in.* Title is locked (see paper.md / main.tex).

### 0.2 Scope decision (LOCKED 2026-06-15)

- **Drone-only.** Satellite / species classifiers (Sentinel-2 RF, GEE embeddings,
  DINOv2, NICFI) are **out of this paper** and become a separate short companion
  paper. They may appear only as *motivation* (drone labels as training data for
  satellite phenology models) and brief future work — never as results here.
- This was the author's call, overriding a meeting-digest suggestion to include
  satellite in the main body. Do not silently reopen it. If the supervisor
  pushes for inclusion, fall back to "satellite as a secondary appendix," using
  the GEE-embedding Acacia result + the crown-size-vs-Sentinel-2-pixel
  scale-mismatch analysis.
- Also explicitly **excluded**: CHM / DSM / DTM / tree-height estimation (that
  was companion-project context, not this thesis's work). The supervisor's
  "discover the height we can fly at" was about flight planning, not tree height.

### 0.3 Source-of-truth files and where they live

- `misc/thesis/paper.md` — heading skeleton (Markdown), main repo.
- `misc/thesis/paper_outline.md` — THIS file: detailed spec + context, main repo.
- `misc/thesis/drone-phenology-thesis-overleaf/` — **separate git repo**, pushed
  to GitHub `hbot07/drone-phenology-thesis-overleaf` (branch `master`), linked to
  Overleaf. `main.tex` is what the supervisor reviews; `references.bib` holds the
  Section-2 source pool. This folder is gitignored by the main repo, so its files
  live ONLY in the Overleaf repo.
- Editing/compiling the LaTeX: edit `main.tex`, then
  `pdflatex main.tex` (twice; add `bibtex main` once citations exist),
  commit with `git -c user.name="Parth" -c user.email="parth5462003@gmail.com"
  commit`, and `git push origin master`. `\usepackage{lmodern}` is required (the
  local basic TeX Live otherwise fails microtype font expansion).
- paper.md and paper_outline.md are edited in the main repo but the author
  commits the main repo themselves; only the Overleaf repo is auto-pushed.

### 0.4 Code architecture (what backs the methodology)

The production pipeline is `src/notebooks/pipeline/00..04_*.py` (discover →
Detectree2 detection → tracking+consensus → phenology → viewer), driven by one
shared `pipeline_config.json`. **Steps 2–3 are thin orchestrators**: the real
algorithms live in `src/flask_app_tracking/tree_tracking.py`
(`TreeTrackingGraph`: alignment, graph, chains, gap-fill, consensus, dedup) and
`src/flask_app_tracking/phenology_leafshed.py` (patch features + deciduousness +
phenophase). `src/notebooks/organised/tracking/` is a cleaner parallel re-write
of the same math (notebook-first) — good as a formalization reference, NOT the
production code. The legacy Flask UI (`app.py`) is superseded by the standalone
Leaflet viewer in Step 4.

### 0.5 Code-verified parameters (state the actual run values in the paper)

- Detection: Detectree2 model `250312_flexi.pth`; tiles **25×25 m, buffer 15 m**
  in the production pipeline (notebook era used 45×45 m / 20 m — pick the values
  of the run that made the final tables); `clean_crowns` simplify tol 0.3, fixed
  IoU 0.7; thresholds `conf_0p15`…`conf_0p65` (0.05 steps).
- Tracking base layer: **`conf_0p45` for LHC, `conf_0p15` for SIT** (SIT's 0.45 is
  too sparse). Alignment anchor layer `conf_0p65`.
- Alignment: default `pcc_tiled`; reference = OM1; each OM aligned to the previous
  one and shifts **accumulated** (cumulative-to-OM1); 4×4 tile PCC with
  texture/error/max-shift gating + MAD-inlier median; shifts saved to config and
  reused by Steps 3–4 (so crop sampling matches tracking via inverse transform).
- Graph: `base_max_dist=30`, `overlap_gate=0.10`, `min_base_similarity=0.30`,
  `classify_mode=balanced`; cases one-to-one / containment / nearby; split/merge
  allowed via `allow_multiple` + `max_edges_per_prev/curr`.
- Chains for consensus: full width-1 chains + extracted backbones of branching
  full chains + partial chains with `len>=5` and one-to-one ratio `>=0.9`.
- Consensus medoid weights: 0.5 centroid (normalized) / 0.4 (1−IoU) / 0.1
  (1−area_ratio). Dedup: IoU `>0.75` + containment (buffer 5.0), larger-area-first.
- Phenology features: GCC=G/(R+G+B), RCC=R/(R+G+B); HSV veg mask h∈[0.18,0.48],
  s≥0.15, v≥0.12; shadow = v<0.12; gray Shannon entropy; 4-neighbour Laplacian
  variance. QC `is_bad` if valid-pixel-frac<0.60 OR shadow>0.55 OR Laplacian<25.
- Deciduousness: amplitudes of interpolated veg/GCC/texture, normalized by A90
  (90th-percentile amplitude across crowns); DS = 0.35·veg_amp + 0.30·depth +
  0.25·gcc_amp + 0.10·tex_amp; depth=(τ−min_veg)/τ with τ=veg_min (default 0.45);
  **deciduous if DS≥0.70 (pipeline Step 3 value; dataclass default is 0.85 —
  report the run value)**. Phenophase on min-max-normalized veg: leaf_on≥0.65,
  leaf_off≤0.35, else transitioning; non-deciduous=stable. Events:
  leaf_off_start_om, full_leaf_off_om (veg trough), leaf_on_return_om.

### 0.6 Feasibility caveats (don't over-claim)

- The explicit lower-threshold gap-filler (`augment_partial_chains_with_multithreshold`)
  EXISTS but is **not called** in the production Step 2. What actually helps is
  the **multi-threshold candidate union** (OM1 = base layer; later OMs = union of
  all threshold layers, deduped). §4.5.4 is named "Gap Filling via Multi-Threshold
  Candidates" to match. Don't describe a predicted-position gap-filler as run.
- The non-tree filter (`apply_non_tree_thresholds`) exists but Step 3 disables it.
  No automated non-tree removal in the current outputs.

### 0.7 Sites and data

- Primary sites: **LHC** and **SIT** on the IIT Delhi campus. Full available data:
  **LHC = 13 OMs, SIT = 19 OMs** (per the 2026-05-19 meeting). Sanjay Van: ~8 OMs
  spots 1–3, ~6 OMs spot 4 (companion-paper data, not this paper).
- Cleanest partial pipeline runs inspected so far (placeholders until the full run):
  `output/lhc_pipeline_fixed` (LHC, 8 OMs, base conf_0p45) →
  ~87 consensus crowns, ~69% deciduous;
  `output/sit_pipeline_fixed` (SIT, 14 OMs, base conf_0p15) →
  ~131 consensus crowns, ~79% deciduous.
  **Final tables must be regenerated from the full 13-OM LHC / 19-OM SIT runs.**
- Crown-detection ground truth: `input/ground_truth.json`, 124 manually annotated
  crowns (COCO). Early SIT detection metrics: P=0.438, R=0.258, F1=0.325, mean
  matched IoU 0.72 — placeholder, regenerate.
- Species: ~372 clean species-labeled crowns; top species include Prosopis
  juliflora, Neem, Ashok, Amaltas, Peepal, Pilkhan. Flowering colors present in
  data: **yellow and red only (no white)**.

### 0.8 Drone / acquisition facts (cite-able)

DJI **Mini 4 Pro** (consumer drone; no public SDK for autonomous fixed-overlap
paths — a real limitation; the Mini 3 had one). Intrinsics: focal 5.067 mm,
sensor 7.6×4.275 mm. Coordinates stored in **XMP, not EXIF**; XMP→EXIF conversion
before WebODM; CRS WGS84 / UTM 43N. Orthomosaics built in **WebODM**. Imprecise
EXIF georeferencing + wind/terrain yaw-pitch-roll → distortion; **~5–7 m residual
offset** even after EXIF-corner placement (this motivates §4.3). External GPS/RTK
investigated and rejected (cost/accuracy). Drone-mapping collaborators: Craig
Dsouza, SS Jayakrishna (acknowledge). Flights ~every 2 weeks, ~50–80 m AGL,
~80% overlap, near-nadir. (Confirm final flight count/dates with the author.)

### 0.9 Field-labeling workflow (for §3.4)

Labeling started in **QField** (guide in `misc/docs`) but was too slow — too many
clicks, one crown at a time. Switched to **paper forms**: crowns were given IDs in
nearby/walking order, and the team walked the site writing running notes (e.g.,
"12, 15, 21 — Amaltas"), carrying (a) a printout of the orthomosaic with crown IDs
and (b) the QField app showing the same IDs to point out the current tree. Labels
later joined to crown IDs in the master geojson. **Photos of the paper forms +
field photos are pending from the author — remind them.**

### 0.10 Author's writing-style instructions

- Natural, human voice; vary sentence structure and word choice; coherent, readable.
- Eliminate AI tells: avoid formulaic transitions ("Furthermore," "In conclusion,"
  "Moreover"), avoid excessive passive voice, avoid templated semicolon list-pileups.
- BUT keep an academic register — the author rejected an overly casual abstract
  rewrite. Aim for clean, precise, standard academic prose, not chatty. Active
  voice where natural ("We detect…", "The tracker links…").

### 0.11 Status / what's blocked on the full data run

Filled now (code-/fact-backed, results-independent): Methodology (§4), §3.2–3.4,
and Introduction framing as it gets written. **Deferred until the full LHC/SIT run
produces results:** §5 Evaluation and Results, §6 Seasonal Phenology Mapping, the
§3.1 site specifics and Appendix D inventories, and any numeric claims. Also
pending: the two-OM unreliability figure (§1.2), the Semal across-time figure
(§1.3; extractable from the species geojson → crown → aligned crops across LHC/SIT
OMs), the §4.1 flowcharts (boxes proposed; awaiting author OK then TikZ), the §2
relation-to-our-work lines, and a manual tracking-validation subset (§5.3.2) +
consensus-vs-GT IoU evaluation (§5.4, GT already in `input/ground_truth.json`).

### 0.12 Section 2 sources

`references.bib` (Overleaf repo) holds a curated, theme-grouped source pool from
TWO complementary deep-research passes. It IS now wired in (`\bibliography`), and
§4 already cites the core methods, so a References list renders. Verify
DOIs/pages/authors before camera-ready (entries with `note={verify}` are unconfirmed).

- The two full research write-ups (with a one-line "relation to our work" for every
  source, plus summaries) are saved at `misc/thesis/others/Related Work *.pdf`.
  When writing §2, lift the relation lines from there.
- Recommended §2.4 narrative (from the research): SORT → DeepSORT → Kuhn/Munkres
  (establish the Hungarian one-to-one paradigm and its split/merge failure) →
  Zhang/Li/Nevatia network-flow global association → our graph-based crown
  association with medoid consensus crowns. This is where the novelty lands hardest.
- Recommended §2.1/§2.6 framing: state plainly that Klosterman & Richardson (2017),
  Berra (2019), and Wu (2021) draw or assume per-date ROIs without an automated
  cross-date crown-linking step — that one sentence positions the consensus-crown
  contribution. Wu (2021) is also the best precedent for the satellite companion goal.
- `[verify authors]` placeholder entries from the research (a GIScience&RS urban
  multi-sensor 2021, a Drones UAV-RS review 2023, an ITCD review preprint) were
  intentionally left OUT of references.bib; add them only after filling real authors.

Field-labeling photos: two are now in the Overleaf repo and placed as figures in
§3.4 — `field_qfield_usage.jpeg` (§3.4.1) and `field_printout_crown_ids.jpeg`
(§3.4.3). The **paper-form** photo is still pending from the author.

---

## Professor Feedback (Round 1) — Structure Directives

Captured 2026-06-15. The heading skeleton in `paper.md` and the Overleaf
`main.tex` is now the source of truth for section structure; where the
"Proposed Paper Structure" below (this document's Section 5) disagrees, these
directives win.

**Introduction**

- **1.1 Background and Motivation** must state **two problem statements**:
  1. Crown detection on a single orthomosaic is not reliable — it depends on
     canopy structure at that moment (some trees leafless; similar trees merge
     into one large canopy blob).
  2. Labeling tree phenology (leaf-shed and flowering times) provides robust
     training data for satellite models that can move back/forth in time,
     analyze multi-year phenology change, and compare against weather events.
     (Satellite stays as *motivation only*; experiments remain deferred.)
  Implemented as subsubsections 1.1.1 / 1.1.2.
- **1.2** recentered on **examples** of single-OM unreliability: show two OMs of
  the same area with crowns missed and crowns merged. (Retitled "Examples of
  Unreliable Single-Orthomosaic Crown Detection".)
- **1.3 renamed** "From Crown Detection to Temporal Tree **Phenology**" (was
  "...Identity"). Show one crown across time — the **Semal** example
  (leaves → leafless → flowers). Doable now: species are marked in the master
  geojson, so extract Semal crops across OMs and lay them out as a figure.

**Related Work (Section 2)**

- Each 2.x subsection must end with a line or two on **how it relates to our
  work** ("we take the same approach", "this does not work for us, so we do X",
  "our work can help solve this", etc.). No heading change.
- Sources gathered (2026-06-15) into `references.bib` in the Overleaf repo,
  grouped by the six themes. Not yet rendered (no `\cite`/`\nocite`). Relation
  lines still to be written. DOIs/pages/authors need verification before
  camera-ready (some AI-suggested metadata; a few entries carry `note={verify}`).

**Study Area and Data (Section 3)**

- **3.2 / 3.3:** give method-level detail but keep it light. Installation,
  library names, exact parameters belong in the GitHub repo, not the paper.
- **3.4:** document that labeling was first attempted in **QField** (guide in
  `misc/docs`) but it was slow — too many clicks, one crown labelled at a time.
  We switched to **paper forms**: crowns were assigned IDs in nearby/walking
  order, and we walked the site writing running notes (e.g., "12, 15, 21 —
  Amaltas"), carrying (a) a printout of the orthomosaic with crown IDs and
  (b) the QField app showing the same IDs to point out the current tree.
  Implemented as subsubsections 3.4.1 (QField pilot + limits), 3.4.2 (paper-form
  workflow), 3.4.3 (crown-ID printouts). Method 4.9.2 retitled accordingly
  ("Paper-Form to Crown-ID Label Joins"). **TODO: insert photos of the paper
  forms and field photos (user to provide).**

**Methodology (Section 4)**

- **4.1** must carry **flowcharts/infographics**: one high-level workflow of the
  main components, plus detailed flowcharts for each smaller component. To be
  drawn (placeholder TODO in the LaTeX).

**Evaluation and Results (Section 5)**

- Be selective: only include tables/graphs we are confident are backed by good
  results and are meaningful. Do not pad with weak/uncertain tables.
- Section retitled **"Evaluation and Results"** — the old Section 6 metric
  results are merged up here (one place for all quantitative evaluation:
  detection, alignment, tracking, consensus, phenology features, plus
  visualization-based error analysis).

**Seasonal Phenology Mapping (Section 6, repurposed)**

- The prof asked how 6 differs from 5. Section 6 is now the **applied seasonal
  output** section, distinct from the metrics in 5:
  - 6.1 per-OM flowering-color labeling: Red / Yellow / None. (White dropped —
    confirmed 2026-06-15 that only yellow and red flowering crowns exist in the
    data.)
  - 6.2 leaf-shed phenophase progression across the season (what the OMs look
    like in March, then April, then May, etc.).
  - 6.3 species-resolved phenology patterns.
  - 6.4 illustrative crown-level trajectories.

**Discussion (Section 7) — REMOVED**

- Per feedback, the standalone Discussion is removed; its points are woven into
  the nearest relevant sections, and a brief future-work paragraph goes in the
  Conclusion. Migration map:
  - "tree identity is the core unit" → Introduction 1.3 / Conclusion.
  - "role of multi-threshold detection" → Method 4.2 / Eval 5.1.
  - "impact of alignment" → Method 4.3 / Eval 5.2.
  - "graph association under split-merge" → Method 4.4 / Eval 5.3.
  - "consensus crowns for stable sampling" → Method 4.6 / Eval 5.4.
  - "field labels and ecological interpretation" → Method 4.9 / Section 6.
  - "operational considerations" → Section 3 / Conclusion.
  - "limitations + future work" → distributed inline + brief Future Work in
    the Conclusion.

---

## 1. Candidate Titles

### Recommended Main Title

**Tracking Individual Tree Phenology from Repeated UAV RGB Orthomosaics Using Graph-Based Crown Association and Consensus Crowns**

Updated working title after moving satellite/species work out of this paper's main structure.

Why this works:

- "Tracking Individual Tree Phenology" immediately states the scientific object.
- "Repeated UAV RGB Orthomosaics" makes the data modality explicit.
- "Graph-Based Crown Association" names the central method.
- "Consensus Crowns" names the stable sampling unit that makes repeated crown-level phenology possible.
- It keeps the paper focused on the UAV tracking/phenology contribution and avoids promising satellite/species results in the title.

### Shorter Alternative

**From Crown Detection to Tree Identity: A UAV RGB Pipeline for Individual Tree Phenology**

Why this works:

- Strong conceptual transition: detection is not the final goal; identity is.
- Shorter and cleaner if the paper target prefers a tighter title.

### Thesis-Style Alternative

**Drone-Based Individual Tree Phenology Monitoring Through Temporal Crown Tracking and Consensus Geometry**

Why this works:

- Slightly broader and more thesis-friendly.
- Makes the two major parts visible:
  1. drone tree tracking;
  2. consensus crowns and phenology.

---

## 2. One-Sentence Thesis Statement

This work presents an end-to-end UAV RGB pipeline that transforms repeated orthomosaics into temporally stable individual-tree crown identities and crown-level phenology trajectories using multi-threshold crown delineation, residual orthomosaic alignment, graph-based temporal crown association, gap filling, consensus crown generation, and image-derived crown-level phenology features.

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

Sampling from noisy per-date polygons produces inconsistent regions of interest. A consensus crown per tracked tree provides a stable spatial object for comparable color, vegetation, texture, and visual crop extraction across dates.

### Claim 5: Drone-derived crown identities enable crown-level ecological interpretation

Once stable individual-tree crown objects exist, they can support:

- RGB phenology time series;
- deciduous/evergreen and leaf-on/leaf-off scoring;
- field/QField species linkage;
- crown-level species and trait summaries;
- interactive visual validation and communication.

Satellite/species classifier experiments remain useful, but they should be treated as appendix material and future companion-paper work rather than as a core claim of this paper.

---

## 4. Abstract Drafts

### Abstract Draft A: Paper-Style

Repeated UAV RGB surveys provide a practical route to monitoring tree phenology at individual-crown resolution, but converting orthomosaic time series into biological tree trajectories remains challenging. Per-date crown detections are affected by georeferencing drift, photogrammetric artifacts, changing illumination, seasonal canopy structure, and segmentation split-merge errors. We present an end-to-end pipeline that converts repeated UAV orthomosaics into temporally stable individual-tree crown identities and crown-level phenology trajectories. The workflow combines Detectree2-based multi-threshold crown delineation, residual orthomosaic alignment using phase-cross-correlation-based image registration, graph-based temporal association of crown polygons, gap filling using lower-confidence detections, and consensus crown construction for stable repeated sampling. For each tracked tree, the system extracts crown-level RGB and texture features across dates and derives interpretable phenology indicators such as green chromatic coordinate, red chromatic coordinate, vegetation fraction, shadow fraction, and leaf-on/leaf-off state. Field and QField annotations are linked to the resulting crown objects, enabling species-aware inspection and ecological interpretation. The resulting framework reframes UAV crown delineation as a temporal identity problem and provides a practical workflow for transforming repeated orthomosaics into crown-level phenology records.

### Abstract Draft B: More Conservative

Repeated UAV RGB orthomosaics can capture fine-scale seasonal changes in tree canopies, but individual-tree phenology monitoring requires stable tree identities across dates. This is difficult because crown detections vary with illumination, leaf state, orthomosaic reconstruction quality, residual spatial misalignment, and segmentation ambiguity. This thesis develops a UAV-based workflow for transforming repeated orthomosaics into individual-tree phenology records. The workflow detects crown polygons using Detectree2, stores detections across multiple confidence thresholds, aligns orthomosaics into a common frame, constructs a temporal graph of plausible crown correspondences, fills gaps using lower-confidence candidates, and generates consensus crown geometries for repeated feature extraction. Crown-level RGB and texture metrics are then used to characterize phenological change, including canopy greenness, reddishness, vegetation fraction, shadow contamination, and deciduousness. Field labels and visual review tools are linked to the tracked crown layer for interpretation and quality assurance. The work demonstrates that the key step in operational UAV phenology monitoring is the conversion of unstable per-date detections into persistent crown identities that can be sampled, validated, and analyzed through time.

### Abstract Draft C: Thesis-Plus-Paper Hybrid

Monitoring tree phenology at the level of individual crowns can improve ecological interpretation of seasonal canopy change. This thesis presents a drone-based system for individual tree phenology monitoring from repeated UAV RGB orthomosaics. The central challenge is temporal identity: crown polygons detected independently on each date do not directly correspond to stable biological trees because of orthomosaic drift, photogrammetric artifacts, changing illumination, leaf shedding, and segmentation split-merge errors. To address this, the system combines multi-threshold Detectree2 crown detection, phase-cross-correlation-based alignment, graph-based temporal crown association, missing-detection recovery, and medoid consensus crown generation. These stable crown objects are used to extract per-date crown crops and RGB/texture phenology features. The resulting trajectories support rule-based deciduousness and leaf-state scoring, manual validation, field species linkage, and interactive visualization. Together, the work shows how repeated drone surveys can be transformed from independent orthomosaics into a temporally structured, crown-level ecological monitoring dataset.

Recommended starting abstract: **Draft C** for thesis, then tighten later into **Draft A** for paper submission.

---

## 5. Proposed Paper Structure

## 1. Introduction

### 1.1 Motivation: Tree-Level Phenology Needs Individual Identity

Write:

- Phenology is a sensitive indicator of climate, water stress, deciduousness, flowering, and ecosystem change.
- Satellite phenology provides broad coverage but often mixes multiple trees/species within one pixel; this motivates UAV-scale monitoring, while satellite classifier experiments are left for future companion work.
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
- Even fewer connect the resulting crown identities to field labels, visual validation, and repeated crown-level phenology extraction.

Avoid saying "no one has done this" unless we verify literature very carefully. Say "this remains an operational bottleneck" or "less attention has been given to".

### 1.4 Contributions

Use a numbered contribution list.

Suggested contribution wording:

1. We develop an end-to-end UAV RGB workflow that converts repeated orthomosaics into temporally stable individual-tree crown identities.
2. We introduce a practical multi-threshold crown-detection and graph-association strategy for handling confidence variation, missing detections, and split-merge crown segmentation errors.
3. We incorporate residual orthomosaic alignment using phase-cross-correlation-based translation estimation to improve crown correspondence across dates.
4. We construct consensus crown geometries from temporal chains, enabling stable repeated sampling of crown-level RGB and texture phenology features.
5. We connect drone-derived crown identities with field species labels and interactive visualization outputs for ecological interpretation and quality assurance.

### 1.5 Paper Roadmap

One short paragraph:

- data and study sites;
- crown detection and tracking pipeline;
- phenology extraction;
- field label integration and visualization;
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

### 2.6 Field Labels, Species Traits, and Crown-Level Ecological Interpretation

This is where field labels and species context support the main UAV phenology paper without turning it into a satellite/species classifier paper.

Write about:

- Crown-level field labels as a way to interpret tracked phenology trajectories.
- Species and trait labels as ecological context for differences in leaf state, flowering color, and deciduousness.
- QField/manual validation workflows for linking labels to crown geometries.
- Visual review and map exports as practical QA tools.
- A brief note that satellite/species classifiers are possible follow-on analyses but are not the main paper claim.

Connect:

> In this work, field labels are used primarily to interpret and validate crown-level UAV phenology records. Satellite and embedding classifiers are deferred to an appendix and a later short paper.

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

Resolved from prof emails / meetings (cite-able):

- DJI Mini 4 Pro is a **consumer** drone, not survey-grade: no public SDK for autonomous fixed-overlap flight paths (the Mini 3 had one), so survey-grade flight automation was a known limitation. Camera intrinsics on record: focal 5.067 mm, sensor 7.6 × 4.275 mm.
- Coordinate metadata is stored in **XMP, not standard EXIF** on the Mini 4 Pro; an XMP→EXIF conversion was required before WebODM. CRS is WGS84 / UTM 43N.
- Imprecise EXIF georeferencing plus wind/terrain-driven yaw/pitch/roll introduce orthomosaic distortion; residual positional offsets of **~5–7 m** remained even after EXIF-corner placement. This is the sourced motivation for the residual-alignment step (§4.3) and a number usable in §6.2.
- External GPS/RTK was investigated and rejected on cost/accuracy grounds (affordable receivers ~2 m), reinforcing image-based alignment over field-GPS precision.
- Drone-mapping collaboration involved Craig Dsouza and SS Jayakrishna; acknowledge as appropriate.

Need from user:

- exact flight count and dates;
- whether all sites used the same altitude/overlap;
- whether the 50-80 m range should be narrowed in final text.

### 3.3 Photogrammetric Processing

Write:

- Raw UAV images processed in WebODM.
- Outputs include georeferenced RGB orthomosaics.
- Orthomosaics are exported as GeoTIFFs.
- CRS and geospatial metadata are preserved for crown polygons, field-label joins, and any future geospatial linkage.

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

### 3.5 Dataset Inventory Notes

Keep this focused on UAV/field data, not satellite classifier data.

Write:

- The main text should summarize LHC and SIT at a high level.
- Appendix D should carry the full orthomosaic/date inventory and dataset manifests.
- The appendix can include dataset versioning and reproducibility metadata.
- Satellite/species classifier datasets are deferred to the future short paper, not documented in detail here.

Need from user later:

- final LHC orthomosaic names and dates;
- final SIT orthomosaic names and dates;
- final field-label table schema;
- final output folders that correspond to thesis results.

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
  -> field/species label linkage
  -> phenology interpretation and visual QA
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

Code-verified settings (resolve discrepancies before writing final numbers):

- **Tile size / buffer (discrepancy):** the *production pipeline* defaults to **25 × 25 m tiles, 15 m buffer** (`00_discover_oms.py`). The earlier *notebook era* used **45 × 45 m tiles, 20 m buffer** (weekly meeting 2025-03-11; notebook-analysis doc). Pick the values from the run that produced the final tables — do NOT inherit the 40 m/buffer-30 figure from the overlapping companion group's text.
- simplify tolerance 0.3, fixed IoU 0.7 for `clean_crowns` (`01_crown_detection.py`).
- thresholds `conf_0p15` … `conf_0p65` in 0.05 steps (11 layers).
- base tracking threshold: **`conf_0p45` for LHC**, **`conf_0p15` for SIT** (SIT's `conf_0p45` is too sparse — README documents this).
- high-confidence alignment threshold `conf_0p65`.
- model: `250312_flexi.pth` (the "flexi" model adopted Oct 2025 for more consistent crown sizes).
- Settings DO vary by site (base threshold). State per-site settings in Table 1 / Appendix.

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

Resolved from code (`tree_tracking.py::align_to_reference_with_method`):

- Reference is OM1. Each OM is aligned to the **previous** OM via phase cross-correlation, and the step shifts are **accumulated** (`shift_t = shift_{t-1} + step_t`), i.e. consecutive/cumulative to OM1 — not each OM registered directly to OM1.
- `pcc_tiled` is the default and final method. It splits the overlap into a 4×4 tile grid, runs PCC per tile, gates tiles on texture/error/max-shift, then takes a **MAD-inlier median** of the per-tile shifts (falls back to whole-image PCC if too few valid tiles). Alternatives `pcc`, `ecc`, `crowns` (centroid), `orb_affine` also exist.
- Step 2 saves the per-OM shifts into `pipeline_config.json`; Steps 3–4 reuse those exact shifts (no re-registration), so crop sampling is consistent with tracking.

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

Resolved from code (`tree_tracking.py`, `02_crown_tracking.py`):

- **Full vs partial inclusion rule:** consensus sources = full width-1 chains + extracted backbones of branching full chains, plus partial chains with `len >= min_partial_len` (default 5) AND a one-to-one edge ratio `>= min_partial_one_to_one_ratio` (default 0.9). See `select_consensus_source_chains`.
- **How multi-threshold actually helps tracking (IMPORTANT correction):** the production Step 2 does NOT call the explicit predicted-position gap-filler (`augment_partial_chains_with_multithreshold` exists but is not invoked). Instead, `load_multithreshold_data` builds the candidate population so that OM1 uses only the base layer while every later OM uses the **union of all threshold layers** (`conf_0p15…conf_0p65`), de-duplicated by geometry. So lower-confidence detections enter tracking as a richer candidate pool, not as targeted gap fills. Section 4.5.4 / paper §4.5.4 should describe this union mechanism (the paper.md heading is now "Gap Filling via Multi-Threshold Candidates").
- **Virtual nodes / provenance:** the gap-fill path does tag appended crowns with `is_augmented=True` and `case="gap_fill"`, but since it is not run in production there are no virtual/augmented nodes in the current outputs. If the stronger gap-fill story is wanted, wire `augment_partial_chains_with_multithreshold` into Step 2 and regenerate.

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

Code-verified weights (`consensus_medoid`): `w_d = 0.5` (centroid distance, normalized by the max pairwise centroid distance in the chain), `w_iou = 0.4` (on `1 - IoU`), `w_a = 0.1` (on `1 - area_ratio`). Dedup after consensus (`deduplicate_crowns`): drop by IoU `> 0.75` and by containment (buffer 5.0), larger-area-first with chain-length/avg-similarity tiebreak.

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

Code-verified (`phenology_leafshed.compute_leafshed_scores`):

- per-crown veg/GCC/texture series are linearly interpolated over missing/bad dates, then amplitudes are normalized by **A90** (the 90th-percentile amplitude across all crowns);
- deciduous score `DS = 0.35·veg_amp + 0.30·depth + 0.25·gcc_amp + 0.10·texture_amp`;
- `depth = (tau_veg - min_veg)/tau_veg` when `min_veg < tau_veg` (else 0), with `tau_veg = veg_min_threshold`, default 0.45;
- deciduous if `DS >= ds_threshold` — **pipeline Step 3 default is 0.70**; the dataclass default is 0.85, so state the actual run value;
- phenophase on min-max-normalized veg: `leaf_on >= 0.65`, `leaf_off <= 0.35`, transitioning in between, `stable` for non-deciduous crowns;
- events: `leaf_off_start_om`, `full_leaf_off_om` (veg trough), `leaf_on_return_om`.
- A non-tree filter (`apply_non_tree_thresholds`) exists but Step 3 disables it — do not claim automated non-tree removal as a result.

Be honest:

> The rule-based phenology score is an interpretable baseline designed for low-label settings. It should be treated as a proxy until validated against denser field phenology labels.

### 4.12 Field Label and Species Integration

This supports ecological interpretation of UAV-derived phenology, not a main-paper satellite classifier claim.

Write:

- Consensus crown IDs become the primary keys for field labels.
- QField/manual labels are joined to crown geometries.
- Labels may include species, genus, deciduousness, flowering category, showy flower color, or other ecological traits.
- Outputs can be visualized in Google Earth or an HTML viewer for review.

Need from user:

- final label columns and definitions;
- final number of field-verified labeled crowns;
- how labels should be summarized in the main paper.

### 4.13 Interactive Visualization and Quality Assurance

Write:

- The viewer links consensus crowns, per-date detections, crop sequences, and phenology features.
- Interactive inspection is used to review tracking errors, split/merge cases, missing detections, and suspicious phenology trajectories.
- Google Earth/KML/KMZ exports support field-facing review and communication.
- This is part of the method because the pipeline produces objects that need visual QA before biological interpretation.

Potential outputs:

- clickable map of consensus crowns;
- before/after alignment overlays;
- chain visualizations;
- crop strips by tree and date;
- plots of GCC/RCC/vegetation fraction and leaf-state labels.

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

Evaluate whether consensus crowns improve sampling stability and spatially correspond to hand-annotated crown polygons.

Primary external evaluation:

- Use the hand-annotated polygon ground truth for the SIT area.
- Compare final SIT consensus crowns against the SIT manual crown polygons.
- This should be the strongest evaluation for consensus geometry because it checks whether the stable crown objects are not only internally consistent, but also spatially meaningful relative to human annotations.

Matching protocol:

- Use polygon IoU between each consensus crown and each hand-annotated SIT crown.
- Match predicted consensus crowns to GT crowns using one-to-one assignment, preferably Hungarian matching or greedy matching sorted by IoU.
- Report results at one or more IoU thresholds, for example `0.3`, `0.5`, and possibly `0.75`.
- Keep unmatched consensus crowns as false positives and unmatched GT crowns as false negatives.
- For matched crowns, report the matched-IoU distribution.

GT comparison metrics:

- number of hand-annotated SIT GT crowns;
- number of final SIT consensus crowns;
- true positives, false positives, false negatives;
- precision, recall, and F1 at selected IoU thresholds;
- mean, median, and percentile matched IoU;
- centroid distance between matched GT and consensus crowns;
- area ratio between matched GT and consensus crowns;
- error categories such as split, merge, missing crown, duplicate consensus, poor boundary overlap, and non-tree false positive.

Possible metrics:

- crop availability across dates;
- valid pixel fraction distribution;
- vegetation feature smoothness;
- reduced sudden jumps from segmentation area changes;
- visual examples comparing per-date polygons vs medoid consensus.
- comparison of consensus-vs-GT IoU against per-date detection-vs-GT IoU, if we want to show that consensus crowns are a better final product than individual-date detections.

Recommended figure:

- one tree row:
  - date-wise crown detections;
  - consensus crown overlay;
  - hand-annotated SIT GT polygon overlay;
  - crop sequence;
  - GCC/vegetation fraction time series.

Recommended table:

| Site | GT crowns | Consensus crowns | IoU threshold | TP | FP | FN | Precision | Recall | F1 | Mean matched IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIT | TBD | TBD | 0.3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SIT | TBD | TBD | 0.5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

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

## 6.7 Interactive Visualization and Error Analysis

This should be a real main-paper results section because the viewer was central to understanding whether the pipeline output was biologically and geometrically plausible.

Report:

- representative tracking viewer screenshots;
- crown crop strips for stable chains, partial chains, split/merge cases, and failure cases;
- before/after examples where alignment improves interpretability;
- examples where consensus crowns stabilize noisy per-date polygons;
- qualitative error taxonomy from visual review.

Possible categories:

- correct stable identity;
- missed detection;
- split crown;
- merged crown;
- poor overlap or distorted orthomosaic region;
- shadow/illumination contaminated crop;
- non-tree or weak vegetation object.

Keep satellite/species classifier results out of this main results section. Save them for the separate short paper.

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

### 7.5 Field Labels and Crown-Level Ecological Interpretation

This should be substantial enough to show that the tracked crowns are useful ecological objects, without making satellite/species classifiers part of the main claim.

Discuss:

- individual-tree drone identities become stable units for field label joins;
- species labels turn crown tracks into ecological objects;
- field labels help interpret phenology trajectories;
- species/trait summaries can be reported descriptively;
- satellite/species classifiers are better handled in a future short paper.

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
- species/trait summaries depend on label quality and crown-label joins;
- results may not generalize across forest types without retuning.

### 7.8 Future Work

Include:

- manual temporal identity benchmark;
- stronger phenology field validation;
- learned temporal association model;
- non-tree and shadow filtering;
- active learning for uncertain chains;
- separate satellite/species short paper using spatially blocked validation;
- species-aware or phenology-aware tracking priors.

---

## 8. Conclusion

Draft conclusion paragraph:

This work demonstrates that individual-tree phenology monitoring from repeated UAV RGB orthomosaics depends on preserving tree identity through time. By combining multi-threshold crown segmentation, residual orthomosaic alignment, graph-based temporal association, gap filling, consensus crown generation, and crown-level feature extraction, the proposed pipeline converts independent orthomosaics into a structured dataset of tracked tree crowns and phenology trajectories. The resulting crown identity layer supports visual inspection, field validation, species linkage, and crown-level ecological interpretation. The study therefore reframes UAV crown delineation as part of a larger temporal monitoring problem and provides a practical foundation for fine-scale drone-based phenology analysis.

---

## 9. Figure Plan

### Figure 1: Overall Workflow

Show:

```text
UAV survey -> WebODM orthomosaic -> Detectree2 crowns -> alignment -> graph tracking -> consensus crowns -> phenology -> field labels and visual QA
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

### Figure 8: Interactive QA and Field Label Review

Show:

- consensus crown labels;
- clickable tracking/crop viewer;
- examples of correct identity, split/merge, missing detection, and shadow-contaminated crops;
- field label or Google Earth review overlay.

Appendix-only figure:

- dataset timeline;
- site/date inventory map;
- example input/output folder manifest.

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

### Table 5: Field Label and Phenology Summary

Columns:

- species/trait group;
- number of labeled crowns;
- number of tracked consensus crowns linked;
- available phenology trajectories;
- summary statistic or example pattern.

Appendix-only table:

- full orthomosaic/date inventory;
- input/output dataset manifest;
- field-label schema;
- dataset versioning and reproducibility notes.

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
- interactive visualization and QA.

### Supplement / Appendix

Move here if too long:

- full code/config tables;
- all threshold sensitivity results;
- full orthomosaic/date inventories;
- field label schema and label inventory;
- pipeline input/output manifests;
- dataset versioning and reproducibility notes;
- detailed notebook history;
- failed methods like early DeepForest, unless used as motivation.

Important:

Satellite/species should **not** stay in the main body or become Appendix D for this paper. Save the full satellite/species argument for a later short paper once the GEE embedding results and validation are complete. Appendix D in this paper should document datasets and dates.

---

## 12. Decisions We Need Before Writing Full Draft

These are the current decisions and remaining open points before polished prose.

### Decision 1: Final Title

Resolved working title:

**Tracking Individual Tree Phenology from Repeated UAV RGB Orthomosaics Using Graph-Based Crown Association and Consensus Crowns**

### Decision 2: Main Study Site

Resolved:

- Primary study area: IIT Delhi campus.
- Primary tracking/phenology demonstration sites: LHC and SIT.
- Current full-data availability:
  - LHC: 13 orthomosaics/dates.
  - SIT: 19 orthomosaics/dates.
- Existing output summaries inspected so far are partial older runs and should be replaced by full-run results later.

### Decision 3: Final Satellite/Species Scope

Resolved for this paper (confirmed by author 2026-06-15):

- Satellite/species work moves out of this paper's main structure.
- The main paper should not rely on satellite/species classifier results.
- The strongest current direction remains Google Earth Engine embeddings, but that will become a separate short paper later.
- Main-body references should be brief and should frame satellite/species as future work.

Counterpoint on record (do not silently reopen): the weekly-meeting digest's "Suggested Paper Updates" recommends satellite/species in the **main body** (Sentinel-2 baselines + GEE embeddings + spatial-holdout evaluation), and the professor's original charter frames cross-scale satellite modeling and species classification as part of the project vision. This was reviewed and **consciously overridden** in favor of a focused drone-only contribution. If the professor pushes for inclusion, the fallback is "satellite as a secondary/appendix contribution" rather than co-equal — and the GEE embedding Acacia result plus the crown-size-vs-Sentinel-2-pixel scale-mismatch analysis are the materials to use.

Need later for the separate short paper:

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
5. Fill Appendix D with dataset/date inventories and result-folder manifests.
6. Write Introduction and Contributions.
7. Write Method as pipeline.
8. Write Results around figures/tables.
9. Write Discussion honestly around tracking, consensus crowns, phenology proxies, and operational limits.

---

## 14. Questions for User

These are the most important questions to answer next.

1. What are the exact 13 LHC orthomosaic filenames/dates in the final run?
2. What are the exact 19 SIT orthomosaic filenames/dates in the final run?
3. Which output folder will contain the latest full-data LHC/SIT results once you share them?
4. Do we have or want a small manually checked tracking-validation subset?
5. What exact field-verified label schema should the species section use?
6. Which additional project datasets, beyond LHC and SIT, should be listed in Appendix D?

---

## Appendix D Plan: Dataset and Date Inventory

Purpose:

- Make the data basis of the thesis transparent without overloading the main paper.
- Record exactly which orthomosaics, dates, labels, and output folders support the reported results.
- Preserve enough metadata to make the reported datasets and outputs traceable.

Suggested Appendix D structure:

### D.1 UAV Orthomosaic Date Inventory

Write:

- table of every orthomosaic used or available;
- site name;
- orthomosaic/date label;
- acquisition date;
- whether included in final thesis run;
- whether excluded, and why.

### D.2 LHC Dataset

Report:

- final 13 LHC orthomosaic names/dates;
- date range;
- any known quality issues;
- final output folder used for LHC results;
- notes on skipped or problematic dates.

### D.3 SIT Dataset

Report:

- final 19 SIT orthomosaic names/dates;
- date range;
- any known quality issues;
- final output folder used for SIT results;
- notes on skipped or problematic dates.

### D.4 Additional Site Datasets

Use only as inventory/context unless results enter the main paper:

- SAC or other IIT Delhi sites;
- Sanjay Van sites/spots;
- any auxiliary orthomosaics not used in the main LHC/SIT results;
- whether each dataset is used, deferred, or only historical context.

### D.5 Field and Crown Label Dataset

Report:

- label file names;
- schema/columns;
- field verification status;
- number of labeled crowns;
- join key or spatial-join method to consensus crowns;
- label-cleaning notes.

### D.6 Pipeline Input and Output Dataset Manifest

Report:

- input orthomosaic folders;
- crown store folders;
- tracking output folders;
- phenology output files;
- visualization folders;
- which output files feed each results table/figure.

### D.7 Dataset Versioning and Reproducibility Notes

Report:

- dataset version names or folder names;
- code/config version used for final runs;
- date when final outputs were generated;
- notes on files excluded from the final analysis;
- checksum or manifest information if available.
