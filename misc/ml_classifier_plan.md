# Classifier Data & Training Plan

*Last updated: 2026-05-13*

---

## Core Architecture: Drone → Labels, Satellite → Features

```
Drone orthomosaics (high-res, sparse in time, limited area)
    → Crown detection + tracking (Detectree2 + graph tracking)
    → Per-crown phenology signals: GCC, RCC, veg_fraction, texture
    → Leafshed classifier → deciduous_score, phenophases, event OMs
    → Field survey (QField) → species, flower colour, health
    ↓
    LABELS for ML training
    (leaf_off_month, deciduous_class, flower_colour, acacia/non-acacia, timing)

Satellite data (Sentinel-2, low-res, dense in time, global coverage)
    → Query per crown centroid/polygon → NDVI, EVI time series
    → Monthly medians, seasonal medians, amplitudes, multi-year stats
    ↓
    INPUT FEATURES for ML classifiers

Train: satellite features → drone-derived labels
Infer: run on any tree anywhere with Sentinel-2 coverage, no drone needed
```

**Why this architecture matters:**
- Drone flights are expensive, time-limited, and spatially limited (SIT ~5ha, LHC ~3ha)
- Satellite is free, continuous every 5 days, and global
- The drone pipeline generates a high-quality labeled dataset once; models then generalize from satellite features alone
- GCC, RCC, texture etc. from the drone are **not** model inputs — they are used internally to derive labels

---

## 1. Data Inventory

### Temporal Data (Repeated Drone Runs — Can Build Time Series)

| Site | Label | OMs | Known Dates | IITD? | Notes |
|------|-------|-----|-------------|-------|-------|
| SIT | `sit` | 14 (sit_om1 – sit_om14) | TBD — numbers only, need date map | Yes | ~14 flights, full seasonal coverage |
| LHC | `lhc` | 9 (from filenames) | Oct 25 / Nov 9 / Nov 20 / Nov 26 / Dec 9 '25, Jan 11 / Feb 4 / Feb 20 / Mar 7 '26 | Yes | Oct–Mar = dry/winter season, strong deciduousness signal |
| Sanjay Van | `sv` | TBD | TBD | No (outside IITD) | Separate vegetation community; repeat runs done |

**Note on LHC date range (Oct–Mar):** This window captures:
- leaf shedding (Nov–Jan for most deciduous IITD trees)
- peak leafless period (Dec–Feb)
- early leaf return (Feb–Mar)
- some early flowering species

### Spatial-Only Data (Single Drone Runs — No Time Series)

| Area | Label | Notes |
|------|-------|-------|
| area1 | `area1` | Single .tif, one-time flight |
| area2 | `area2` | Single .tif |
| area3 | `area3` | Single .tif |
| area6 | `area6` | Single .tif |
| area5? | TBD | User mentioned 5 areas; may still be incoming |

**What single-run areas can contribute:**
- Crown geometry + crown-level appearance features (texture, colour, NDVI)
- If the flight date happened to be during a distinctive phenological event (flowering, peak leaf-off), those features are usable
- Cannot contribute to timing classifiers (no time series)
- Can contribute to: Acacia/Non-Acacia, Flower Colour (if flight timing was right), Evergreen/Deciduous appearance-only features

---

## 2. Foundational Prerequisites (Needed Before Any Classifier)

These are shared requirements across all classifiers. Build them once.

### P1 — OM-to-Date Table
A single CSV or JSON mapping `(site, om_id)` → `(flight_date, calendar_month, fortnight)`.

| site | om_id | flight_date | month | fortnight |
|------|-------|-------------|-------|-----------|
| sit | 1 | 2025-XX-XX | Nov | Nov-1 |
| lhc | 1 | 2025-10-25 | Oct | Oct-2 |
| lhc | 2 | 2025-11-09 | Nov | Nov-1 |
| sv | 1 | TBD | ... | ... |

**Why needed:** The regression classifiers predict "month/fortnight when event occurs." Without this, leaf_off_start_om = 3 is meaningless. With it, leaf_off_start_om = 3 → "November, Fortnight 1."

**Status:** LHC dates parseable from filenames (already in notebook). SIT dates: need to add manually or retrieve from flight logs.

### P2 — Per-Site Ground Truth Species Table
One row per tree per site, from QField ground visits:

| site | chain_id | species | common_name | deciduous_class | flowers | flower_colour | flower_month | notes |
|------|----------|---------|-------------|-----------------|---------|---------------|-------------|-------|
| sit | sit_chain_00042 | Acacia nilotica | Babul | Deciduous | Yes | Yellow | Nov | prominent, visible from above |
| lhc | lhc_chain_00011 | Peltophorum pterocarpum | Copper pod | Semi-evergreen | Yes | Yellow | Apr–Jun | TBD if visible in drone |

**Why needed:** This is the source of training labels for all 6 classifiers. The drone sees spectral signals; this table tells us what those signals mean biologically.

**Status:** Partially available from QField field visits (in `ground_visit.js`). Needs review and completion, especially for:
- Species with uncertain deciduousness class (semi-evergreen edge cases)
- Flowering visibility from drone (the "possibly visible" cases professor mentioned)
- Trees that are field-matched but species not yet identified

### P3 — Per-Classifier Species Inclusion/Exclusion Lists
Following professor's guidance: for each classifier, a table of:
- Species to include in **positive set**
- Species to include in **negative set**
- Species to **discard from training** (ambiguous/too rare/not visible)

This is a grid we will iterate over during experiments. Template:

| species | common_name | flower_colour_positive | acacia_binary | deciduous_class | notes |
|---------|-------------|----------------------|---------------|-----------------|-------|
| Acacia nilotica | Babul | discard (no showy flower) | POSITIVE | Deciduous | |
| Cassia fistula | Amaltas | YELLOW positive | negative | Deciduous | flowers very visible from above |
| Delonix regia | Gulmohar | RED positive | negative | Deciduous | iconic; very visible |
| Terminalia arjuna | Arjun | discard (leaf colour change) | negative | Semi-evergreen | leaves go red-orange; confound for flower colour? |
| Eucalyptus | Eucalyptus | discard | negative | Evergreen | |
| Prosopis juliflora | Vilayati Babul | discard | POSITIVE | Semi-evergreen | |
| Pongamia pinnata | Karanj | WHITE/pink? | negative | Semi-evergreen | needs review |

*(This is a starting template; filling it in is a task — see Section 7.)*

### P4 — "Possibly Visible" Review Task
Professor noted that for some species/classifiers we need to look at drone imagery and determine: include in +ve set, include in -ve set, or discard?

**How to do this:** For each "possibly visible" species:
1. Find OMs where that species would theoretically be flowering or in a distinctive phenophase
2. Open the crop images for those trees in those OMs from the viewer
3. Visually judge: is the signal clearly visible from above?
4. Record decision in the species inclusion/exclusion table

---

## 3. Classifier Plans

---

### Classifier 1 — Flower Colour

**What it predicts:** Does this tree get red / yellow / white flowers? (and which colour)

**Type:** Multi-class (red / yellow / white / none) or separate one-vs-rest binary per colour

**Why multi-class vs one-vs-rest matters here:**
- A single tree can only have one flower colour → multi-class is appropriate if the classes are mutually exclusive
- BUT: some species have cream-yellow borders — if uncertain, one-vs-rest lets you threshold each separately
- Professor recommends experimenting with both

**Input features:**
- RCC (red chromatic coordinate) time series across OMs — peaks when red flowers appear
- GCC (green chromatic coordinate) — drops when leaves replaced by flowers
- Spectral embedding of patch crops at peak OM

**Training sites:** All IITD OMs (SIT + LHC + single-run areas). SV separately if species overlap.

**Key data requirements:**
- Which OMs contain the flowering window for each species? (Need P1 date map + botanical knowledge)
- Example: Delonix regia (Gulmohar) flowers April–June; if SIT has OMs in this window → those OMs are the positive signal
- Without the right OM, you only see green canopy — no signal for the classifier

**Training set construction rules (professor's guidance):**
1. **Positive samples:** Only trees where flowering is clearly prominent and visible from above. Discard trees that flower but aren't visible at drone altitude.
2. **Dramatic leaf colour change:** Some trees go red/orange before leaf fall (e.g., Terminalia arjuna). This will look like "red" to the classifier but isn't a flower. Decision: construct a **separate leaf colour change classifier**, or deliberately include these in training as hard negatives.
3. **"Possibly visible" cases:** Drone image review required — see P4.

**Open questions:**
- Do any SIT/LHC OMs cover April–June (peak flowering for most species)? If not, flower colour training may need future OMs.
- Are white flowers distinguishable from drone altitude at all?

---

### Classifier 2 — Flower Colour Time

**What it predicts:** In which month/fortnight does this tree flower (for each colour)?

**Type:** Regression — output is a continuous month number or discrete fortnight label

**Input features:** Same as Classifier 1, plus OM-to-date mapping to anchor when the spectral signal occurs

**Training sites:** SIT + LHC OMs. Maybe SV.

**Key data requirements:**
- Needs P1 (OM-to-date table) — critical; can't train without it
- Need to know, per species, which OMs show the flowering peak
- Training target: `flower_peak_month` or `flower_peak_fortnight` (per tree)

**Training set construction:**
- Only deciduous/flowering species with clearly visible flowers from drone
- Only use OMs in the flowering window as positive observation points
- Need enough temporal coverage (multiple OMs around the flowering event)

**Challenge:** LHC date range is Oct–Mar only. If most flowering happens Apr–Jun, we have no training signal for this classifier from current data.

---

### Classifier 3 — Leaf-off Start / Peak Time

**What it predicts:** In which month/fortnight does this tree start / peak leaf shedding?

**Type:** Regression — output is `leaf_off_start_month`, `leaf_off_peak_month`

**Input features:**
- `veg_fraction_hsv` time series (normalized)
- `gcc_mean` time series
- `deciduous_score`, `veg_amplitude`
- `phenophases` label sequence (indirect feature)

**Training sites:** SIT + LHC. Maybe SV.

**Key data requirements:**
- Needs P1 (OM-to-date table) to convert `leaf_off_start_om` → `leaf_off_start_month`
- `leaf_off_start_om` and `leaf_off_peak_om` already computed and stored in master GeoJSON!
- P2 (species table) to know which species are truly deciduous

**Training set construction:**
- Only deciduous trees (where shedding is a real biological event)
- Exclude non-tree and evergreen crowns
- For semi-evergreen: experiment with including/excluding (following prof's grid search approach)
- Per-species filtering: some species may have very noisy signals — list them in P3

**This classifier is closest to ready** because we already have the raw training targets.

---

### Classifier 4 — Leaf-on Start / Completed Time

**What it predicts:** In which month/fortnight does this tree start / complete leaf regrowth?

**Type:** Regression — output is `leaf_on_start_month`, `leaf_on_complete_month`

Same structure as Classifier 3, using `leaf_on_return_om` already computed.

**Additional complication:** "Leaf-on start" and "leaf-on complete" are two separate events. We currently only store `leaf_on_return_om` (complete). Need to decide if we can detect "leaf-on start" separately from the phenophase sequence.

**Action:** Derive `leaf_on_start_om` from phenophase sequence — the first OM with `transitioning` after the trough.

---

### Classifier 5 — Evergreen / Semi-Evergreen / Deciduous

**What it predicts:** Which of the 3 classes does this tree belong to?

**Type:** Multi-class (3 classes) or one-vs-rest

**Input features:**
- `deciduous_score`, `veg_amplitude`, `gcc_amplitude`, `min_veg`
- `phenophases` label sequence
- Spectral summary statistics

**Training sites:** All IITD OMs together (SIT + LHC + single-run areas)

**Key data requirements:**
- Need 3-class labels per tree — `field_species` → expert mapped to evergreen/semi-evergreen/deciduous
- **Semi-evergreen is the hard case:** Partially sheds leaves but not completely. The `deciduous_score` likely sits in a middle range. Need to define threshold carefully.
- Current `is_deciduous` is binary — this classifier generalizes it to 3 classes

**Training set construction (professor's guidance):**
- Clear deciduous and clear evergreen are easy — high confidence positives
- Semi-evergreen: requires careful species-level annotation. Some species are "semi-evergreen in some climates, deciduous in others" — for IITD specifically, may need visual review
- Grid search: experiment with which species to include in semi-evergreen set vs merge into deciduous/evergreen

**Single-run area contribution:** 
- If flight happened at peak leaf-off season → can distinguish evergreen (still green) vs deciduous (bare)
- Appearance at one timepoint is weaker evidence; use as supplementary training data with lower weight

---

### Classifier 6 — Acacia / Non-Acacia

**What it predicts:** Is this tree an Acacia species or not?

**Type:** Binary classifier

**Input features:**
- Crown texture (Acacia has fine feathery leaves → distinct texture)
- `veg_fraction_hsv`, `gcc_mean`, `rcc_mean` time series
- Spectral crop embeddings
- Leaf-off timing (many Acacias are semi-evergreen with different shedding patterns than broad-leaved deciduous trees)

**Training sites:** All IITD OMs + SV

**Why SV is included:** If SV has additional Acacia trees with field labels, this increases positive sample count

**Key data requirements:**
- P2 (species table): `field_species` must be populated for training trees
- Acacia is a genus — includes Acacia nilotica, Prosopis juliflora (Vilayati Babul — debated), Acacia auriculiformis, etc. Need to decide which genera count as "Acacia" for this classifier
- Single-run areas can contribute if species labels are available and flight captured distinctive crown appearance

**Training set construction:**
- Positive: confirmed Acacia species with clear crown detections
- Negative: confirmed non-Acacia with clear crowns (Eucalyptus, Ficus, Terminalia, Delonix, Neem, etc.)
- Discard: ambiguous trees, unlabeled crowns, very small crowns

---

## 4. Satellite Pipeline Status

**The satellite feature extraction pipeline has been started** (`src/notebooks/sat_data6May26.ipynb`, `sat_data.ipynb`). Current state:

### What's already been built:
- Queries Sentinel-2 L2A (STAC API) for the SIT bounding box, years 2022–2024
- SCL-mask filtering: removes cloud, shadow, saturated pixels per observation
- Per-crown NDVI and EVI extraction on each valid Sentinel-2 date
- Validity flags: `is_valid_strict`, `is_valid_relaxed` based on `valid_fraction`
- Aggregation to per-crown model features (see below)
- Drone labels already joined into the feature table: `is_deciduous`, `deciduous_score`, `leaf_off_start_om`, `full_leaf_off_om`, `leaf_on_return_om`

### Existing feature columns in `crown_level_model_features.csv`:
```
chain_id, n_obs_strict, n_years
ndvi_mean, ndvi_std, ndvi_min, ndvi_max, ndvi_amp
evi_mean, evi_std, evi_min, evi_max, evi_amp
valid_fraction_median
ndvi_late_winter_median, evi_late_winter_median
ndvi_pre_monsoon_median, evi_pre_monsoon_median
ndvi_monsoon_median, evi_monsoon_median
ndvi_post_monsoon_median, evi_post_monsoon_median
ndvi_month_01 ... ndvi_month_12  (monthly medians)
--- joined drone labels ---
is_deciduous, deciduous_score
leaf_off_start_om, full_leaf_off_om, leaf_on_return_om
```

### What's in `satellite_observations_all.csv` (raw per-date data):
```
date, datetime, year, iso_year, iso_week, month
item_id, cloud_cover, chain_id
ndvi_mean, evi_mean, valid_pixels, total_pixels, valid_fraction
is_valid_relaxed, is_valid_strict
--- joined drone labels ---
is_deciduous, deciduous_score, leaf_off_start_om, full_leaf_off_om, leaf_on_return_om
```

### Current scope:
- SIT only, 126 filtered crowns, 20 sampled so far (experimental)
- 2541 observation rows (all dates × crowns)
- 3 years of Sentinel-2 data pulled
- LHC, SV, single-run areas: not yet queried

### What still needs to be built:
- [ ] Run on full 126 crowns (not just 20-sample)
- [ ] Run for LHC crowns
- [ ] Add `rcc`-equivalent features from Sentinel-2 red band (for flower detection when flowering OMs exist)
- [ ] Add SAR features from Sentinel-1 (a separate run exists: `sit_sentinel1_6May26/`)
- [ ] Convert `leaf_off_start_om` → `leaf_off_start_month` using OM date map (T1 below)
- [ ] Add `flower_colour` and `acacia` label columns once species table is complete

---

## 5. Cross-Classifier Data Flow

```
Drone OMs (SIT / LHC / SV)
        │
        ▼
Crown detection + graph tracking → consensus crowns
        │
        ├──► Drone phenology features (GCC, RCC, veg_fraction, texture per OM)
        │         │
        │         ▼
        │    Leafshed classifier (phenology_leafshed.py)
        │    → deciduous_score, is_deciduous, phenophases
        │    → leaf_off_start_om, full_leaf_off_om, leaf_on_return_om
        │
        ├──► Field survey (QField) → species, flower colour, health
        │
        ▼
Master GeoJSON (one feature per tree, all labels merged) ← SOURCE OF TRUTH
        │
        ▼
Satellite pipeline (sat_data6May26.ipynb)
        │  Input: crown polygon/centroid from master GeoJSON
        │  Query: Sentinel-2 STAC, multi-year
        │  Output: NDVI/EVI time series + monthly/seasonal aggregates
        │
        ▼
crown_level_model_features.csv (X = sat features, Y = drone labels)
        │
        ├──► Classifier 3: Leaf-off start/peak timing
        ├──► Classifier 4: Leaf-on timing
        ├──► Classifier 5: Evergreen/Semi/Deciduous
        ├──► Classifier 6: Acacia/Non-Acacia
        │
        │    [Requires future drone OMs in Apr–Jun + satellite during flowering]
        ├──► Classifier 1: Flower Colour
        └──► Classifier 2: Flower Colour Timing
```

---

## 6. Feature Sets Per Classifier (Corrected)

| Classifier | Input Features (Satellite) | Labels (From Drone) | Status |
|-----------|---------------------------|---------------------|--------|
| Flower Colour | ndvi monthly profile, seasonal medians, red band statistics at flowering months | `flower_colour` per species from field survey | ❌ Labels not yet in feature table; needs flowering-season OMs |
| Flower Colour Timing | ndvi/evi month-by-month profile, red band peak month | `flower_peak_month` per species | ❌ Needs OM date map + flowering OMs |
| Leaf-off Start/Peak | ndvi_month_01–12, ndvi_amp, evi_amp, seasonal medians | `leaf_off_start_om` → converted to month | ⚠️ Features ready; need OM→date map to convert label to month |
| Leaf-on Timing | ndvi_month_01–12, seasonal medians | `leaf_on_return_om` → converted to month | ⚠️ Same — features ready, label conversion pending |
| Evergreen/Semi/Deciduous | ndvi_amp, evi_amp, ndvi_min, monthly profiles | `is_deciduous` (binary now), extended to 3-class via species table | ⚠️ Binary label exists; semi-evergreen 3rd class needs species annotation |
| Acacia/Non-Acacia | ndvi/evi time series, seasonal patterns, SAR (Sentinel-1) | `is_acacia` from field_species | ❌ `is_acacia` label column not yet added; needs species table |

---

## 7. Grid Search / Experiment Axes

Following professor's suggestion to systematically experiment:

| Axis | What to vary | Why |
|------|-------------|-----|
| Species inclusion | Which species in +ve/-ve set per classifier | Some species will hurt accuracy; find best subset |
| Semi-evergreen treatment | Include in deciduous class / as own class / discard | Class boundary is fuzzy |
| Leaf colour change trees | Treat as flower colour positive / hard negative / separate class | Risk of confounding Classifier 1 |
| Single-run areas | Include / exclude from training | Different coverage, seasonal snapshot only |
| Feature set | Raw time series vs amplitude summaries vs embeddings | Model sensitivity |
| Positive set strictness | All flowering trees vs only prominent ones (professor's note) | Noisy positives hurt precision |

Record all experiment configs and scores in a tracking spreadsheet. One row = one experiment config.

---

## 8. Immediate Task List

**Data / labeling tasks (required before any model training):**

- [ ] **T1:** Complete OM-to-date table for SIT (look up flight logs; LHC already parseable from filenames). Required to convert `leaf_off_start_om` → `leaf_off_start_month`.
- [ ] **T2:** Complete `field_species` for all field-matched trees in SIT and LHC (review QField data gaps)
- [ ] **T3:** Create species → phenological class mapping table (deciduous / semi-evergreen / evergreen, flower colour if any, typical flowering month for IITD)
- [ ] **T4:** Map species → `is_acacia` binary label; add to master GeoJSON and satellite feature table
- [ ] **T5:** "Possibly visible" review — for each candidate flowering species, open drone crops in viewer and decide +ve / -ve / discard per classifier
- [ ] **T6:** For single-run areas (area1–area6): check flight dates vs phenological calendar; determine which classifiers they can contribute labels to
- [ ] **T7:** Build initial per-classifier species inclusion/exclusion table (P3 above); will be iterated

**Satellite pipeline tasks:**

- [ ] **T8:** Run satellite extraction on all 126 SIT crowns (not just 20-sample)
- [ ] **T9:** Run satellite extraction for LHC crowns
- [ ] **T10:** Add OM-to-date conversion to satellite feature table (`leaf_off_start_month` etc.) once T1 is done
- [ ] **T11:** Add `flower_colour` and `is_acacia` label columns to feature table once T3/T4 done
- [ ] **T12:** Evaluate Sentinel-1 SAR features (`sit_sentinel1_6May26/`) for Acacia classifier

**Drone pipeline tasks:**

- [ ] **T13:** Derive `leaf_on_start_om` from phenophase sequence (first OM with `transitioning` after trough) — currently only `leaf_on_return_om` (complete) is stored
- [ ] **T14:** Integrate SV (Sanjay Van) into tracking + phenology pipeline
- [ ] **T15:** Add `date_iso` and `fortnight` to phenology CSVs and master GeoJSON metadata

---

## 9. What We're Still Missing (Data Gaps)

| Gap | Impact | Resolution |
|-----|--------|------------|
| SIT OM flight dates not in filenames | Cannot convert OM IDs to months; timing classifiers blocked | Look up flight logs; add date map to notebook config |
| No OMs covering Apr–Jun flowering season | Classifiers 1 & 2 (flower colour/timing) cannot be trained | Plan future drone flights Apr–Jun |
| Satellite run covers only 20 crowns | Can't evaluate model on full SIT dataset | Run full 126 crowns (T8) |
| LHC not yet in satellite pipeline | Training set is only SIT | Run LHC extraction (T9) |
| SV not yet in drone pipeline | Can't use SV for Acacia classifier | Run SV through tracking + phenology pipeline (T14) |
| Species labels incomplete for many crowns | Small training sets, especially for rare species | Complete QField annotation; fill gaps via visual review |
| Semi-evergreen class not defined | Classifier 5 is currently binary only | Species-level expert annotations + visual validation |
| `is_acacia` label not in feature table | Classifier 6 cannot be trained | Complete species table (T3) and add column (T11) |
