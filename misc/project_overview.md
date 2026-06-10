# Project Understanding & Work Plan
### Drone Phenology Monitoring — IITD Urban Forest

*Prepared: 2026-05-13*

---

## 1. What This Project Is Actually Doing

The core idea is: **use expensive high-quality drone observations to label trees once, then train cheap satellite-based models that work anywhere**.

```
Drone flights (expensive, high-res, spatially limited, sparse in time)
         ↓
Crown detection + multi-flight tracking
         ↓
Per-crown phenology signals: GCC, RCC, vegetation fraction, texture
         ↓
Leafshed classifier → deciduousness score, leaf-off/on event months
+ Field survey (QField) → species, flower colour, health
         ↓
   ┌─────────────────────────────────────┐
   │  LABELS  (the "ground truth")       │
   │  • is_deciduous / deciduous_score   │
   │  • leaf_off_start_month             │
   │  • flower_colour                    │
   │  • is_acacia                        │
   └────────────────┬────────────────────┘
                    │
Satellite: Sentinel-2 + Sentinel-1
(free, 5-day repeat, global, 10m resolution)
         ↓
Per-crown NDVI/EVI time series, monthly profiles,
SAR backscatter (VV/VH) — structural proxy
         ↓
   ┌─────────────────────────────────────┐
   │  INPUT FEATURES  for ML classifiers │
   └────────────────┬────────────────────┘
                    │
   Train: satellite features → drone-derived labels
   Infer: run on ANY tree with satellite coverage — no drone needed
```

**Why this matters:** The drone pipeline generates a high-quality labeled dataset across IITD once. The trained models then generalize — you could run them on any urban forest in India from satellite data alone.

GCC/RCC/texture from drone are **not** model inputs — they are used internally to derive the training labels.

---

## 2. What Has Already Been Built

### 2.1 Drone Pipeline (Complete for SIT and LHC)

| Component | Status | Output |
|-----------|--------|--------|
| Crown detection | ✅ Done (Detectree2) | Per-OM crown polygons |
| Multi-flight tracking | ✅ Done (graph-based, IoU + centroid + overlap similarity) | `consensus_crowns` GPKG |
| Phenology feature extraction | ✅ Done | GCC, RCC, veg_fraction, texture per crown per OM |
| Leafshed classifier | ✅ Done | `deciduous_score`, `is_deciduous`, `phenophases`, `leaf_off_start_om`, `full_leaf_off_om`, `leaf_on_return_om` |
| Master GeoJSON export | ✅ Done | `consensus_crowns_om1_phenology_sit.geojson` — one row per tracked tree, all labels merged |
| Interactive viewer | ✅ Done | Leaflet-based viewer with per-OM crown overlays, visible from browser |

**SIT:** 131 raw crowns → 126 filtered → all with tracking and phenology labels.  
**LHC:** 9 OMs, dates span Oct 2025 – Mar 2026 (captures full dry-season leaf cycle).

### 2.2 Satellite Pipeline (Partially Complete)

| Component | Status | File |
|-----------|--------|------|
| Sentinel-2 STAC query (2022–2024) | ✅ Done | `sat_data6May26.ipynb` |
| SCL cloud/shadow masking | ✅ Done | per-observation `valid_fraction` |
| NDVI + EVI extraction per crown | ✅ Done | `satellite_observations_all.csv` |
| Monthly + seasonal aggregation | ✅ Done | `crown_level_model_features.csv` |
| Drone labels joined to feature table | ✅ Done | `is_deciduous`, `deciduous_score`, leaf event OMs all joined |
| Sentinel-1 SAR features (VV/VH backscatter) | ✅ Done | `sentinel1_crown_level_features.csv` |
| **Sentinel-1 + Sentinel-2 fused feature table** | ✅ Done | `sentinel1_sentinel2_fused_features.csv` |
| Full 126-crown run (currently 20-crown sample) | ⚠️ Pending | Scale-up task T8 |
| LHC satellite extraction | ⚠️ Pending | T9 |

**Key point:** The fused feature table already has SAR + optical features joined to drone labels. This is the direct input to classifiers 3, 4, 5, and partially 6.

### 2.3 Feature Table Contents (Ready for Training)

**Sentinel-2 features (per crown):**
- `ndvi_mean/std/min/max/amp`, `evi_mean/std/min/max/amp`
- Monthly medians: `ndvi_month_01` through `ndvi_month_12` (and EVI)
- Seasonal medians: `late_winter`, `pre_monsoon`, `monsoon`, `post_monsoon`

**Sentinel-1 SAR features (per crown):**
- `vv_db_mean/std/amp` — VV backscatter (sensitive to canopy structure)
- `vh_db_mean/std/amp` — VH backscatter
- `vh_vv_ratio_mean/std` — ratio (proxy for canopy complexity)
- `vh_minus_vv_db_mean/std`

**Drone-derived labels joined:**
- `is_deciduous`, `deciduous_score`
- `leaf_off_start_om`, `full_leaf_off_om`, `leaf_on_return_om`

### 2.4 Validation Set

`input/ground_truth.json` contains **124 manually annotated tree crowns** in COCO segmentation format (from a single large orthomosaic). This is used for evaluating the crown detection step (Detectree2 IoU metrics).

---

## 3. The 6 Classifiers: Plans and Readiness

### Classifier 1 — Flower Colour (Red / Yellow / White / None)

**Input:** Satellite NDVI/EVI monthly profiles + red-band time series at flowering months  
**Labels:** `flower_colour` per species (from field survey + botanical knowledge)  
**Readiness:** ❌ **Blocked**

Missing:
- No drone OMs in Apr–Jun (peak flowering season for most IITD deciduous trees)
- Current OMs cover Oct–Mar (LHC) and various months (SIT), but flowering season not well represented
- Species-level `flower_colour` label not yet in feature table

**Path forward:** Plan drone flights in April–June; complete species annotation table.

---

### Classifier 2 — Flower Colour Timing

**Input:** Same as Classifier 1 + OM-to-date mapping  
**Labels:** `flower_peak_month` per species  
**Readiness:** ❌ **Blocked** (same reasons as Classifier 1)

---

### Classifier 3 — Leaf-off Start / Peak Month

**Input:** `ndvi_month_01–12`, `ndvi_amp`, `evi_amp`, seasonal medians  
**Labels:** `leaf_off_start_om` → converted to calendar month via OM date map  
**Readiness:** ⚠️ **Nearly ready**

What's done:
- All satellite features exist in the fused table
- `leaf_off_start_om` is already computed for all tracked trees

What's needed:
- **T1:** SIT OM-to-date map (LHC already has dates in filenames; SIT needs flight logs)
- Convert `leaf_off_start_om` → `leaf_off_start_month` and add to feature table
- Increase sample from 20 → 126 crowns (T8)

**This is the closest classifier to being trainable.** With 1–2 days of data work (T1 + T8), we could run a first model.

---

### Classifier 4 — Leaf-on Return Month

**Input:** Same as Classifier 3  
**Labels:** `leaf_on_return_om` → converted to calendar month  
**Readiness:** ⚠️ **Nearly ready** (same prerequisite as Classifier 3)

Additional note: currently only `leaf_on_return_om` (completed regrowth) is stored. Deriving `leaf_on_start_om` (first transitioning OM after trough) is a one-pass computation on the existing phenophase sequences.

---

### Classifier 5 — Evergreen / Semi-Evergreen / Deciduous

**Input:** `ndvi_amp`, `evi_amp`, `ndvi_min`, `ndvi_month_01–12` profiles  
**Labels:** 3-class label per tree from species table  
**Readiness:** ⚠️ **Binary version ready; 3-class needs work**

What's done:
- `is_deciduous` (binary) already joined in the feature table
- `deciduous_score` (continuous 0–1) is there — this can already distinguish evergreen (near 0) from deciduous (near 1)

What's needed:
- Define semi-evergreen threshold on `deciduous_score` — or better, annotate directly per species
- Complete P2 (species table) with deciduous class column
- Professor's suggestion: experiment with including/excluding semi-evergreen species in grid search

---

### Classifier 6 — Acacia / Non-Acacia

**Input:** Satellite NDVI/EVI profiles + **Sentinel-1 SAR features** (VV/VH backscatter)  
**Labels:** `is_acacia` binary per tree from species table  
**Readiness:** ❌ **Features exist, label missing**

What's done:
- Fused Sentinel-1 + Sentinel-2 feature table already built
- SAR features are specifically useful here: Acacia has fine feathery leaves → distinct backscatter signature

What's needed:
- Complete P2 (species table) with genus annotations
- Add `is_acacia` column to feature table (Acacia nilotica, Acacia auriculiformis, possibly Prosopis — need to decide the genus boundary)
- SV (Sanjay Van) could increase Acacia positive samples

---

## 4. Prioritized Task Sequence

Tasks are ordered by dependencies and impact. Doing T1–T5 unblocks model training.

### Phase 1: Unlock the trainable classifiers (Classifiers 3, 4, 5)

| Task | What | Why urgent |
|------|------|-----------|
| **T1** | Build SIT OM-to-date map | Converts OM IDs to calendar months; unlocks Classifiers 3 & 4 labels |
| **T2** | Complete `field_species` for all matched trees in SIT + LHC | Source of all 6 classifier labels |
| **T3** | Create species → phenological class table | Maps species names to: deciduous/semi/evergreen, flower_colour, typical flowering month |
| **T8** | Scale satellite extraction from 20 → 126 SIT crowns | Full training set for SIT |
| **T9** | Run satellite extraction for LHC | Adds 9-OM time series, especially useful for classifiers 3 & 4 |
| **T10** | Convert `leaf_off_start_om` → `leaf_off_start_month` in feature table | Direct training labels for Classifiers 3 & 4 |

After T1 + T8 + T10: **can train first models for Classifiers 3 and 4.**

### Phase 2: Enable the remaining classifiers

| Task | What | Why |
|------|------|-----|
| **T4** | Add `is_acacia` label from species table into feature table | Unlocks Classifier 6 training |
| **T5** | "Possibly visible" flower review — open drone crops in viewer, decide +ve/-ve/discard | Locks in Classifier 1 & 2 positive set once OMs exist |
| **T6** | Check single-run area flight dates vs phenological calendar | Determines which classifiers those areas can contribute to |
| **T11** | Add `flower_colour` and `is_acacia` to feature table | Unlocks Classifiers 1, 2, 6 |
| **T12** | Evaluate Sentinel-1 features specifically for Acacia discrimination | VH/VV ratio may be the key discriminating feature |
| **T13** | Derive `leaf_on_start_om` from phenophase sequences | Finer-grained label for Classifier 4 |

### Phase 3: Future data collection

| Task | What |
|------|------|
| Plan April–June drone flights over SIT + LHC | Required to even attempt Classifiers 1 & 2 |
| SV (Sanjay Van) tracking + phenology pipeline | Adds Acacia training samples outside IITD context |
| T14 | Run SV through full tracking and phenology pipeline |

---

## 5. Key Architecture Decisions Made

### Why satellite is the feature source (not drone)
Drone orthomosaics are high-resolution but only cover a few hectares. A trained model that takes drone imagery as input would only work within those few hectares. Satellite data is global — a model trained on satellite features can predict for any tree anywhere with Sentinel coverage.

### Why both Sentinel-2 and Sentinel-1
- **Sentinel-2 (optical):** NDVI and EVI track vegetation greenness → excellent for deciduousness, leaf timing
- **Sentinel-1 (SAR):** Radar backscatter tracks canopy structure → fine feathery canopy (Acacia) vs broad-leaved canopy have distinct VH/VV signatures; SAR penetrates cloud cover, providing observations even in monsoon months where Sentinel-2 is often cloud-masked

### Professor's guidance: careful positive/negative set construction
For each classifier, the training set must be built from deliberate choices:
- Which species go in the positive set (e.g., only prominent-flower trees for Classifier 1)
- Which species go in the negative set (e.g., clear non-Acacia for Classifier 6)
- Which species are discarded from training (ambiguous or too rare)
- This is not a one-time decision — it becomes a grid search axis in experiments

### The semi-evergreen problem
The binary `is_deciduous` flag doesn't capture semi-evergreen behaviour. Some species (Pongamia, Prosopis) partially shed leaves. For Classifier 5, we need 3-class labels, and the class boundaries need careful annotation at the species level rather than just thresholding `deciduous_score`.

---

## 6. Data Inventory Summary

### Sites with full temporal coverage (multiple OMs)

| Site | OMs | Known Dates | Crown Count | Pipeline Status |
|------|-----|-------------|-------------|----------------|
| SIT | 14 | Need flight log lookup | 126 filtered | ✅ Full drone pipeline done; satellite 20/126 sampled |
| LHC | 9 | Oct 25, Nov 9, Nov 20, Nov 26, Dec 9, Jan 11, Feb 4, Feb 20, Mar 7 | TBD | ✅ Full drone pipeline done; satellite not yet run |
| SV (Sanjay Van) | TBD | TBD | TBD | ⚠️ Not yet integrated |

### Single-run areas (spatial only, no time series)

| Area | Flight Date Known? | Can Contribute To |
|------|-------------------|-------------------|
| area1 | TBD | Species appearance, Acacia (if labeled), Evergreen/Deciduous appearance |
| area2 | TBD | Same as area1 |
| area3 | TBD | Same |
| area6 | TBD | Same |

### Validated annotation set
`ground_truth.json` — 124 COCO-format manually annotated crowns, used for Detectree2 IoU validation.

---

## 7. Document Map (What Else Exists)

| File | Purpose |
|------|---------|
| `misc/CLASSIFIER_DATA_PLAN.md` | Full classifier plans with per-classifier feature/label specs, data flow diagram, experiment grid design |
| `misc/MASTER_GEOJSON_SCHEMA.md` | Schema for the canonical per-tree GeoJSON with all labels |
| `misc/PHENOLOGY_METRICS_FORMAL.md` | Mathematical formalization of phenology metrics |
| `output/sat_data/sit_scl_masked_6May26/` | Sentinel-2 feature extraction output: raw observations, crown-level features, run metadata |
| `output/sat_data/sit_sentinel1_6May26/` | Sentinel-1 SAR features, including fused S1+S2 table |
| `src/flask_app_tracking/tree_tracking.py` | Crown tracking and consensus generation |
| `src/flask_app_tracking/phenology_leafshed.py` | Leafshed/phenophase scoring |
| `src/notebooks/crown_tracking_31Mar26.ipynb` | Main orchestration notebook: exports consensus GeoJSON, viewer, crop manifests |
| `src/notebooks/Phenology_signals_25Mar26.ipynb` | Phenology feature extraction + normalization |
| `src/notebooks/sat_data6May26.ipynb` | Sentinel-2 satellite feature pipeline |
| `src/notebooks/sat_data_sentinel1.ipynb` | Sentinel-1 SAR pipeline |
