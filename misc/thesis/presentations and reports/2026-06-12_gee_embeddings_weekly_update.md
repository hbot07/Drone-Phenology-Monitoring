# Weekly Update: GEE Satellite Embeddings for Acacia Classification

Date: 12 June 2026  
Project: Drone Phenology Monitoring  
Focus: Acacia vs non-Acacia classification using Google Earth Engine Satellite Embedding V1, original crown geometries only

## Slide 1: Goal

Build practical Acacia/non-Acacia classifiers using satellite embeddings instead of raw Sentinel bands.

Current constraint:

- Use only GEE Satellite Embedding V1 annual embeddings.
- Use original crown geometries, not 20 m buffers.
- Prioritize Sanjay Van visually annotated labels because crowns are small and buffer context is not appropriate.

## Slide 2: Data Used

Input embedding data:

- `iitd_sv_gee_embeddings_2024_original.csv`
- One 64-dimensional vector per crown: `A00` to `A63`
- Features extracted in Earth Engine using median reducer over original crown geometry
- No 20 m buffer used in current experiments

New visual label data:

- `labeling_sheet.csv`
- 400 visually annotated Sanjay Van crowns
- Label distribution:
  - Acacia: 193
  - Non-Acacia: 207

Mapped sites:

| Site | Non-Acacia | Acacia | Total |
|---|---:|---:|---:|
| SV_S1 | 86 | 47 | 133 |
| SV_S2 | 76 | 62 | 138 |
| SV_S3 | 29 | 64 | 93 |
| SV_S4 | 16 | 20 | 36 |

## Slide 3: Important Data Correction

The image names in the visual label sheet are zero-based, while GeoJSON crown IDs are one-based.

Example:

- `s1_tree_006.tif` maps to `SV_S1:crown_0007`

After applying this offset:

- All 400 visual labels mapped correctly to the crown GeoJSON.
- All 400 visual labels mapped correctly to the GEE embedding CSV.
- Earlier low agreement numbers were caused by the off-by-one mapping issue.

## Slide 4: Label Quality Audit

Agreement with species-based Acacia labels:

| Comparison | Comparable Crowns | Agreement |
|---|---:|---:|
| Visual vs species | 24 | 83.3% |
| Clustering vs species | 108 | 80.6% |
| Clustering vs species, same 24 visual-overlap crowns | 24 | 66.7% |
| Visual vs clustering | 400 | 86.0% |

Interpretation:

- Visual labels appear more reliable than clustering labels on the overlap where species labels are available.
- However, only 24 visually labelled crowns also have species-based ground truth, so this is promising but not yet a large validation set.
- Clustering labels provide many more rows, but they introduce label noise.

## Slide 5: Why This Is Hard

This is not only a "too little data" problem.

Main bottlenecks:

- Only 400 clean visual labels are available.
- Crowns are small relative to 10 m satellite pixels.
- GEE embeddings are annual 10 m representations, so many crown polygons are smaller than one pixel or mixed with nearby canopy/context.
- Site-to-site shift is strong: models that perform well in random split often drop in leave-site-out evaluation.
- Clustering labels add volume but are not consistent enough to improve generalization reliably.

Small crown diagnostic:

- Visual vs clustering agreement in smallest crown-area quartile: 79%
- Visual vs clustering agreement in larger quartiles: about 87-89%

This suggests some of the label ambiguity is concentrated among smaller crowns.

## Slide 6: Classifier Configurations Tested

Label configurations:

- Species-only labels
- Clustering-only labels
- Visual-only labels
- Visual or species labels
- Visual or clustering labels
- Species or clustering labels
- Priority labels: visual first, then species, then clustering

Model families:

- Logistic regression with class balancing
- Random forest
- Extra trees
- RBF SVM
- Histogram gradient boosting
- Threshold-tuned variants

Evaluation:

- Random stratified split
- Leave-site-out split for SV_S1, SV_S2, SV_S3, SV_S4
- Confusion matrices and per-site error summaries

## Slide 7: Main Results

Balanced accuracy summary:

| Label Config | Random Split | Leave-Site-Out Range | Interpretation |
|---|---:|---:|---|
| Visual-only | 0.776 | 0.567-0.812 | Cleanest trustworthy setup |
| Visual or species | 0.837 | 0.572-0.700 | Strong random split, less stable by site |
| Clustering-only | 0.696 | 0.597-0.636 | More data, noisier labels |
| Visual or clustering | 0.687 | 0.585-0.635 | Did not improve over visual-only |
| Species or clustering | 0.725 | 0.579-0.656 | Slightly better random split, still site-limited |
| All priority | 0.721 | 0.578-0.631 | More rows but no better generalization |
| Species-only | 0.934 | 0.658-0.681 | High random score, but small and site-skewed label set |

Best current practical visual-only model:

- Model: balanced random forest
- Features: `A00-A63`
- Training labels: 400 visual Acacia/non-Acacia labels
- Random split balanced accuracy: 0.776
- Random split macro F1: 0.775

## Slide 8: Visual-Only Leave-Site-Out Confusion Matrices

Rows are true labels, columns are predicted labels. Label 0 is non-Acacia, label 1 is Acacia.

| Holdout Site | Best Model | Balanced Acc | Confusion Matrix |
|---|---|---:|---|
| SV_S1 | Logistic regression, threshold tuned | 0.615 | `[[71, 15], [28, 19]]` |
| SV_S2 | Logistic regression, threshold tuned | 0.677 | `[[49, 27], [18, 44]]` |
| SV_S3 | RBF SVM, threshold tuned | 0.567 | `[[17, 12], [29, 35]]` |
| SV_S4 | Logistic regression | 0.812 | `[[10, 6], [0, 20]]` |

Key observation:

- SV_S1 Acacia recall is weak: 19/47 Acacia crowns correctly detected.
- SV_S4 performs well, but it has only 36 held-out visual labels.

## Slide 9: Targeted Improvement Experiments

Tested:

- Dropping smallest 25% of visual crowns by polygon area
- Dropping smallest 50% of visual crowns by polygon area
- Using only visual labels that agree with clustering labels
- Adding crown polygon area as an extra feature
- Visual-or-species combined label training

Best targeted results:

| Experiment | Split | Site | Balanced Acc | Notes |
|---|---|---|---:|---|
| Visual + species, embeddings + crown area | Random | NA | 0.844 | Best random score, but still not stable by site |
| Visual labels agreeing with clustering | Leave-site-out | SV_S4 | 0.825 | Small test set, 32 crowns |
| Visual-only baseline | Leave-site-out | SV_S4 | 0.812 | Similar to agreement-filtered setup |
| Visual-only baseline | Leave-site-out | SV_S2 | 0.668 | Stable moderate performance |
| Visual-only baseline | Leave-site-out | SV_S1 | 0.664 in targeted run | Improved over earlier suite, but Acacia recall still limited |
| Visual-only baseline | Leave-site-out | SV_S3 | 0.551 | Hardest site |

Conclusion:

- Filtering small crowns did not consistently improve results.
- Adding crown area helped random split slightly, but not consistently in leave-site-out.
- Using only visual labels that agree with clustering can help on some sites, but it reduces training data and does not solve generalization.

## Slide 10: Diagnosis

What is limiting performance?

1. Label quantity
   - 400 visual labels is useful but still small for site-general classifiers.

2. Spatial resolution
   - Many crowns are much smaller than a 10 m satellite pixel.
   - Embeddings may represent local canopy/context rather than the individual crown.

3. Site shift
   - Random splits are consistently optimistic.
   - Leave-site-out results show that models do not generalize equally across SV sub-sites.

4. Label noise in clustering labels
   - Clustering gives thousands of labels, but adding them often reduces site-level performance.

## Slide 11: Current Recommendation

Use visual-only GEE original embeddings as the main trustworthy baseline.

Recommended model artifact:

- `src/notebooks/satellite/embeddings/models/gee_original_acacia_label_configs/label_acacia_visual_rf_balanced.joblib`

Use clustering-heavy models only as exploratory or weak-label baselines, not as the main result.

For reporting to professor:

- Emphasize that the project moved from raw satellite bands to learned satellite embeddings.
- The embeddings contain useful signal, but current performance is limited by resolution, label noise, and site shift.
- The new visual labels are valuable and appear cleaner than clustering labels.

## Slide 12: Next Steps

Immediate next steps:

- Add more visually labelled crowns, especially species-overlap crowns for independent validation.
- Balance labels across SV_S1, SV_S2, SV_S3, and SV_S4.
- Prioritize ambiguous and small crowns for additional annotation.
- Consider higher-resolution imagery sources for crown-scale classification.
- Treat GEE embeddings as a site-level or context-aware baseline, not final crown-level evidence.

Potential technical next steps:

- Train models with site-aware cross-validation only.
- Use visual labels as clean labels and clustering labels only as weak supervision.
- Build confidence-ranked predictions for manual review.
- Compare with finer-resolution imagery if available.

## Files Created

Prepared datasets:

- `src/notebooks/satellite/embeddings/exports/normalized_iitd_sv_gee_embeddings_2024_original_visual_acacia.csv`
- `src/notebooks/satellite/embeddings/exports/gee_original_acacia_label_configs.csv`
- `src/notebooks/satellite/drone_phenology_rf_package/data/sv_crowns_visual_acacia_labeled.geojson`

Experiment outputs:

- `src/notebooks/satellite/embeddings/outputs/gee_original_acacia_label_configs_full/`
- `src/notebooks/satellite/embeddings/outputs/gee_original_acacia_visual_quality/`

Model artifacts:

- `src/notebooks/satellite/embeddings/models/gee_original_acacia_label_configs/`

Scripts:

- `src/notebooks/satellite/embeddings/prepare_visual_acacia_labels.py`
- `src/notebooks/satellite/embeddings/prepare_acacia_label_configs.py`
- `src/notebooks/satellite/embeddings/analyze_acacia_config_errors.py`
- `src/notebooks/satellite/embeddings/visual_label_quality_experiments.py`

## Slide 13: Next Practical Output - Annotation Review Queue

After training the visual-only classifier, all 3,212 crowns were scored with Acacia probability.

Purpose:

- Use the classifier as an annotation assistant, not as final ground truth.
- Identify high-confidence unlabeled crowns for rapid verification.
- Identify uncertain crowns where new manual labels are most valuable.
- Identify places where model predictions disagree with clustering labels.

Output files:

- `src/notebooks/satellite/embeddings/outputs/gee_original_acacia_review_candidates/all_crowns_scored.csv`
- `src/notebooks/satellite/embeddings/outputs/gee_original_acacia_review_candidates/high_confidence_unlabeled.csv`
- `src/notebooks/satellite/embeddings/outputs/gee_original_acacia_review_candidates/uncertain_unlabeled.csv`
- `src/notebooks/satellite/embeddings/outputs/gee_original_acacia_review_candidates/model_cluster_disagreements.csv`

## Slide 14: Review Queue Results

Summary from the visual-only model:

| Quantity | Count |
|---|---:|
| Total crowns scored | 3,212 |
| Already visually labelled | 400 |
| Unlabelled by visual annotation | 2,812 |
| Unlabelled predicted Acacia | 915 |
| Unlabelled predicted non-Acacia | 1,897 |
| Unlabelled uncertain, probability 0.4-0.6 | 1,347 |
| Model-vs-clustering disagreements | 683 |
| High-confidence unlabelled, confidence >= 0.80 | 407 |
| High-confidence unlabelled, confidence >= 0.90 | 91 |

Interpretation:

- The model produces a useful triage list even though it is not final.
- The 683 model-vs-clustering disagreements are good candidates for manual review because they may reveal clustering label errors or model failure modes.
- The 1,347 uncertain crowns are ideal targets for additional annotation because they sit near the decision boundary.

## Slide 15: Predicted Acacia Distribution in Unlabelled Crowns

Unlabelled predictions by site:

| Site | Predicted non-Acacia | Predicted Acacia |
|---|---:|---:|
| A1 | 158 | 0 |
| A2 | 147 | 6 |
| A3 | 184 | 3 |
| A4 | 166 | 1 |
| A5 | 201 | 0 |
| MITTAL | 34 | 2 |
| SIT | 119 | 12 |
| SV_S1 | 359 | 164 |
| SV_S2 | 341 | 238 |
| SV_S3 | 120 | 415 |
| SV_S4 | 68 | 74 |

Observation:

- The classifier predicts most Acacia candidates inside the Sanjay Van sub-sites, especially SV_S3.
- This may reflect real site composition, but it may also reflect site/context learning.
- These predictions should be used to prioritize annotation, not as final ecological counts.

## Slide 16: Finer-Than-10 m Satellite Data Options

Why needed:

- Many crowns are smaller than a 10 m pixel.
- GEE embeddings are powerful but may encode surrounding canopy/context instead of the individual crown.
- Finer imagery should better align with crown polygons.

Candidate data sources:

| Source | Approx. resolution | Access | Notes |
|---|---:|---|---|
| Planet NICFI basemaps | about 4.77 m | Available in Earth Engine for eligible tropical regions/non-commercial use | Most practical first test because workflow is similar to GEE embeddings |
| PlanetScope | about 3-5 m class | Commercial/research access | Daily-to-near-daily optical imagery; useful if we can get access |
| Planet SkySat | sub-meter class | Commercial/research access | Much closer to crown scale; may be expensive |
| Maxar WorldView / Legion | about 30 cm class | Commercial | Very high resolution; best crown-scale satellite option but likely costly |
| Airbus Pleiades Neo | about 30 cm class | Commercial | Similar very-high-resolution option |
| Indian high-resolution optical products | sub-meter to few-meter class depending product | Check NRSC/Bhuvan availability | Potentially relevant because sites are in Delhi, but access/licensing needs checking |

Recommended order:

1. Try Planet NICFI first because it can be processed in Earth Engine and is finer than 10 m.
2. If NICFI coverage/access is insufficient, look for institutional access to PlanetScope or SkySat.
3. For true crown-scale validation, use drone imagery or commercial sub-meter satellite imagery.

Sources:

- Google Earth Engine Planet NICFI dataset: `https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_asia`
- Planet SkySat documentation: `https://docs.planet.com/data/imagery/skysat/`
- PlanetScope documentation: `https://docs.planet.com/data/imagery/planetscope/`
- Maxar/Vantor constellation information: `https://vantor.com/company/constellation/`
- Airbus Pleiades Neo information: `https://www.intelligence-airbusds.com/imagery/constellation/pleiades-neo/`

## Slide 17: New Finer-Resolution Extraction Path

Added script:

- `scripts/extract_gee_nicfi_features.py`

What it does:

- Uses Planet NICFI basemaps from Earth Engine.
- Extracts median and standard deviation band features over original crown polygons.
- Exports one row per crown to Google Drive.
- Keeps the same crown-level classifier workflow as the GEE embedding experiments.

Example command:

```bash
python3 scripts/extract_gee_nicfi_features.py \
  --project adept-vigil-418410 \
  --crowns-asset projects/adept-vigil-418410/assets/iitd_sv_crowns_master_shapefile_for_gee \
  --region asia \
  --start-date 2024-01-01 \
  --end-date 2025-01-01 \
  --geometry-mode polygon
```

Expected next result:

- Compare NICFI crown features against GEE embeddings on the same visual labels.
- If NICFI improves leave-site-out performance, it suggests spatial resolution is a major limiting factor.
- If NICFI does not improve much, then label quality/site shift may be the dominant limitation.
