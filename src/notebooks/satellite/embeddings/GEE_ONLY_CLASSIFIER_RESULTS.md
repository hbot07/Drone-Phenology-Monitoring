# GEE-Only Classifier Results

Date: 17 June 2026

## Scope

This benchmark uses Google Earth Engine Satellite Embedding V1 features only. Planet/NICFI is not part of the active path.

Compared GEE feature sources:

- median over original crown polygons
- centroid pixel
- mean over original crown polygons
- median over 20 m centroid buffer
- fused original + buffer20 embeddings

Primary outputs:

- `outputs/_gee_only_all_results.csv`
- `outputs/_gee_only_best_random_by_label.csv`
- `outputs/_gee_only_best_lao_by_label.csv`
- `outputs/_gee_only_recommended_artifacts.csv`

## Best Random-Split Classifiers

These are the best saved full-data model artifacts by label. Random split is useful for model selection, but it is optimistic; check leave-area-out before making strong claims.

| Label | Best GEE source | Model | Decision | Balanced accuracy | Macro-F1 |
|---|---|---|---|---:|---:|
| `label_acacia` | buffer20 median | RF balanced | threshold 0.10 | 0.939 | 0.852 |
| `label_acacia_species` | original median | Extra Trees | threshold 0.20 | 0.934 | 0.880 |
| `label_acacia_visual` | original+buffer20 fused | SVC RBF | default | 0.786 | 0.782 |
| `label_acacia_visual_or_species` | original+buffer20 fused | Extra Trees K-best | threshold 0.30 | 0.849 | 0.808 |
| `label_yellow_broad` | original+buffer20 fused | Extra Trees K-best | threshold 0.35 | 0.825 | 0.818 |
| `label_yellow_strict` | centroid pixel | Logistic regression | default | 0.801 | 0.657 |
| `label_deciduous` | centroid pixel | HistGradientBoosting | threshold 0.15 | 0.688 | 0.667 |
| `label_showy_flower` | original median | RF deeper | threshold 0.15 | 0.699 | 0.566 |
| `label_esd` | buffer20 median | RF deeper | default | 0.529 | 0.526 |
| `label_red_showy` | centroid pixel | Logistic regression | threshold 0.30 | 0.547 | 0.498 |

## Best Leave-Area-Out Signals

Leave-area-out is the better proxy for transfer to new places. These are the strongest GEE-only holdout results seen so far.

| Label | Best GEE source | Holdout | Model | Balanced accuracy | Macro-F1 |
|---|---|---|---|---:|---:|
| `label_acacia_species` | original+buffer20 fused | `SV_S3` | Extra Trees | 0.846 | 0.857 |
| `label_acacia_visual` | original median or centroid | `SV_S4` | Logistic regression | 0.812 | 0.819 |
| `label_acacia` | buffer20 median | `SV_S3` | Extra Trees | 0.783 | 0.802 |
| `label_acacia_visual_or_species` | centroid pixel | `SV_S4` | Logistic regression | 0.725 | 0.749 |
| `label_yellow_broad` | original+buffer20 fused | `SV_S1` | RF deeper | 0.733 | 0.698 |
| `label_yellow_strict` | original or buffer20 median | `A4` | HistGradientBoosting | 0.726 | 0.681 |
| `label_showy_flower` | buffer20 median | `MITTAL` | SVC RBF | 0.833 | 0.817 |
| `label_deciduous` | buffer20 median | `A3` | Logistic regression | 0.654 | 0.599 |

## What Works

Good enough to keep moving forward:

- `label_acacia` / `label_acacia_species`: strongest and most repeatable GEE-only classifiers.
- `label_acacia_visual`: cleanest label source; fewer rows but better trust.
- `label_acacia_visual_or_species`: good random split, moderate transfer.
- `label_yellow_broad`: strong random split and some useful holdouts.
- `label_yellow_strict`: promising random split, but transfer is area-dependent.

Use cautiously:

- `label_deciduous`: moderate performance; useful as a baseline, not a final result.
- `label_showy_flower`: some good holdouts, but random macro-F1 is weaker.

Do not claim as working yet:

- `label_esd`: near chance.
- `label_red_showy`: underpowered and near chance because positives are too rare.
- clustering-heavy Acacia labels: useful for triage/review, but they do not improve transfer.

## Recommended Artifacts

The current artifact manifest is:

- `outputs/_gee_only_recommended_artifacts.csv`

Each artifact now stores:

- trained sklearn model/pipeline
- label name
- feature columns
- class list
- training counts
- best random balanced accuracy and macro-F1
- decision mode
- tuned decision threshold when applicable

## Next Experiments

Highest-yield next steps within GEE embeddings:

1. Focus manual review on Acacia disagreements and uncertain crowns, then retrain `label_acacia_visual`.
2. Keep original polygon and centroid pixel as primary geometry modes; use fused geometry only when leave-area-out improves.
3. For yellow broad/strict, inspect area-specific failures before adding more models.
4. Avoid spending time on `label_esd` and `label_red_showy` until labels improve.
