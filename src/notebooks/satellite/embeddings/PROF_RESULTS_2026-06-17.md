# Sanjay Van / Campus Crown Classifiers — Results Summary

Date: 17 June 2026

Labels come from the drone-phenology ground truth: field species calls + a
species→phenology mapping sheet, plus desk visual review of crown images for
Acacia. Confusion matrices are shown because the labelled set is small (tens of
positives per class) and a single accuracy number hides where the errors land.

## How to read this

Confusion matrix layout (binary tasks), rows = truth, columns = prediction:

```
              pred 0 (non)   pred 1 (target)
true 0 (non)      TN              FP
true 1 (target)   FN              TP
```

- **Balanced accuracy (bacc)** = average of per-class recall — fair under class imbalance (chance = 0.50).
- **macro-F1** = unweighted mean F1 across classes.
- **random split**: stratified 70/30. Optimistic — the same sites appear in train and test.
- **leave-area-out (LAO)**: an entire site (e.g. a Sanjay Van section) is held out and the model is trained on the others. This is the honest test of whether the model generalizes to a *new location*. **LAO is the number to trust.**

## Models in the sweep

Every task is run through 7 pipelines (all with median imputation; linear/kernel models are standardized; tree models are not):

| Model | What it is | When it tends to win |
|---|---|---|
| `rf_balanced` | Random Forest, 300 trees, `balanced_subsample` class weights, leaf≥2, √-features | robust default on small tabular data |
| `rf_deeper` | Random Forest, deeper (leaf=1, 50% features/split) | when more capacity helps |
| `extra_trees` | Extremely Randomized Trees (random split thresholds) | lower variance than RF on noisy features |
| `extra_trees_kbest` | SelectKBest (top-40 ANOVA-F features) → Extra Trees | high-dimensional inputs (DINO/temporal) |
| `logistic_l2` | L2-regularized logistic regression, class-balanced | linearly separable embeddings; strong baseline |
| `svc_rbf` | RBF-kernel SVM, class-balanced | non-linear boundaries on standardized features |
| `hist_gradient` | Histogram gradient boosting | structured non-linear interactions |

`threshold_cv` next to a model means the decision threshold on P(class=1) was
tuned by 5-fold cross-validation on the training set to maximise balanced
accuracy, instead of the default 0.5 (matters for imbalanced classes).

## Feature sources compared

| Source | Dims | Description |
|---|---|---|
| GEE weighted-mean | 64 | Google Satellite-Embedding-V1 2024 annual, area-weighted mean over the original crown polygon |
| GEE centroid | 64 | same embedding sampled at the crown centroid pixel |
| DINOv2 patch | 384 / 1152 | local DINOv2 features on Sentinel-2 RGB patches centred on the crown (Mar / Mar+Apr+May 2025) |
| **S2 temporal** | 724 | **multi-year (2022–2025) Sentinel-2 band + index seasonal time-series statistics** |
| Fusions | up to 1172 | concatenations of the above by `crown_uid` |

The 20 m buffer source was dropped on the prof's instruction (too coarse for
sub-pixel crowns). Weighted-mean ≈ centroid ≈ median (mean |Δ|≈0.002 over 64
dims) because these crowns are essentially one 10 m pixel.

---

# Best classifier per task (with confusion matrices)

## 1. Acacia vs non-Acacia — STRONG, the headline result
Label = field/species ground truth (`label_acacia`, 65 acacia / 326 non).

**Best model: GEE + DINOv2 + S2-temporal fusion (1172 features), Random Forest.**

```
random split           bacc 0.973   macro-F1 0.931   n_test=113
              pred non   pred acacia
true non         88          5
true acacia       0         20         -> all 20 acacia found, 5 FP of 93

leave-area-out (hold out Sanjay Van S4)   bacc 0.863  n_test=28
              pred non   pred acacia
true non          7          1
true acacia       3         17         -> 17/20 acacia correct in an unseen site
```

Single-modality alternatives (so the prof sees it isn't just the kitchen sink):
- **S2-temporal alone (724d), Extra-Trees:** random 0.934 `[[90,8],[1,19]]`; **LAO S4 0.850** `[[6,2],[1,19]]`.
- **DINOv2-Mar (384d), RF:** random 0.972 `[[85,5],[0,19]]`; LAO S3 0.849 `[[7,1],[3,14]]`.
- GEE annual embedding alone (64d): random 0.93 but **LAO only ~0.67** `[[5,4],[2,7]]`.

**Analysis.** Acacia is genuinely separable and — crucially — **transfers to a new
site**. The annual GEE embedding alone is strong on random split but collapses to
~0.67 leave-site-out (it partly memorises site context). Adding **multi-year
temporal Sentinel-2** lifts leave-site-out from ~0.67 to **0.85–0.86**, because
Acacia (esp. *Prosopis*) has a distinct evergreen + phenological signature that a
year-long time series captures and a single annual vector does not. This is a
publishable, defensible classifier.

## 2. Yellow-flowering (broad) — MODERATE/GOOD
`label_yellow_broad` (100 yellow / 255 not) — Amaltas, Kasod, etc.

**Best: GEE + DINOv2 + S2-temporal fusion, RBF-SVM (threshold-tuned).**

```
random split            bacc 0.848   macro-F1 0.841   n_test=107
              pred 0   pred 1
true 0          69       8
true 1           6      24

leave-area-out (hold out SV_S1)    bacc 0.731   n_test=31
              pred 0   pred 1
true 0           7       3
true 1           5      16
```

**Analysis.** Yellow flowering is a strongly temporal trait (bloom timing), so the
temporal features help (random 0.795 → 0.848). Leave-site-out is a respectable
0.73 but more variable across sites — usable as a screening flag, not a final call.

## 3. Deciduous — WEAK/MODEST
`label_deciduous` (115 deciduous / 227 not).

**Best: S2-temporal (724d), logistic regression.**

```
random split            bacc 0.705   n_test=103
              pred 0   pred 1
true 0          57      11
true 1          15      20

leave-area-out (hold out SIT)    bacc 0.673   n_test=75
              pred 0   pred 1
true 0          26      16
true 1           9      24
```

**Analysis.** Temporal features are the right tool (leaf-on/leaf-off is timing) and
do beat the annual embedding, but ~0.70 random / 0.67 LAO is only modestly above
chance. Bottlenecked by label count, not feature choice.

## 4. Showy-flowering — WEAK / UNSTABLE
`label_showy_flower` (63 showy / 292 not). Best LAO (SIT) 0.734 `[[59,5],[5,6]]`
but random only ~0.66, and only 6 positives in the held-out test — treat as
exploratory, not reliable.

## 5. Not currently usable (underpowered)
- **`label_yellow_strict`** (40 pos): high apparent bacc but matrices show it mostly predicts the majority class; LAO test sets have ~2–8 positives. Not reliable.
- **`label_esd`** (3-class evergreen/semi/deciduous): ~0.51–0.59, near chance.
- **`label_red_showy`** (12 positives total): not learnable; "high" LAO scores are on n=8 test crowns (noise).

---

# Synthetic clustering-label experiment (the auto-generated Acacia labels)

We made a large clustering-derived ("synthetic") Acacia label set (2,179 crowns)
to try to scale beyond the hand-labelled crowns. Two tests, on GEE centroid features:

- **Synthetic labels predicting ground truth:** train on 1,779 clustering-only
  crowns, test on the 400 clean visual labels → bacc **0.689** `[[134,73],[52,141]]`.
  Real signal, but ~14% label noise propagates.
- **Do synthetic labels improve the clean classifier?** No. Visual-only random
  holdout 0.770 → visual+synthetic 0.744; leave-site-out S4 0.812 → 0.706. The
  synthetic labels *dilute* the clean ones.

**Takeaway:** the synthetic clustering labels are useful for coarse pre-screening
but are too noisy to add to training. The lever for the weak tasks is **more
hand-labelled positives**, not more features or synthetic data.

---

# One-paragraph summary for the prof

Acacia detection works and generalises across sites (0.97 random, **0.86
leave-a-whole-Sanjay-Van-section-out**); the key enabler is multi-year temporal
Sentinel-2 features, which fix the annual embedding's tendency to memorise site
context (LAO 0.67→0.86). Yellow-flowering is a moderate secondary result (0.85 /
0.73). Deciduous is weak-but-real with temporal features. ESD, yellow-strict,
showy and red-flower remain underpowered — limited by the number of labelled
crowns, not by the satellite features. The synthetic clustering Acacia labels
carry signal (~0.69 vs ground truth) but are too noisy to improve the clean model.

> Caveat: the 22 desk-label corrections from the latest crown-image review are
> applied to `labeling_sheet.csv` but not yet re-propagated into the modelling
> inputs; they affect only the `label_acacia_visual` track. All numbers above use
> `label_acacia` (field/species ground truth), which is unaffected.
