# Acacia Label Package Analysis

## Source Data

The label packages combine four label sources:

1. Species labels from field-identified crowns.
   These are mapped to Acacia/non-Acacia using
   `acacia_vs_non_acacia.json`.
2. Field/QField `Tree type` labels from Sanjay Van crown mapping.
3. Desk visual labels from crown images.
4. Clustering-derived labels for Sanjay Van crowns.

The maintained species trait table is `species_labels.csv` in the config set.

When labels disagree, final labels use this order:

`species > field > desk > clustering`

The clean package stops before clustering:

`species > field > desk`

## Current Area Coverage

Both GeoJSONs contain these areas:

| Area | Crowns |
|---|---:|
| A1 | 158 |
| A2 | 153 |
| A3 | 187 |
| A4 | 167 |
| A5 | 201 |
| LHC | 49 |
| MITTAL | 36 |
| SIT | 131 |
| SV_S1 | 656 |
| SV_S2 | 717 |
| SV_S3 | 628 |
| SV_S4 | 178 |
| **Total** | **3,261** |

## Clean Confident Package

File: `acacia_clean_confident_labels.geojson`

Final label: `label_acacia_clean_confident`

| Class | Count |
|---|---:|
| Acacia (`1`) | 336 |
| non-Acacia (`0`) | 576 |
| unlabelled / ignore (`-1`) | 2,349 |
| **Total crowns** | **3,261** |

Final label sources:

| Source | Count |
|---|---:|
| species | 391 |
| field | 153 |
| desk | 368 |
| none | 2,349 |

Area-by-area class counts:

| Area | Ignore | non-Acacia | Acacia |
|---|---:|---:|---:|
| A1 | 124 | 33 | 1 |
| A2 | 123 | 30 | 0 |
| A3 | 132 | 55 | 0 |
| A4 | 126 | 41 | 0 |
| A5 | 193 | 8 | 0 |
| LHC | 1 | 48 | 0 |
| MITTAL | 25 | 11 | 0 |
| SIT | 56 | 75 | 0 |
| SV_S1 | 496 | 95 | 65 |
| SV_S2 | 564 | 82 | 71 |
| SV_S3 | 390 | 78 | 160 |
| SV_S4 | 119 | 20 | 39 |

Use this package when the experiment should use only high-confidence labels.
It is the better choice for a defensible baseline and for reporting clean
ground-truth performance.

## Clean Plus Clustering Package

File: `acacia_clean_plus_clustering_labels.geojson`

Final label: `label_acacia_clean_plus_clustering`

| Class | Count |
|---|---:|
| Acacia (`1`) | 951 |
| non-Acacia (`0`) | 1,530 |
| unlabelled / ignore (`-1`) | 780 |
| **Total crowns** | **3,261** |

Final label sources:

| Source | Count |
|---|---:|
| species | 391 |
| field | 153 |
| desk | 368 |
| clustering | 1,569 |
| none | 780 |

Area-by-area class counts:

| Area | Ignore | non-Acacia | Acacia |
|---|---:|---:|---:|
| A1 | 124 | 33 | 1 |
| A2 | 123 | 30 | 0 |
| A3 | 132 | 55 | 0 |
| A4 | 126 | 41 | 0 |
| A5 | 193 | 8 | 0 |
| LHC | 1 | 48 | 0 |
| MITTAL | 25 | 11 | 0 |
| SIT | 56 | 75 | 0 |
| SV_S1 | 0 | 474 | 182 |
| SV_S2 | 0 | 426 | 291 |
| SV_S3 | 0 | 261 | 367 |
| SV_S4 | 0 | 68 | 110 |

Use this package when the goal is to test whether the larger clustering-derived
label set helps model training. It gives much more Sanjay Van coverage, but
clustering labels are weaker than species, field, and desk labels. Results from
this package should be presented as augmented or proxy-label experiments, not as
pure ground truth.
