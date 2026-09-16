# `gee_original_acacia_label_configs.csv` — Dataset Documentation

> **Purpose:** Documentation of the GEE embedding dataset and its Acacia-label configurations.
>
> **Important:** `label_acacia_visual` is the **confident-label column** currently used for the confidently labelled Acacia/non-Acacia samples. The other `label_acacia_*` columns represent alternative/combined label configurations and should not be treated as independent raw datasets.

## 1. Dataset overview

- **Rows / samples:** 3,212
- **Columns:** 95
- **Embedding dimensions detected (`A00` … `A63`):** 64
- **Embedding range:** `A00` to `A63`
- **Acacia label/configuration columns:** 8
- **Species column:** `species_clean`
- **Raw species column:** `species_raw`
- **Species status column:** `species_status`

## 2. Column groups

| Group | Columns | Meaning |
|---|---|---|
| GEE embeddings | `A00`–`A63` | Numerical embedding dimensions; 64 dimensions per sample |
| Metadata / source fields | 23 columns | Source identifiers, geometry/metadata, species information, etc. |
| Acacia labels | 8 columns | Derived label configurations |

## 3. Label convention

The label columns use the following convention where applicable:

| Value | Interpretation |
|---:|---|
| `1` | Acacia |
| `0` | Non-Acacia |
| `-1` | Unknown / not confidently assigned |
| `NaN` | Missing value |

### Primary confident-label column

**`label_acacia_visual`** is the current **confident-label column**. Use this column when an experiment requires the confident visual Acacia/non-Acacia labels.

Do **not** assume that every `1` in another configuration is a manually/confidently labelled sample. The configuration must be interpreted according to the code/rules that generated that column.

## 4. Acacia label configurations

| Column | Role / interpretation |
|---|---|
| `label_acacia` | Base Acacia label configuration. |
| `label_acacia_visual` | Confident visual labels; current primary confident-label column. |
| `label_acacia_clustering` | Clustering-based Acacia configuration. |
| `label_acacia_species` | Species-based Acacia configuration. |
| `label_acacia_visual_or_species` | Combined visual OR species configuration. |
| `label_acacia_visual_or_clustering` | Combined visual OR clustering configuration. |
| `label_acacia_species_or_clustering` | Combined species OR clustering configuration. |
| `label_acacia_all_priority` | Combined priority configuration. |

> **Provenance note:** The exact species/rule membership of each configuration should be documented from the script/notebook that generated the CSV. Column names alone are not sufficient to reconstruct those rules.

## 5. Overall label statistics

| Configuration | Acacia (1) | Non-Acacia (0) | Unknown (-1) | NaN | Other | Total |
|---|---:|---:|---:|---:|---:|---:|
| `label_acacia` | 65 | 326 | 2,821 | 0 | 0 | 3,212 |
| `label_acacia_visual` | 203 | 197 | 2,812 | 0 | 0 | 3,212 |
| `label_acacia_clustering` | 901 | 1,278 | 1,033 | 0 | 0 | 3,212 |
| `label_acacia_species` | 65 | 326 | 2,821 | 0 | 0 | 3,212 |
| `label_acacia_visual_or_species` | 251 | 516 | 2,445 | 0 | 0 | 3,212 |
| `label_acacia_visual_or_clustering` | 923 | 1,256 | 1,033 | 0 | 0 | 3,212 |
| `label_acacia_species_or_clustering` | 913 | 1,549 | 750 | 0 | 0 | 3,212 |
| `label_acacia_all_priority` | 931 | 1,531 | 750 | 0 | 0 | 3,212 |

### `label_acacia_visual` summary

- **Confident labelled samples (`0` or `1`):** 400
- **Acacia (`1`):** 203
- **Non-Acacia (`0`):** 197
- **Unknown (`-1`):** 2,812
- **Missing (`NaN`):** 0

## 6. Species breakdown

| Rank | Species | Count |
|---:|---|---:|
| 1 | MISSING / UNKNOWN | 2,840 |
| 2 | Prosopis Juliflora | 53 |
| 3 | Neem | 51 |
| 4 | Ashok | 29 |
| 5 | Amaltas | 25 |
| 6 | Peepal | 24 |
| 7 | Pilkhan | 23 |
| 8 | Maulsari | 17 |
| 9 | Banyan | 16 |
| 10 | Kasod | 13 |
| 11 | Mandphali / Marodphali | 13 |
| 12 | Saptparni | 13 |
| 13 | Shahtoot | 9 |
| 14 | Siris | 7 |
| 15 | Maha neem / Mahneem | 7 |
| 16 | Sheesham / Shisham | 6 |
| 17 | Jamun | 6 |
| 18 | Buddha's Coconut | 5 |
| 19 | Gulmohar | 5 |
| 20 | Arjun | 5 |
| 21 | Bamboo clump | 4 |
| 22 | Subabool | 3 |
| 23 | Semal / Silk cotton | 3 |
| 24 | Frangipani | 3 |
| 25 | Imli | 2 |
| 26 | Teak | 2 |
| 27 | Caribbean trumpet | 2 |
| 28 | Karanj | 2 |
| 29 | Chamrod | 2 |
| 30 | Mango | 2 |
| 31 | Bottlebrush | 2 |
| 32 | Goolar | 2 |
| 33 | Bakain | 2 |
| 34 | Palm tree | 1 |
| 35 | Cassia | 1 |
| 36 | Acacia unknown | 1 |
| 37 | Banana | 1 |
| 38 | Anjan | 1 |
| 39 | Native acacia | 1 |
| 40 | Kanju | 1 |
| 41 | Chukrasia | 1 |
| 42 | Amla | 1 |
| 43 | Bel | 1 |
| 44 | Moringa | 1 |
| 45 | Kachnar | 1 |
| 46 | Sausage tree | 1 |
| 47 | Palash | 1 |

### Species × confident visual label

| Species | Total | Acacia (1) | Non-Acacia (0) | Unknown (-1) |
|---|---:|---:|---:|---:|
| MISSING / UNKNOWN | 2,840 | 190 | 189 | 2,461 |
| Prosopis Juliflora | 53 | 13 | 1 | 39 |
| Neem | 51 | 0 | 2 | 49 |
| Ashok | 29 | 0 | 0 | 29 |
| Amaltas | 25 | 0 | 2 | 23 |
| Peepal | 24 | 0 | 1 | 23 |
| Pilkhan | 23 | 0 | 0 | 23 |
| Maulsari | 17 | 0 | 0 | 17 |
| Banyan | 16 | 0 | 1 | 15 |
| Mandphali / Marodphali | 13 | 0 | 0 | 13 |
| Kasod | 13 | 0 | 0 | 13 |
| Saptparni | 13 | 0 | 0 | 13 |
| Shahtoot | 9 | 0 | 0 | 9 |
| Siris | 7 | 0 | 0 | 7 |
| Maha neem / Mahneem | 7 | 0 | 0 | 7 |
| Sheesham / Shisham | 6 | 0 | 0 | 6 |
| Jamun | 6 | 0 | 0 | 6 |
| Arjun | 5 | 0 | 0 | 5 |
| Buddha's Coconut | 5 | 0 | 0 | 5 |
| Gulmohar | 5 | 0 | 0 | 5 |
| Bamboo clump | 4 | 0 | 0 | 4 |
| Subabool | 3 | 0 | 1 | 2 |
| Frangipani | 3 | 0 | 0 | 3 |
| Semal / Silk cotton | 3 | 0 | 0 | 3 |
| Bakain | 2 | 0 | 0 | 2 |
| Teak | 2 | 0 | 0 | 2 |
| Mango | 2 | 0 | 0 | 2 |
| Imli | 2 | 0 | 0 | 2 |
| Goolar | 2 | 0 | 0 | 2 |
| Bottlebrush | 2 | 0 | 0 | 2 |
| Caribbean trumpet | 2 | 0 | 0 | 2 |
| Karanj | 2 | 0 | 0 | 2 |
| Chamrod | 2 | 0 | 0 | 2 |
| Cassia | 1 | 0 | 0 | 1 |
| Bel | 1 | 0 | 0 | 1 |
| Amla | 1 | 0 | 0 | 1 |
| Acacia unknown | 1 | 0 | 0 | 1 |
| Banana | 1 | 0 | 0 | 1 |
| Anjan | 1 | 0 | 0 | 1 |
| Kanju | 1 | 0 | 0 | 1 |
| Kachnar | 1 | 0 | 0 | 1 |
| Chukrasia | 1 | 0 | 0 | 1 |
| Palash | 1 | 0 | 0 | 1 |
| Palm tree | 1 | 0 | 0 | 1 |
| Moringa | 1 | 0 | 0 | 1 |
| Native acacia | 1 | 0 | 0 | 1 |
| Sausage tree | 1 | 0 | 0 | 1 |

## 7. Raw species → cleaned species mapping

| Raw species | Clean species | Count |
|---|---|---:|
| MISSING / UNKNOWN | MISSING / UNKNOWN | 2,794 |
| Prosopis Juliflora | Prosopis Juliflora | 53 |
| Neem | Neem | 51 |
| Ashok | Ashok | 29 |
| Amaltas | Amaltas | 25 |
| Peepal | Peepal | 24 |
| Pilkhan | Pilkhan | 23 |
| Maulsari | Maulsari | 17 |
| Banyan | Banyan | 16 |
| Kasod | Kasod | 13 |
| Saptparni | Saptparni | 13 |
| Others | MISSING / UNKNOWN | 13 |
| Mandphali | Mandphali / Marodphali | 11 |
| Shahtoot | Shahtoot | 9 |
| Siris | Siris | 7 |
| Semal fig | MISSING / UNKNOWN | 6 |
| Jamun | Jamun | 6 |
| Arjun | Arjun | 5 |
| Sheesham | Sheesham / Shisham | 5 |
| Buddha's Coconut | Buddha's Coconut | 5 |
| Mahneem | Maha neem / Mahneem | 5 |
| Gulmohar | Gulmohar | 5 |
| Kachnar/Karanj | MISSING / UNKNOWN | 3 |
| Unknown | MISSING / UNKNOWN | 3 |
| Mango, Jamun | MISSING / UNKNOWN | 3 |
| Subabool | Subabool | 3 |
| Bamboo | Bamboo clump | 3 |
| Franjipani | Frangipani | 3 |
| Caribbean trumpet, Ashok | MISSING / UNKNOWN | 3 |
| Semal | Semal / Silk cotton | 2 |
| Bottlebrush | Bottlebrush | 2 |
| Chamrod | Chamrod | 2 |
| Amaltas, Peepal | MISSING / UNKNOWN | 2 |
| Marodphali | Mandphali / Marodphali | 2 |
| Mango | Mango | 2 |
| Teak | Teak | 2 |
| Imli | Imli | 2 |
| Goolar | Goolar | 2 |
| Karanj | Karanj | 2 |
| Maha neem | Maha neem / Mahneem | 2 |
| Amla | Amla | 1 |
| Acacia | Acacia unknown | 1 |
| Bakain tree | Bakain | 1 |
| Bakain | Bakain | 1 |
| Ashok and unknown species | MISSING / UNKNOWN | 1 |
| Anjan | Anjan | 1 |
| Chukrasia | Chukrasia | 1 |
| Caribbean trumpet, Buddha's Coconut | MISSING / UNKNOWN | 1 |
| Bananas | Banana | 1 |
| Kachnar | Kachnar | 1 |
| Bel, next to Moringa | MISSING / UNKNOWN | 1 |
| Bel | Bel | 1 |
| Banyan, Neem | MISSING / UNKNOWN | 1 |
| Caribbean Trumpet | Caribbean trumpet | 1 |
| Caribbean trumpet | Caribbean trumpet | 1 |
| Chir pine or Ashok | MISSING / UNKNOWN | 1 |
| Cassia | Cassia | 1 |
| Bamboo clump | Bamboo clump | 1 |
| Mandphali, Kachnar | MISSING / UNKNOWN | 1 |
| Kanju | Kanju | 1 |
| Gulmohar-like / Amla | MISSING / UNKNOWN | 1 |
| Oblong fruiting tree | MISSING / UNKNOWN | 1 |
| Neem, Bakain | MISSING / UNKNOWN | 1 |
| Moringa | Moringa | 1 |
| Native acacia | Native acacia | 1 |
| Neem, Ashok | MISSING / UNKNOWN | 1 |
| Palash | Palash | 1 |
| Palm tree | Palm tree | 1 |
| Neem/Peepal | MISSING / UNKNOWN | 1 |
| Pilkhan, Neem | MISSING / UNKNOWN | 1 |
| Sausage tree | Sausage tree | 1 |
| Pilkhan or Semal fig | MISSING / UNKNOWN | 1 |
| Silk cotton | Semal / Silk cotton | 1 |
| Shisham | Sheesham / Shisham | 1 |

## 8. Species-status breakdown

| Species status | Count |
|---|---:|
| missing | 2,794 |
| clean | 371 |
| ambiguous_or_unknown | 46 |
| acacia_unknown | 1 |

## 9. Species membership by label configuration

The following sections list the species represented among each label value. Counts are sample/crown counts, not unique species.

### `label_acacia`

**Acacia (`1`): 65 samples**

| Species | Count |
|---|---:|
| Prosopis Juliflora | 53 |
| MISSING / UNKNOWN | 5 |
| Subabool | 3 |
| Neem | 2 |
| Acacia unknown | 1 |
| Native acacia | 1 |

**Non-Acacia (`0`): 326 samples**

| Species | Count |
|---|---:|
| Neem | 49 |
| Ashok | 29 |
| Amaltas | 25 |
| Peepal | 24 |
| Pilkhan | 23 |
| Maulsari | 17 |
| Banyan | 16 |
| MISSING / UNKNOWN | 14 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Kasod | 13 |
| Shahtoot | 9 |
| Maha neem / Mahneem | 7 |
| Siris | 7 |
| Sheesham / Shisham | 6 |
| Jamun | 6 |
| Arjun | 5 |
| Buddha's Coconut | 5 |
| Gulmohar | 5 |
| Bamboo clump | 4 |
| Semal / Silk cotton | 3 |
| Frangipani | 3 |
| Karanj | 2 |
| Mango | 2 |
| Caribbean trumpet | 2 |
| Teak | 2 |
| Chamrod | 2 |
| Imli | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Bottlebrush | 2 |
| Banana | 1 |
| Cassia | 1 |
| Anjan | 1 |
| Palm tree | 1 |
| Kanju | 1 |
| Chukrasia | 1 |
| Amla | 1 |
| Bel | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

**Unknown (`-1`): 2,821 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 2,821 |

### `label_acacia_visual`

**Acacia (`1`): 203 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 190 |
| Prosopis Juliflora | 13 |

**Non-Acacia (`0`): 197 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 189 |
| Neem | 2 |
| Amaltas | 2 |
| Banyan | 1 |
| Peepal | 1 |
| Prosopis Juliflora | 1 |
| Subabool | 1 |

**Unknown (`-1`): 2,812 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 2,461 |
| Neem | 49 |
| Prosopis Juliflora | 39 |
| Ashok | 29 |
| Peepal | 23 |
| Pilkhan | 23 |
| Amaltas | 23 |
| Maulsari | 17 |
| Banyan | 15 |
| Kasod | 13 |
| Mandphali / Marodphali | 13 |
| Saptparni | 13 |
| Shahtoot | 9 |
| Siris | 7 |
| Maha neem / Mahneem | 7 |
| Sheesham / Shisham | 6 |
| Jamun | 6 |
| Buddha's Coconut | 5 |
| Gulmohar | 5 |
| Arjun | 5 |
| Bamboo clump | 4 |
| Frangipani | 3 |
| Semal / Silk cotton | 3 |
| Karanj | 2 |
| Caribbean trumpet | 2 |
| Teak | 2 |
| Bottlebrush | 2 |
| Subabool | 2 |
| Chamrod | 2 |
| Mango | 2 |
| Imli | 2 |
| Goolar | 2 |
| Bakain | 2 |
| Palm tree | 1 |
| Cassia | 1 |
| Acacia unknown | 1 |
| Banana | 1 |
| Anjan | 1 |
| Native acacia | 1 |
| Kanju | 1 |
| Chukrasia | 1 |
| Amla | 1 |
| Bel | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

### `label_acacia_clustering`

**Acacia (`1`): 901 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 853 |
| Prosopis Juliflora | 45 |
| Acacia unknown | 1 |
| Bamboo clump | 1 |
| Banyan | 1 |

**Non-Acacia (`0`): 1,278 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 1,237 |
| Amaltas | 10 |
| Neem | 10 |
| Prosopis Juliflora | 8 |
| Peepal | 3 |
| Subabool | 3 |
| Bamboo clump | 2 |
| Teak | 2 |
| Banyan | 1 |
| Karanj | 1 |
| Kanju | 1 |

**Unknown (`-1`): 1,033 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 750 |
| Neem | 41 |
| Ashok | 29 |
| Pilkhan | 23 |
| Peepal | 21 |
| Maulsari | 17 |
| Amaltas | 15 |
| Banyan | 14 |
| Kasod | 13 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Shahtoot | 9 |
| Siris | 7 |
| Maha neem / Mahneem | 7 |
| Jamun | 6 |
| Sheesham / Shisham | 6 |
| Gulmohar | 5 |
| Buddha's Coconut | 5 |
| Arjun | 5 |
| Frangipani | 3 |
| Semal / Silk cotton | 3 |
| Mango | 2 |
| Chamrod | 2 |
| Imli | 2 |
| Caribbean trumpet | 2 |
| Bottlebrush | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Cassia | 1 |
| Anjan | 1 |
| Banana | 1 |
| Bamboo clump | 1 |
| Karanj | 1 |
| Palm tree | 1 |
| Chukrasia | 1 |
| Bel | 1 |
| Amla | 1 |
| Native acacia | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

### `label_acacia_species`

**Acacia (`1`): 65 samples**

| Species | Count |
|---|---:|
| Prosopis Juliflora | 53 |
| MISSING / UNKNOWN | 5 |
| Subabool | 3 |
| Neem | 2 |
| Acacia unknown | 1 |
| Native acacia | 1 |

**Non-Acacia (`0`): 326 samples**

| Species | Count |
|---|---:|
| Neem | 49 |
| Ashok | 29 |
| Amaltas | 25 |
| Peepal | 24 |
| Pilkhan | 23 |
| Maulsari | 17 |
| Banyan | 16 |
| MISSING / UNKNOWN | 14 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Kasod | 13 |
| Shahtoot | 9 |
| Maha neem / Mahneem | 7 |
| Siris | 7 |
| Sheesham / Shisham | 6 |
| Jamun | 6 |
| Arjun | 5 |
| Buddha's Coconut | 5 |
| Gulmohar | 5 |
| Bamboo clump | 4 |
| Semal / Silk cotton | 3 |
| Frangipani | 3 |
| Karanj | 2 |
| Mango | 2 |
| Caribbean trumpet | 2 |
| Teak | 2 |
| Chamrod | 2 |
| Imli | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Bottlebrush | 2 |
| Banana | 1 |
| Cassia | 1 |
| Anjan | 1 |
| Palm tree | 1 |
| Kanju | 1 |
| Chukrasia | 1 |
| Amla | 1 |
| Bel | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

**Unknown (`-1`): 2,821 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 2,821 |

### `label_acacia_visual_or_species`

**Acacia (`1`): 251 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 194 |
| Prosopis Juliflora | 52 |
| Subabool | 2 |
| Acacia unknown | 1 |
| Neem | 1 |
| Native acacia | 1 |

**Non-Acacia (`0`): 516 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 201 |
| Neem | 50 |
| Ashok | 29 |
| Amaltas | 25 |
| Peepal | 24 |
| Pilkhan | 23 |
| Maulsari | 17 |
| Banyan | 16 |
| Kasod | 13 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Shahtoot | 9 |
| Maha neem / Mahneem | 7 |
| Siris | 7 |
| Jamun | 6 |
| Sheesham / Shisham | 6 |
| Gulmohar | 5 |
| Arjun | 5 |
| Buddha's Coconut | 5 |
| Bamboo clump | 4 |
| Semal / Silk cotton | 3 |
| Frangipani | 3 |
| Caribbean trumpet | 2 |
| Imli | 2 |
| Bottlebrush | 2 |
| Mango | 2 |
| Chamrod | 2 |
| Teak | 2 |
| Karanj | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Anjan | 1 |
| Cassia | 1 |
| Palm tree | 1 |
| Banana | 1 |
| Prosopis Juliflora | 1 |
| Subabool | 1 |
| Kanju | 1 |
| Chukrasia | 1 |
| Amla | 1 |
| Bel | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

**Unknown (`-1`): 2,445 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 2,445 |

### `label_acacia_visual_or_clustering`

**Acacia (`1`): 923 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 873 |
| Prosopis Juliflora | 48 |
| Acacia unknown | 1 |
| Bamboo clump | 1 |

**Non-Acacia (`0`): 1,256 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 1,217 |
| Amaltas | 10 |
| Neem | 10 |
| Prosopis Juliflora | 5 |
| Peepal | 3 |
| Subabool | 3 |
| Bamboo clump | 2 |
| Banyan | 2 |
| Teak | 2 |
| Karanj | 1 |
| Kanju | 1 |

**Unknown (`-1`): 1,033 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 750 |
| Neem | 41 |
| Ashok | 29 |
| Pilkhan | 23 |
| Peepal | 21 |
| Maulsari | 17 |
| Amaltas | 15 |
| Banyan | 14 |
| Kasod | 13 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Shahtoot | 9 |
| Siris | 7 |
| Maha neem / Mahneem | 7 |
| Jamun | 6 |
| Sheesham / Shisham | 6 |
| Gulmohar | 5 |
| Buddha's Coconut | 5 |
| Arjun | 5 |
| Frangipani | 3 |
| Semal / Silk cotton | 3 |
| Mango | 2 |
| Chamrod | 2 |
| Imli | 2 |
| Caribbean trumpet | 2 |
| Bottlebrush | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Cassia | 1 |
| Anjan | 1 |
| Banana | 1 |
| Bamboo clump | 1 |
| Karanj | 1 |
| Palm tree | 1 |
| Chukrasia | 1 |
| Bel | 1 |
| Amla | 1 |
| Native acacia | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

### `label_acacia_species_or_clustering`

**Acacia (`1`): 913 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 853 |
| Prosopis Juliflora | 53 |
| Subabool | 3 |
| Neem | 2 |
| Acacia unknown | 1 |
| Native acacia | 1 |

**Non-Acacia (`0`): 1,549 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 1,237 |
| Neem | 49 |
| Ashok | 29 |
| Amaltas | 25 |
| Peepal | 24 |
| Pilkhan | 23 |
| Maulsari | 17 |
| Banyan | 16 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Kasod | 13 |
| Shahtoot | 9 |
| Maha neem / Mahneem | 7 |
| Siris | 7 |
| Sheesham / Shisham | 6 |
| Jamun | 6 |
| Arjun | 5 |
| Buddha's Coconut | 5 |
| Gulmohar | 5 |
| Bamboo clump | 4 |
| Semal / Silk cotton | 3 |
| Frangipani | 3 |
| Karanj | 2 |
| Mango | 2 |
| Caribbean trumpet | 2 |
| Teak | 2 |
| Chamrod | 2 |
| Imli | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Bottlebrush | 2 |
| Banana | 1 |
| Cassia | 1 |
| Anjan | 1 |
| Palm tree | 1 |
| Kanju | 1 |
| Chukrasia | 1 |
| Amla | 1 |
| Bel | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

**Unknown (`-1`): 750 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 750 |

### `label_acacia_all_priority`

**Acacia (`1`): 931 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 874 |
| Prosopis Juliflora | 52 |
| Subabool | 2 |
| Acacia unknown | 1 |
| Neem | 1 |
| Native acacia | 1 |

**Non-Acacia (`0`): 1,531 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 1,216 |
| Neem | 50 |
| Ashok | 29 |
| Amaltas | 25 |
| Peepal | 24 |
| Pilkhan | 23 |
| Maulsari | 17 |
| Banyan | 16 |
| Kasod | 13 |
| Saptparni | 13 |
| Mandphali / Marodphali | 13 |
| Shahtoot | 9 |
| Maha neem / Mahneem | 7 |
| Siris | 7 |
| Jamun | 6 |
| Sheesham / Shisham | 6 |
| Gulmohar | 5 |
| Arjun | 5 |
| Buddha's Coconut | 5 |
| Bamboo clump | 4 |
| Semal / Silk cotton | 3 |
| Frangipani | 3 |
| Caribbean trumpet | 2 |
| Imli | 2 |
| Bottlebrush | 2 |
| Mango | 2 |
| Chamrod | 2 |
| Teak | 2 |
| Karanj | 2 |
| Bakain | 2 |
| Goolar | 2 |
| Anjan | 1 |
| Cassia | 1 |
| Palm tree | 1 |
| Banana | 1 |
| Prosopis Juliflora | 1 |
| Subabool | 1 |
| Kanju | 1 |
| Chukrasia | 1 |
| Amla | 1 |
| Bel | 1 |
| Moringa | 1 |
| Kachnar | 1 |
| Sausage tree | 1 |
| Palash | 1 |

**Unknown (`-1`): 750 samples**

| Species | Count |
|---|---:|
| MISSING / UNKNOWN | 750 |

## 10. Complete column inventory

| # | Column | Data type | Non-null | Null |
|---:|---|---|---:|---:|
| 1 | `system:index` | `object` | 3,212 | 0 |
| 2 | `A00` | `float64` | 3,212 | 0 |
| 3 | `A01` | `float64` | 3,212 | 0 |
| 4 | `A02` | `float64` | 3,212 | 0 |
| 5 | `A03` | `float64` | 3,212 | 0 |
| 6 | `A04` | `float64` | 3,212 | 0 |
| 7 | `A05` | `float64` | 3,212 | 0 |
| 8 | `A06` | `float64` | 3,212 | 0 |
| 9 | `A07` | `float64` | 3,212 | 0 |
| 10 | `A08` | `float64` | 3,212 | 0 |
| 11 | `A09` | `float64` | 3,212 | 0 |
| 12 | `A10` | `float64` | 3,212 | 0 |
| 13 | `A11` | `float64` | 3,212 | 0 |
| 14 | `A12` | `float64` | 3,212 | 0 |
| 15 | `A13` | `float64` | 3,212 | 0 |
| 16 | `A14` | `float64` | 3,212 | 0 |
| 17 | `A15` | `float64` | 3,212 | 0 |
| 18 | `A16` | `float64` | 3,212 | 0 |
| 19 | `A17` | `float64` | 3,212 | 0 |
| 20 | `A18` | `float64` | 3,212 | 0 |
| 21 | `A19` | `float64` | 3,212 | 0 |
| 22 | `A20` | `float64` | 3,212 | 0 |
| 23 | `A21` | `float64` | 3,212 | 0 |
| 24 | `A22` | `float64` | 3,212 | 0 |
| 25 | `A23` | `float64` | 3,212 | 0 |
| 26 | `A24` | `float64` | 3,212 | 0 |
| 27 | `A25` | `float64` | 3,212 | 0 |
| 28 | `A26` | `float64` | 3,212 | 0 |
| 29 | `A27` | `float64` | 3,212 | 0 |
| 30 | `A28` | `float64` | 3,212 | 0 |
| 31 | `A29` | `float64` | 3,212 | 0 |
| 32 | `A30` | `float64` | 3,212 | 0 |
| 33 | `A31` | `float64` | 3,212 | 0 |
| 34 | `A32` | `float64` | 3,212 | 0 |
| 35 | `A33` | `float64` | 3,212 | 0 |
| 36 | `A34` | `float64` | 3,212 | 0 |
| 37 | `A35` | `float64` | 3,212 | 0 |
| 38 | `A36` | `float64` | 3,212 | 0 |
| 39 | `A37` | `float64` | 3,212 | 0 |
| 40 | `A38` | `float64` | 3,212 | 0 |
| 41 | `A39` | `float64` | 3,212 | 0 |
| 42 | `A40` | `float64` | 3,212 | 0 |
| 43 | `A41` | `float64` | 3,212 | 0 |
| 44 | `A42` | `float64` | 3,212 | 0 |
| 45 | `A43` | `float64` | 3,212 | 0 |
| 46 | `A44` | `float64` | 3,212 | 0 |
| 47 | `A45` | `float64` | 3,212 | 0 |
| 48 | `A46` | `float64` | 3,212 | 0 |
| 49 | `A47` | `float64` | 3,212 | 0 |
| 50 | `A48` | `float64` | 3,212 | 0 |
| 51 | `A49` | `float64` | 3,212 | 0 |
| 52 | `A50` | `float64` | 3,212 | 0 |
| 53 | `A51` | `float64` | 3,212 | 0 |
| 54 | `A52` | `float64` | 3,212 | 0 |
| 55 | `A53` | `float64` | 3,212 | 0 |
| 56 | `A54` | `float64` | 3,212 | 0 |
| 57 | `A55` | `float64` | 3,212 | 0 |
| 58 | `A56` | `float64` | 3,212 | 0 |
| 59 | `A57` | `float64` | 3,212 | 0 |
| 60 | `A58` | `float64` | 3,212 | 0 |
| 61 | `A59` | `float64` | 3,212 | 0 |
| 62 | `A60` | `float64` | 3,212 | 0 |
| 63 | `A61` | `float64` | 3,212 | 0 |
| 64 | `A62` | `float64` | 3,212 | 0 |
| 65 | `A63` | `float64` | 3,212 | 0 |
| 66 | `area` | `object` | 3,212 | 0 |
| 67 | `crown_num` | `int64` | 3,212 | 0 |
| 68 | `crown_uid` | `object` | 3,212 | 0 |
| 69 | `field_description` | `object` | 28 | 3,184 |
| 70 | `field_status` | `object` | 2,028 | 1,184 |
| 71 | `label_acacia` | `int64` | 3,212 | 0 |
| 72 | `label_deciduous` | `int64` | 3,212 | 0 |
| 73 | `label_esd` | `int64` | 3,212 | 0 |
| 74 | `label_red_showy` | `int64` | 3,212 | 0 |
| 75 | `label_showy_flower` | `int64` | 3,212 | 0 |
| 76 | `label_yellow_broad` | `int64` | 3,212 | 0 |
| 77 | `label_yellow_strict` | `int64` | 3,212 | 0 |
| 78 | `lat` | `float64` | 3,212 | 0 |
| 79 | `lon` | `float64` | 3,212 | 0 |
| 80 | `orig_crown_id` | `object` | 3,212 | 0 |
| 81 | `species_clean` | `object` | 372 | 2,840 |
| 82 | `species_raw` | `object` | 418 | 2,794 |
| 83 | `species_status` | `object` | 3,212 | 0 |
| 84 | `source_file` | `object` | 3,212 | 0 |
| 85 | `source_index` | `int64` | 3,212 | 0 |
| 86 | `tree_type_raw` | `object` | 106 | 3,106 |
| 87 | `.geo` | `object` | 3,212 | 0 |
| 88 | `label_acacia_visual` | `int64` | 3,212 | 0 |
| 89 | `visual_label_source` | `object` | 400 | 2,812 |
| 90 | `label_acacia_clustering` | `int64` | 3,212 | 0 |
| 91 | `label_acacia_species` | `int64` | 3,212 | 0 |
| 92 | `label_acacia_visual_or_species` | `int64` | 3,212 | 0 |
| 93 | `label_acacia_visual_or_clustering` | `int64` | 3,212 | 0 |
| 94 | `label_acacia_species_or_clustering` | `int64` | 3,212 | 0 |
| 95 | `label_acacia_all_priority` | `int64` | 3,212 | 0 |

## 11. GEE embedding columns

The dataset contains **64 embedding dimensions**, from `A00` through `A63`.

| Property | Value |
|---|---:|
| Number of dimensions | 64 |
| First dimension | `A00` |
| Last dimension | `A63` |
| Numeric columns with complete/non-null values | 64 / 64 |

## 12. Recommended usage

### For supervised evaluation using confident labels

Use:

```python
X = df[[f'A{i:02d}' for i in range(64)]]
y = df['label_acacia_visual']
```

Filter out unknown/missing labels before supervised training:

```python
mask = df['label_acacia_visual'].isin([0, 1])
X = df.loc[mask, [f'A{i:02d}' for i in range(64)]]
y = df.loc[mask, 'label_acacia_visual']
```

### For experiments comparing label expansion/configurations

Keep the embedding matrix fixed and change only the selected label column, e.g. `label_acacia_visual`, `label_acacia_clustering`, or one of the combined configurations.

## 13. Important interpretation notes

1. `A00`–`A63` are feature dimensions and are not species labels.
2. `label_acacia_visual` is the current **confident-label** field.
3. The label columns are derived fields; they are not separate datasets.
4. A value of `-1` should generally be treated as unknown/unlabelled, not as a third biological class.
5. The exact taxonomy/rules behind each derived label should be maintained in the code/configuration that generated this CSV.
6. When reporting results, always state which label configuration was used.

## 14. Reproducibility

This documentation describes the contents of the exported dataset. For complete reproducibility, keep this file together with the script/notebook/configuration that generated the `label_acacia_*` columns.
