# Satellite Data Bundle

## Files

### `acacia_clean_confident_labels.geojson`

Primary clean Acacia/non-Acacia label package.

- Geometry: original crown polygons in the flat crown-feature GeoJSON format.
- Rows: 3,261 crown features.
- Final label column: `label_acacia_clean_confident`.
- Source column: `label_acacia_clean_confident_source`.
- Clustering labels are included as a source-specific diagnostic column, but are
  not used for the final clean label.

### `acacia_clean_plus_clustering_labels.geojson`

Larger Acacia/non-Acacia label package that fills remaining Sanjay Van crowns
with clustering-derived labels.

- Geometry: original crown polygons.
- Rows: 3,261 crown features.
- Final label column: `label_acacia_clean_plus_clustering`.
- Source column: `label_acacia_clean_plus_clustering_source`.

### `configs/`

Species-to-class mapping configs and maintained species trait table used for
the spatial classifiers:

- `acacia_vs_non_acacia.json`
- `deciduous_vs_rest.json`
- `esd_multiclass.json`
- `red_showy.json`
- `showy_flower_vs_rest.json`
- `yellow_broad.json`
- `yellow_showy_strict.json`
- `species_labels.csv`

The JSON configs define which species become positive, negative, multiclass, or
ignored labels for each classifier task. `species_labels.csv` is the compact
species-by-task table to maintain alongside those configs.

### `gee_embeddings/`

Original-crown Google Earth Engine Satellite Embedding export:

- `iitd_sv_gee_embeddings_2024_original.csv`

This table uses the original crown geometries only. Buffer-based embedding
exports are intentionally not included.

## Label Values

For Acacia labels:

- `1`: Acacia
- `0`: non-Acacia
- `-1`: unlabelled / ignored for training

## Label Source Columns

Each GeoJSON keeps source-specific label columns:

- `label_acacia_species_only`: species-derived label from crown species and config.
- `label_acacia_field`: QField/ground mapped Tree type label.
- `label_acacia_desk`: desk visual label from crown images.
- `label_acacia_clustering`: clustering-derived label.

The final label chooses among these source columns according to the package's
priority rule.
