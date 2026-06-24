# Make Crown-ID Printouts

Create printable orthomosaic maps with numbered crown polygons. The key requirement is traceability: every printed ID should link back to the exact crown layer used for annotation and later analysis.

Use printouts for:

1. Species annotation.
2. Crown-condition checks.
3. Field notes.
4. Quick review when QField is unnecessary.

For repeated monitoring, prefer consensus crowns because `crown_id` represents a tracked tree, not one date's detection.

## Script

```text
src/utility/numbered_crown_overlay.py
```

Reference notebook:

```text
src/utility/overlap_numbered_crowns.ipynb
```

Use the script for repeatable work.

## Inputs

1. Checked orthomosaic GeoTIFF.
2. Crown polygon file in the same coordinate frame.
3. Output `.png` path.
4. Crown ID field to print.

Common crown sources:

| Source | Use |
|---|---|
| Detectree2 `conf_0p45` | Quick single-date inspection. |
| Lower Detectree2 threshold | When missed trees are worse than extra candidates. |
| Consensus crowns | Species labels and repeated monitoring. |
| Manually edited crowns | After QGIS cleanup. |

## Single-Date Detectree2 Printout

```bash
conda run -n dpm-tracking python src/utility/numbered_crown_overlay.py \
  --orthomosaic /path/to/clean_orthomosaics/site_a_15-01-26.tif \
  --crowns /path/to/output/site_a_run/01_detectree/crowns_multithreshold/site_a_15-01-26_multithreshold.gpkg \
  --layer conf_0p45 \
  --output /path/to/printouts/site_a_15-01-26_numbered_crowns.png
```

## Consensus-Crown Printout

```bash
conda run -n dpm-tracking python src/utility/numbered_crown_overlay.py \
  --orthomosaic /path/to/clean_orthomosaics/site_a_15-01-26.tif \
  --crowns /path/to/output/site_a_run/02_tracking/consensus_crowns_complete_all.gpkg \
  --id-field crown_id \
  --output /path/to/printouts/site_a_consensus_numbered_crowns.png
```

## Options

| Option | Meaning |
|---|---|
| `--layer` | GeoPackage layer name, e.g. `conf_0p45`. |
| `--id-field` | Attribute field used as printed label. |
| `--max-size` | Maximum raster preview dimension in pixels. |
| `--dpi` | Output image DPI. |

If `--id-field` is omitted, labels are assigned sequentially. Use stable IDs for serious field annotation.

## Crown ID Rules

Use:

1. `crown_id` from consensus crowns.
2. A manually assigned stable ID.
3. A row ID from an archived single-date layer.

Avoid:

1. Auto-numbering if the crown file may be regenerated.
2. Row order from a layer that may be sorted or edited.
3. IDs from one threshold layer joined to another threshold layer.

Archive the exact crown file used to make the printout.

## Check Before Printing

1. Crown outlines align with the orthomosaic.
2. Labels are readable at paper size.
3. The map is not too crowded.
4. Landmarks or site boundary are visible enough.
5. Filename includes site/date/crown source.
6. Orthomosaic and crown source are recorded.

If labels overlap heavily, split the site into sheets.

## Split Large Sites

Suggested naming:

```text
site_a_15-01-26_sheet_01.png
site_a_15-01-26_sheet_02.png
site_a_15-01-26_sheet_03.png
```

Record:

```csv
sheet,orthomosaic,crowns,layer_or_id_field,notes
sheet_01,site_a_15-01-26.tif,consensus_crowns_complete_all.gpkg,crown_id,north side
sheet_02,site_a_15-01-26.tif,consensus_crowns_complete_all.gpkg,crown_id,south side
```

If layouts are made in QGIS, save the QGIS project/export settings.

## Annotation Sheet

A printout set usually includes:

1. Numbered orthomosaic map.
2. Blank note table.
3. Species/code list if used.
4. Overview map for large sites.
5. Visit date or survey date.

Minimum table:

```text
crown_id | species | notes
```

Better table:

```text
crown_id | species | confidence | health/status | photo_id | notes
```

## Annotation

1. Confirm map orientation.
2. Mark crowns that cannot be located.
3. Mark merged/split crowns.
4. Record uncertainty.
5. Write photo IDs when photos are taken.
6. Do not change crown IDs unless necessary.

## Digitise Notes

Store annotations in a table:

```csv
crown_id,species,species_confidence,status,photo_id,notes,visit_date
101,Azadirachta indica,high,checked,IMG_0012,,2026-01-15
102,,low,uncertain,,could not identify,2026-01-15
```

Then check:

1. Every `crown_id` exists in the printout crown layer.
2. No duplicate IDs unless repeated visits are intended.
3. Species names are standardised.
4. Uncertain labels are preserved.
5. Paper scans/photos are archived if available.
6. Exact crown file is archived.

## Join Notes Back

Join on `crown_id`.

Before joining:

1. Crown layer has unique `crown_id`.
2. Notes table has one row per crown ID.
3. Blanks and uncertain labels have a clear rule.
4. Output is saved as a new file.

Suggested outputs:

```text
site_a_consensus_crowns_species_2026-01-15.gpkg
site_a_species_notes_2026-01-15.csv
```

## Printout Log

```csv
site,visit_date,orthomosaic,crowns,layer,id_field,output_image,notes
site_a,2026-01-15,site_a_15-01-26.tif,consensus_crowns_complete_all.gpkg,,crown_id,site_a_consensus_numbered_crowns.png,
```

## Troubleshooting

1. Crowns do not line up: check CRS and source files.
2. Labels overlap: split into smaller sheets or increase size/DPI.
3. IDs changed: use the archived crown file.
4. Orientation is unclear: add landmarks or overview map.
5. Species names vary: clean names before joining.
6. Many crowns cannot be located: check date, site, and crown layer.
