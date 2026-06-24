# QField Crown Annotation

Prepare a QGIS/QField project for crown-level field annotation.

Main reference:

- https://github.com/jayakrishnascientist/Forest-tree-crown-species-classification-geo-AI/blob/88014b11fc79ba63619e0a6363de2ef3045c356e/QGIS_QField_Tutorial.md

Use QField for editable crown attributes, photo attachments, standard value lists, and less crowded map interaction than paper printouts.

## Inputs

1. Checked orthomosaic.
2. Crown polygon layer, preferably GeoPackage.
3. Stable crown IDs.
4. Field schema.
5. Species/value lists if available.
6. Sync and archive plan.

For repeated monitoring, consensus crowns are usually better than single-date Detectree2 crowns.

## Choose A Crown Layer

| Layer | Use |
|---|---|
| Detectree2 single-date layer | Quick one-date field check. |
| Consensus crowns | Species labels and repeated monitoring. |
| Manually cleaned crowns | Final campaigns after geometry review. |

Archive the crown layer before field editing.

## Field Schema

Minimum:

```text
crown_id, species, species_confidence, notes
```

Broader:

```text
crown_id, species, species_confidence, status, tree_type, health, photo, notes, observer, visit_date
```

Recommended fields:

| Field | Type | Use |
|---|---|---|
| `crown_id` | text/integer | Stable join key; do not edit in field. |
| `species` | text/value map | Species label. |
| `species_confidence` | value map | `high`, `medium`, `low`, `unknown`. |
| `status` | value map | `pending`, `checked`, `uncertain`, `not_found`. |
| `tree_type` | value map | `evergreen`, `deciduous`, `unknown`. |
| `health` | value map/text | Field condition. |
| `photo` | attachment | Field photo. |
| `notes` | text | Free notes. |
| `observer` | text | Annotation source or collector ID. |
| `visit_date` | date/text | Annotation date. |

Keep field names short and stable.

## Prepare Crown Layer

In QGIS:

1. Open the crown layer.
2. Confirm unique `crown_id`.
3. Add missing schema fields.
4. Save as GeoPackage if needed.
5. Prevent geometry edits if only attributes should be edited.
6. Save a backup before QField sync.

Avoid temporary or unsaved memory layers.

## Prepare Orthomosaic

1. Confirm alignment with crowns.
2. Check raster size for phone/tablet use.
3. Create a lighter field copy if needed.
4. Keep the analysis-quality raster unchanged.

## Build QGIS Project

1. Create a dedicated project folder.
2. Create a new QGIS project.
3. Set project CRS.
4. Add orthomosaic base layer.
5. Add editable crown GeoPackage.
6. Confirm alignment at several zoom levels.
7. Style crowns with transparent fill and clear outline.
8. Label crowns with `crown_id` if useful.
9. Configure attribute forms.
10. Save the project.

## Configure Forms

| Field | Widget |
|---|---|
| `species` | Value map or value relation. |
| `species_confidence` | Value map. |
| `status` | Value map. |
| `tree_type` | Value map. |
| `photo` | Attachment. |
| `notes` | Text edit. |
| `visit_date` | Date/calendar or default value. |
| `observer` | Text with default value if useful. |

Example value maps:

```text
species_confidence: high, medium, low, unknown
status: pending, checked, uncertain, not_found
health: good, stressed, dead, damaged, unknown
tree_type: evergreen, deciduous, unknown
```

Use value maps for standard fields. Use free text only where variation is expected.

## Style For Field Use

1. Transparent polygon fills.
2. Bright outlines.
3. Clear selected-crown style.
4. Optional status-based styling.
5. Labels that are readable but not overwhelming.

Example status styling:

| Status | Display |
|---|---|
| `pending` | Yellow outline. |
| `checked` | Green outline. |
| `uncertain` | Orange outline. |
| `not_found` | Red outline. |

## Test Before Fieldwork

1. Save and reopen QGIS project.
2. Check layer paths.
3. Edit a test crown attribute.
4. Test photo attachments if used.
5. Sync/package for QField.
6. Open on phone/tablet.
7. Test offline access.
8. Check raster speed.
9. Check form usability.

## QField Cloud Flow

1. Prepare and save QGIS project.
2. Upload/sync with QField Cloud.
3. Open QField on device.
4. Download project before going to site.
5. Select crowns and fill forms.
6. Save edits regularly.
7. Sync after the visit.
8. Open synced project in QGIS.
9. Export/archive edited crown layer.

Test offline mode before fieldwork if connectivity is uncertain.

## Field Protocol

1. Do not edit `crown_id`.
2. Set `status` for visited crowns.
3. Use `uncertain` instead of guessing.
4. Note merged, split, missing, dead, or inaccessible trees.
5. Attach photos only when useful.
6. Standardise species spelling before fieldwork.
7. Split large sites into zones or ID ranges.

## After Fieldwork

1. Save a copy of the edited GeoPackage.
2. Confirm edits are present.
3. Confirm `crown_id` values are unique and unchanged.
4. Check for accidental geometry edits.
5. Standardise species names.
6. Preserve uncertain labels.
7. Export CSV summary if needed.
8. Archive project, edited layer, and photos.

Suggested names:

```text
site_a_qfield_project_2026-01-15/
site_a_crowns_qfield_edited_2026-01-15.gpkg
site_a_species_labels_2026-01-15.csv
```

## Join Labels Back

Join on `crown_id`.

Before joining:

1. Edited QField layer came from the analysis crown layer.
2. Labels are one row per crown unless repeated visits are intended.
3. `unknown`, `uncertain`, and blanks have clear rules.
4. Joined output is saved as a new file.

Keep the raw edited QField layer unchanged.

## Problems And Fixes

| Problem | Likely cause | Fix |
|---|---|---|
| Layers missing on phone | Broken paths or unsupported setup | Repackage/re-sync; prefer GeoPackage. |
| Raster too slow | Orthomosaic too large | Use lighter field raster. |
| Cannot edit layer | Wrong format or editing disabled | Save as GeoPackage and enable editing. |
| Species names inconsistent | Free-text entry | Use value maps or clean after sync. |
| IDs changed | `crown_id` edited or wrong layer used | Restore backup and lock ID editing next time. |
| Sync is huge | Photos or heavy raster | Reduce attachments or field raster size. |
| Crowns do not align | CRS mismatch or wrong layer pair | Check CRS and source files in QGIS. |

## Deployment Checklist

1. Orthomosaic opens on device.
2. Crown layer opens and is editable.
3. Crown IDs display correctly.
4. Attribute form is simple and complete.
5. Value lists work.
6. Test edit syncs back.
7. Project works offline if needed.
8. Original crown layer is backed up.
9. Field schema is documented.
