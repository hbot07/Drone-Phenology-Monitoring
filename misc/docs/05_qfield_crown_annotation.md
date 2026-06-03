# QField Crown Annotation

Use this workflow to take the orthomosaic and crown polygons into the field and annotate crowns on a phone.

Main reference:

- https://github.com/jayakrishnascientist/Forest-tree-crown-species-classification-geo-AI/blob/88014b11fc79ba63619e0a6363de2ef3045c356e/QGIS_QField_Tutorial.md

The basic workflow is:

1. Prepare the orthomosaic and crown polygons.
2. Prefer GeoPackage for the crown layer.
3. Make sure the CRS is consistent.
4. Create a QGIS project and load the orthomosaic and crown layer.
5. Add the fields needed for annotation.
6. Configure the attribute form.
7. Upload the project with QField Cloud.
8. Open it in QField on the phone.
9. Annotate crowns in the field.
10. Sync the edits back.

In practice, the desktop side is done in QGIS and the field side is done in QField.

The normal sequence is:

1. Load the orthomosaic and crown polygons in QGIS.
2. Confirm that the layers align properly.
3. Add the fields you want to record in the field.
4. Set up the form so data entry is quick and consistent.
5. Push the project to QField Cloud.
6. Download the project on the phone.
7. Use the phone to locate crowns and update attributes.
8. Push the edits back and sync them into QGIS.

When setting up the QGIS project, use the orthomosaic as the visual base layer and the crown polygons as the editable annotation layer. Before uploading anything to QField, check that the project CRS is correct and that the crown polygons sit exactly where they should on the orthomosaic.

Suggested fields:

- `crown_id`
- `species`
- `description`
- `photo`
- `status`
- `tree_type`
- `health`

These can be adjusted depending on the field campaign, but `crown_id` and `species` are the most important ones to keep stable.

If the purpose of the field visit is species identification, the minimum useful setup is usually `crown_id`, `species`, and a free-text notes field. If the visit is broader, then `status`, `tree_type`, `health`, and `photo` become useful additions.

Useful form setup:

- `species` as a value map if you already have a species list
- `status` as a small fixed list such as `pending` and `completed`
- `photo` as an attachment field

Using fixed lists where possible helps keep the data clean. Free-text entry is still useful for notes, but it is better not to rely on it for fields that should stay standardized.

For styling, use a transparent polygon fill and a clear outline so the orthomosaic remains visible underneath. If you are tracking progress in the field, style `pending` and `completed` crowns differently so people can see at a glance what is left to do.

Before sending the project to the phone, make sure the raster is not too heavy for mobile use and that the crown layer is in a format that edits cleanly, usually GeoPackage.

After the field visit, push changes from QField to QField Cloud and sync the updated project back into QGIS or export the updated GeoPackage.

Once the edited layer is back on the desktop, keep a copy of the updated dataset so the field labels used in later analysis are tied to a known version of the crown layer.

Troubleshooting:

1. If the project opens on the phone but layers are missing or broken, check that paths were prepared properly before upload and that the layer formats are supported cleanly.
2. If the raster is too slow on the phone, use a lighter orthomosaic for the field project instead of the heaviest original version.
3. If edits are messy or inconsistent, simplify the form and replace free-text fields with value lists where possible.
4. If multiple people are working on the same project, agree on the field schema and allowed values before going out.
5. After syncing back, archive the updated layer version instead of overwriting files casually, especially if those labels will be used for later analysis.
