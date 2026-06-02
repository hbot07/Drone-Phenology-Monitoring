# Orthomosaic Printout with Crown IDs

Use:

- `Drone-Phenology-Monitoring\\src\\utility\\overlap_numbered_crowns.ipynb`

This notebook overlays detected crowns on the orthomosaic, assigns numbers to the crowns, and saves an output image for printing.

This is the bridge between the remote workflow and the ground visit. Once the crowns are numbered and printed, the printout can be carried into the field and used to record crown-level labels such as species.

Input needed:

1. The orthomosaic.
2. The crown file for that orthomosaic, usually as `.gpkg` or `.geojson`.
3. The output path for the numbered overlay image.

If the crowns are stored in a multi-layer GPKG, first make sure you are using the specific layer you want to number.

In the notebook, set:

```python
img_path = r"D:\path\to\orthomosaic.tif"
geojson_path = r"D:\path\to\crowns.geojson"
output_path = r"D:\path\to\numbered_crowns_overlay.png"
```

Then run the notebook cell that calls:

```python
save_overlay_with_numbers(img_path, geojson_path, output_path)
```

That function reads the orthomosaic, reads the crown polygons, draws the polygon boundaries, writes a number label on each crown, and saves the final overlay image.

After that:

1. Check that the crown boundaries line up with the orthomosaic.
2. Check that the labels are readable.
3. Print the output image.
4. Take the printout to the field.

If the labels are too crowded, it is better to change the plotting scale or split the output into smaller sections before printing than to take an unreadable sheet to the field.

The printout is used during ground visits to note down crown-level labels such as species. In practice the field note is usually a mapping like:

```text
crown_id -> species
```

The same printout can also be used for other crown-level notes such as condition, health, or anything else needed during the ground visit.

After the visit, those notes should be entered into a clean table so they can be joined back to the crown layer later.

If needed, the numbered crown layer can also be loaded in QGIS and synced to QField to help locate crowns on the ground.

Troubleshooting:

1. If the crowns do not line up with the orthomosaic, check that the crown file and orthomosaic are from the same run and same area.
2. If labels overlap too much, reduce the map extent, split the area into sections, or adjust the plotting settings before printing.
3. If the numbering changes after rerunning crown detection, do not assume the old printout is still valid. Keep the exact crown file used to generate the printed version.
