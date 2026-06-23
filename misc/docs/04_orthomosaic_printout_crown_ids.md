# Orthomosaic Printouts With Crown IDs

This guide creates a printable image where crown polygons are overlaid on an orthomosaic and each crown is numbered. The printout can be used during ground visits for species labels, tree condition notes, or other crown-level observations.

Use the generic CLI script:

```text
src/utility/numbered_crown_overlay.py
```

## Inputs

You need:

1. One checked orthomosaic GeoTIFF.
2. A crown polygon file for that same orthomosaic or monitoring area.
3. An output image path.

If the crown file is a multi-layer GPKG from Detectree2, choose the confidence layer you want to print, for example `conf_0p45`.

## Basic Command

```bash
conda run -n dpm-tracking python src/utility/numbered_crown_overlay.py \
  --orthomosaic input/input_om_site_a/site_a_15-01-26.tif \
  --crowns output/my_run/01_detectree/crowns_multithreshold/site_a_15-01-26_multithreshold.gpkg \
  --layer conf_0p45 \
  --output output/printouts/site_a_15-01-26_numbered_crowns.png
```

If your crown layer already has stable IDs:

```bash
conda run -n dpm-tracking python src/utility/numbered_crown_overlay.py \
  --orthomosaic input/input_om_site_a/site_a_15-01-26.tif \
  --crowns output/my_run/02_tracking/consensus_crowns_complete_all.gpkg \
  --id-field crown_id \
  --output output/printouts/site_a_consensus_numbered_crowns.png
```

Useful options:

- `--layer`: GPKG layer name.
- `--id-field`: existing attribute field to use as label text.
- `--max-size`: maximum raster preview dimension in pixels.
- `--dpi`: output image DPI.

## Before Printing

Check the output image on screen:

1. Crown polygons align with the orthomosaic.
2. Labels are readable.
3. Labels are not too crowded for field use.
4. The map covers the correct site extent.
5. The crown file used for printing is archived or recorded.

If labels are crowded, split the area into smaller sections or increase output resolution. An unreadable field sheet usually costs more time than making a cleaner printout.

## Field Notes

The minimum useful field record is:

```text
crown_id -> species
```

Other useful fields include:

```text
crown_id -> species, notes, health, status, photo_id
```

After the visit, enter the notes into a clean table or QField-edited layer so labels can be joined back to the same crown file later.

## Appendix: Notebook Reference

The older notebook is still available:

```text
src/utility/overlap_numbered_crowns.ipynb
```

Use it as a reference for plotting experiments, but prefer the CLI script for repeatable project work.

## Troubleshooting

1. If crowns do not line up, check that the crown file and orthomosaic are from the same site/date or same consensus coordinate frame.
2. If labels overlap too much, split the area or increase `--max-size`/`--dpi`.
3. If numbering changes after rerunning crown detection, do not reuse an old printout without also preserving the exact crown file used for that printout.
