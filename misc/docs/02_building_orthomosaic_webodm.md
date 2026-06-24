# Build Orthomosaics

Start with one raw image folder per site/date. End with one checked GeoTIFF in a clean analysis folder.

Use either:

1. WebODM browser UI.
2. NodeODM/ODM command-line scripts.

## Output

ODM writes:

```text
<output_root>/<dataset_name>/odm_orthophoto.tif
```

Copy the checked result to a clean folder:

```text
clean_orthomosaics/<site>_DD-MM-YY.tif
clean_orthomosaics/<site>_DD-MM-YY_dateNotConfirmed.tif
clean_orthomosaics/<site>_spot<id>_DD-MM-YY.tif
```

Examples:

```text
site_a_15-01-26.tif
site_a_29-01-26.tif
forest_spot1_10-05-26.tif
```

The analysis pipeline should read the clean folder, not the full ODM processing directory.

## Key Terms

| Term | Meaning |
|---|---|
| Raw image folder | Original drone images from one site/date. |
| ODM/WebODM/NodeODM | Photogrammetry software used to build maps. |
| Orthomosaic | Georeferenced image mosaic corrected for camera perspective. |
| GeoTIFF | `.tif` file with spatial reference information. |
| Clean analysis folder | Folder containing only checked orthomosaics. |
| Dataset name | Name of one ODM processing run. |

## Choose A Route

Use WebODM for small/manual batches and visual job management.

Use scripted NodeODM when processing many flights or when runs should be recorded in a CSV.

## Scripts

| Script | Use |
|---|---|
| `misc/ODM/make_om.ps1` | Build one orthomosaic. |
| `misc/ODM/run_nodeodm_orthomosaic.ps1` | Lower-level NodeODM API runner. |
| `misc/ODM/run_odm_batch.ps1` | Build many orthomosaics from CSV. |
| `misc/ODM/odm_batch_runs.example.csv` | Batch CSV example. |

Local machine setup notes live in `misc/ODM/ODM_OM_RUNBOOK.md`.

## Before Processing

Check each image folder:

1. One site/date only.
2. No mixed spots, dates, or reflights.
3. Original drone files, not screenshots or edited copies.
4. Nadir images if expected.
5. GPS metadata is readable.
6. Enough overlap and coverage.
7. Enough disk space.
8. Final clean-output location is known.

Check GPS on one image:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "<path-to-one-image.JPG>"
```

If GPS exists only in XMP, use `-PrepareExifFromXmp`.

## Folder Layout

```text
project_data/
  raw_drone_images/
    site_a/
      2026_01_15/
      2026_01_29/
  odm_outputs/
    site_a_2026_01_15/
    site_a_2026_01_29/
  clean_orthomosaics/
    site_a_15-01-26.tif
    site_a_29-01-26.tif
```

Keep raw images, ODM outputs, and clean analysis inputs separate.

## Build One Orthomosaic

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "<raw_image_folder>" `
  -OutputRoot "<odm_output_root>" `
  -DatasetName "site_a_2026_01_15"
```

Expected output:

```text
<odm_output_root>/site_a_2026_01_15/odm_orthophoto.tif
```

Default mode is `full`. Use it for final analysis unless there is a specific reason to use preview mode.

## Fast Preview

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "<raw_image_folder>" `
  -OutputRoot "<odm_output_root>" `
  -DatasetName "site_a_2026_01_15_preview" `
  -Mode fast
```

Use `fast` only for quick diagnostics.

## Batch From CSV

CSV:

```csv
image_folder,dataset_name,output_root,mode,prepare_exif_from_xmp,prepared_image_folder,skip_existing
<raw_image_folder_1>,site_a_2026_01_15,<odm_output_root>,full,false,,true
<raw_image_folder_2>,site_a_2026_01_29,<odm_output_root>,full,false,,true
<raw_image_folder_3>,site_a_2026_02_12,<odm_output_root>,full,false,,true
```

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\run_odm_batch.ps1" `
  -RunsCsv "<path-to-odm-runs.csv>" `
  -MinFreeGB C=20,D=50
```

CSV columns:

| Column | Required | Meaning |
|---|---|---|
| `image_folder` | yes | Raw image folder for one site/date. |
| `dataset_name` | yes | ODM output folder name. |
| `output_root` | no | Root folder for ODM outputs. |
| `mode` | no | `full` or `fast`; defaults to `full`. |
| `prepare_exif_from_xmp` | no | `true` if GPS must be copied from XMP to EXIF. |
| `prepared_image_folder` | no | Prepared-image output folder. |
| `skip_existing` | no | Skip row if `odm_orthophoto.tif` already exists. |

Cleanup flags:

| Flag | Action |
|---|---|
| `-CleanupNodeTasks` | Remove completed NodeODM task data after each run. |
| `-DockerPrune` | Prune Docker containers and build cache after each run. |
| `-ClearTemp` | Clear the Windows temp folder after each run. |

Use cleanup flags carefully on shared machines.

## WebODM Manual Flow

1. Create a project.
2. Create one task for one site/date.
3. Upload only that date's images.
4. Use consistent settings across dates.
5. Run the task.
6. Export the orthophoto GeoTIFF.
7. Rename/copy the checked output into the clean folder.

Track jobs:

```csv
site,date,task_name,input_folder,output_tif,status,notes
site_a,2026-01-15,site_a_2026_01_15,<raw_image_folder>,<clean_orthomosaic_path>,checked,
```

## Copy GPS From XMP To EXIF

Check metadata first:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "<path-to-one-image.JPG>"
```

If needed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "<raw_image_folder>" `
  -OutputRoot "<odm_output_root>" `
  -DatasetName "site_a_2026_01_15" `
  -PrepareExifFromXmp
```

Batch row:

```csv
image_folder,dataset_name,output_root,mode,prepare_exif_from_xmp,prepared_image_folder,skip_existing
<raw_image_folder>,site_a_2026_01_15,<odm_output_root>,full,true,<prepared_image_folder>,true
```

Write prepared images to a separate folder. Do not overwrite raw images.

## Quality Check

Open the GeoTIFF in QGIS, WebODM, or another GIS viewer.

Check:

1. Full monitoring area plus buffer is covered.
2. No large holes or missing strips.
3. No obvious folding, warping, or severe blur.
4. Crowns are sharp enough to inspect.
5. Shadows do not hide most crown boundaries.
6. CRS and location are sensible.
7. Site/date is correct.
8. Adjacent dates are broadly comparable.

## Clean Analysis Inputs

After checking:

```text
clean_orthomosaics/site_a_15-01-26.tif
clean_orthomosaics/site_a_29-01-26.tif
clean_orthomosaics/site_a_12-02-26.tif
```

Pipeline examples:

```bash
DPM_OM_DIR=/path/to/clean_orthomosaics
```

```bash
bash src/pipeline/run_pipeline.sh --om-dir /path/to/clean_orthomosaics
```

Keep questionable dates outside the clean folder unless they are clearly marked.

## Processing Log

```csv
site,date,raw_folder,dataset_name,odm_output,clean_tif,status,notes
site_a,2026-01-15,<raw_image_folder>,site_a_2026_01_15,<odm_output_root>/site_a_2026_01_15/odm_orthophoto.tif,<clean_orthomosaic_path>,checked,
site_a,2026-01-29,<raw_image_folder>,site_a_2026_01_29,<odm_output_root>/site_a_2026_01_29/odm_orthophoto.tif,<clean_orthomosaic_path>,check,strong shadows
```

## Handoff To Detectree2

Confirm:

1. Clean folder exists.
2. File names contain site/date information.
3. Bad dates are excluded or marked.
4. GeoTIFFs open correctly.
5. Sites/spots are not mixed.

Then continue to [03_running_detectree2.md](03_running_detectree2.md).

## Troubleshooting

1. Immediate ODM failure: check image folder path and supported image files.
2. Georeferencing failure: inspect EXIF GPS, then XMP GPS.
3. Holes: check overlap, missing images, and interruptions.
4. Warping: check mixed dates, poor GPS, low overlap, or strong wind.
5. Need quick diagnostic: retry with `-Mode fast`.
6. Bad date compared with neighbours: keep it out of tracking until understood.
7. Full disk: clean known temporary outputs only; never delete raw images.
