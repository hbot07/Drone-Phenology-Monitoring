# Building Orthomosaics

This guide covers the generic path from a folder of drone images to one georeferenced orthomosaic GeoTIFF. The repository supports a direct NodeODM workflow through PowerShell scripts in `misc/ODM/`; the same ideas also apply if you use the WebODM browser UI.

The reusable scripts are:

- `misc/ODM/make_om.ps1`: single orthomosaic entry point.
- `misc/ODM/run_nodeodm_orthomosaic.ps1`: lower-level NodeODM API runner used by `make_om.ps1`.
- `misc/ODM/run_odm_batch.ps1`: CSV-driven batch runner for many image folders.
- `misc/ODM/odm_batch_runs.example.csv`: example batch config.

Machine-specific notes are kept in `misc/ODM/ODM_OM_RUNBOOK.md`.

## Goal

For every flight date, produce one checked GeoTIFF:

```text
<output_root>/<dataset_name>/odm_orthophoto.tif
```

After checking it visually, copy or rename it into the cleaned analysis input folder used by the pipeline.

Recommended analysis names:

```text
<site>_DD-MM-YY.tif
<site>_DD-MM-YY_dateNotConfirmed.tif
<site>_spot<id>_DD-MM-YY.tif
```

## Before Processing

Check that:

1. The image folder contains one site, one date, and one intended mission or reflown mission.
2. The images are nadir shots if the monitoring campaign expects nadir imagery.
3. The images have usable GPS metadata.
4. The folder has enough images and overlap to reconstruct the whole monitoring area.
5. You have enough disk space for ODM outputs and temporary files.

ODM needs camera positions. Check metadata on one or two images:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "D:\path\to\image.JPG"
```

Use `-PrepareExifFromXmp` only when GPS exists in XMP but not in EXIF.

## Single Orthomosaic

From PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\processed\ODM" `
  -DatasetName "site_a_2026_01_15"
```

Default mode is `full`. It uses a full dense reconstruction path and writes a compressed orthophoto at 2 cm per pixel.

For a quick preview:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\processed\ODM" `
  -DatasetName "site_a_2026_01_15_preview" `
  -Mode fast
```

Use `fast` only for diagnostics or accepted lower-quality previews. For final crown detection and tracking, prefer `full`.

## Batch Orthomosaics From CSV

Create a CSV like:

```csv
image_folder,dataset_name,output_root,mode,prepare_exif_from_xmp,prepared_image_folder,skip_existing
D:\raw\site_a\2026_01_15,site_a_2026_01_15,D:\processed\ODM,full,false,,true
D:\raw\site_a\2026_01_29,site_a_2026_01_29,D:\processed\ODM,full,false,,true
```

Then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\run_odm_batch.ps1" `
  -RunsCsv "D:\path\to\odm_runs.csv" `
  -MinFreeGB C=20,D=10
```

Optional cleanup flags:

- `-CleanupNodeTasks`: remove completed NodeODM task data after each run.
- `-DockerPrune`: prune Docker containers/build cache after each run.
- `-ClearTemp`: clear the Windows temp folder after each run.

These cleanup flags are intentionally opt-in because they delete local temporary/container data.

## If GPS Must Be Copied From XMP To EXIF

Single run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\processed\ODM" `
  -DatasetName "site_a_2026_01_15" `
  -PrepareExifFromXmp
```

Batch CSV:

```csv
image_folder,dataset_name,output_root,mode,prepare_exif_from_xmp,prepared_image_folder,skip_existing
D:\raw\site_a\2026_01_15,site_a_2026_01_15,D:\processed\ODM,full,true,D:\prepared\site_a_2026_01_15_exif,true
```

## Quality Check Before Analysis

Open each orthomosaic before using it downstream. Check:

1. No missing strips or large holes.
2. No obvious warping.
3. The footprint covers the monitoring area plus a small buffer.
4. The GeoTIFF opens in QGIS or another GIS viewer.
5. The site/date name is correct.
6. Nearby dates look comparable enough for crown tracking.

If one date is visibly poor, keep it out of the tracking series until it is understood. A single bad orthomosaic can create many bad graph edges.

## Cleaned Analysis Inputs

Do not point the tracking pipeline at arbitrary ODM output folders. Copy checked orthomosaics into a clean analysis folder with stable names, for example:

```text
input/input_om_site_a/site_a_15-01-26.tif
input/input_om_site_a/site_a_29-01-26.tif
input/input_om_site_a/site_a_12-02-26.tif
```

Then use that folder as `DPM_OM_DIR` or `--om-dir`.

## Troubleshooting

1. If the run fails immediately, check that the image folder exists and contains supported images: `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.png`.
2. If ODM cannot georeference properly, check EXIF GPS first.
3. If GPU processing is expected, verify Docker/WSL/NVIDIA setup using the machine-specific runbook.
4. If full mode fails but you need a quick diagnostic, retry with `-Mode fast`.
5. If the orthomosaic looks wrong, check for mixed dates, poor overlap, heavy wind, or an image folder containing the wrong mission.
