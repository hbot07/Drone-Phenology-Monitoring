# ODM Quick Start

Use this when you want the shortest path from a drone image folder to an orthomosaic. For the full operational guide, see [../docs/02_building_orthomosaic_webodm.md](../docs/02_building_orthomosaic_webodm.md). For the current machine-specific WSL/Docker/GPU notes, see [ODM_OM_RUNBOOK.md](ODM_OM_RUNBOOK.md).

## Single Orthomosaic

From the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\processed\ODM" `
  -DatasetName "site_a_2026_01_15"
```

Default mode is `full`. It runs the full dense reconstruction path with:

```text
--skip-report
--build-overviews
--orthophoto-compression DEFLATE
--orthophoto-resolution 2
```

The final file is:

```text
<OutputRoot>\<DatasetName>\odm_orthophoto.tif
```

## Fast Preview

Use this only for diagnostics or quick previews:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\processed\ODM" `
  -DatasetName "site_a_2026_01_15_preview" `
  -Mode fast
```

Fast mode adds:

```text
--fast-orthophoto
--skip-3dmodel
```

## XMP To EXIF GPS Conversion

Use this only when GPS exists in XMP but not EXIF:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\processed\ODM" `
  -DatasetName "site_a_2026_01_15" `
  -PrepareExifFromXmp
```

Check metadata first:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "D:\path\to\image.JPG"
```

## Batch Orthomosaics

Copy the example CSV:

```text
misc\ODM\odm_batch_runs.example.csv
```

Edit it:

```csv
image_folder,dataset_name,output_root,mode,prepare_exif_from_xmp,prepared_image_folder,skip_existing
D:\raw\site_a\2026_01_15,site_a_2026_01_15,D:\processed\ODM,full,false,,true
D:\raw\site_a\2026_01_29,site_a_2026_01_29,D:\processed\ODM,full,false,,true
```

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\run_odm_batch.ps1" `
  -RunsCsv "D:\path\to\odm_runs.csv" `
  -MinFreeGB C=20,D=10
```

Optional cleanup flags:

```powershell
-CleanupNodeTasks
-DockerPrune
-ClearTemp
```

These are opt-in because they remove temporary/container data.

## After ODM Finishes

1. Open `odm_orthophoto.tif` and check the map visually.
2. Confirm the site/date and footprint.
3. Rename or copy the checked TIFF into a clean analysis folder.
4. Use that folder as `DPM_OM_DIR` or `--om-dir` for the analysis pipeline.

Recommended cleaned names:

```text
<site>_DD-MM-YY.tif
<site>_DD-MM-YY_dateNotConfirmed.tif
<site>_spot<id>_DD-MM-YY.tif
```

## Local Project Scripts

The older LHC/SIT/Sanjay Van helper scripts are still in this folder, but they contain local paths and current-project backlog assumptions. Prefer `run_odm_batch.ps1` for reusable work, and see [../docs/06_appendix_local_project_scripts.md](../docs/06_appendix_local_project_scripts.md) for context.
