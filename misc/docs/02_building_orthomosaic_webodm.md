# Building an Orthomosaic

This project builds orthomosaics with local ODM and NodeODM scripts rather than the WebODM browser UI. The working scripts and runbooks are now stored in `misc/ODM/`.

Start with these references:

1. `misc/ODM/ODM_QUICKSTART.md` - shortest working path.
2. `misc/ODM/ODM_OM_RUNBOOK.md` - detailed machine-specific setup, GPU notes, and troubleshooting.
3. `misc/ODM/make_om.ps1` - normal single-dataset entry point.
4. `misc/ODM/run_nodeodm_orthomosaic.ps1` - lower-level NodeODM runner used by the wrapper.

The output needed by the phenology pipeline is one GeoTIFF per date: `odm_orthophoto.tif`. After checking it, copy or rename it into the cleaned orthomosaic input folder used by the pipeline.

## Overall Workflow

1. Put all images from one flight date in a single folder.
2. Confirm that the images belong to one site, one date, and one mission or intended reflown mission.
3. Check that the images have usable GPS metadata in EXIF.
4. If GPS exists only in XMP, run the wrapper with `-PrepareExifFromXmp`.
5. Run `misc/ODM/make_om.ps1` on the image folder.
6. Wait for ODM to write the final orthophoto.
7. Inspect the orthomosaic visually.
8. Copy or rename the final TIFF into the cleaned `input/input_om_*` folder for analysis.

## Standard Single Orthomosaic Run

Run from PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images"
```

By default, the wrapper chooses an output root automatically. If the image folder path contains `Raw_data`, output goes to the matching `Processed\ODM` location. Otherwise output goes to `misc\ODM\ODM_Output` relative to the script location.

To choose the output folder and dataset name explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\outputs" `
  -DatasetName "site_date"
```

The final orthomosaic is written to:

```text
<OutputRoot>\<DatasetName>\odm_orthophoto.tif
```

## Full Mode And Fast Mode

The default mode is `full`. It uses:

```text
--skip-report
--build-overviews
--orthophoto-compression DEFLATE
--orthophoto-resolution 2
```

This is the intended mode for main phenology analysis because it keeps the full dense reconstruction path and produces a compressed orthophoto at 2 cm per pixel.

For a quicker diagnostic run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -Mode fast
```

Fast mode adds:

```text
--fast-orthophoto
--skip-3dmodel
```

Use fast mode to check whether a dataset processes at all, to produce a quick preview, or to work through a backlog where lower reconstruction detail is acceptable. For final crown detection and tracking, prefer full mode unless the project has explicitly accepted the quality tradeoff.

## Metadata Check

ODM needs camera location metadata. Before starting a run, check one or two sample images with ExifTool:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "D:\path\to\image.JPG"
```

Interpret the result like this:

1. If `GPSLatitude` and `GPSLongitude` exist in EXIF, run orthomosaic creation normally.
2. If only `XMP:GPSLatitude` and `XMP:GPSLongitude` exist, use `-PrepareExifFromXmp`.
3. If neither exists, fix the image metadata or source data before running ODM.

When conversion is needed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "misc\ODM\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -PrepareExifFromXmp
```

That calls `src/utility/xmp_to_exif.py`, creates a converted copy, and runs ODM on the converted folder. Use this only when the dataset actually needs it.

## Batch Helpers In misc/ODM

The folder `misc/ODM/` also contains site-specific helpers that document the backlog and naming conventions used for this project:

1. `misc/ODM/lhc_sit_make_oms.ps1` - batch-creates missing LHC and SIT orthomosaics.
2. `misc/ODM/sv_make_oms.ps1` - batch-creates Sanjay Van spot orthomosaics.
3. `misc/ODM/drone_data.csv` - LHC/SIT date-level raw-data and orthomosaic status.
4. `misc/ODM/sanjay_van_data.csv` - Sanjay Van spot/date status and filer notes.
5. `misc/ODM/lhc_sit_upload_oms.sh` and `misc/ODM/sv_upload_oms.sh` - upload helpers for processed orthomosaics.
6. `misc/ODM/sv_sync_input_oms.sh` - rebuilds the local Sanjay Van input orthomosaic folder from filer data.

Use these files as references when preparing dated analysis inputs. They help answer questions such as which dates already have orthomosaics, which spots exist for Sanjay Van, and which dates were placeholder or filer-derived.

## Cleaned Analysis Inputs

After ODM finishes, do not point the detection pipeline at arbitrary processed folders. Put checked orthomosaics into the cleaned input folders using stable names:

1. `input/input_om_lhc`: `lhc_DD-MM-YY.tif`
2. `input/input_om_sit`: `sit_DD-MM-YY.tif`
3. `input/input_om_sv/spot_X`: `sv_spotX_DD-MM-YY.tif`

The detection and tracking scripts rely on predictable naming and date ordering. If a date is uncertain, preserve the uncertainty in the filename and notes rather than silently treating it as confirmed.

## Quality Check Before Detection

Before moving to Detectree2:

1. Open the orthomosaic and check for missing strips, blurred regions, or obvious warping.
2. Confirm the site footprint matches the expected monitoring area.
3. Confirm the date and site name are correct.
4. Check that the GeoTIFF opens in QGIS or another GIS viewer.
5. Compare the orthomosaic against nearby dates if the canopy or alignment looks unusual.

## Troubleshooting

1. If the run fails immediately, check that the image folder exists and contains supported image types: `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.png`.
2. If ODM cannot use image locations properly, check EXIF GPS first.
3. If GPU-backed processing is not working, test Docker GPU support:

```bash
wsl docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

4. If the GPU test fails, fix WSL/Docker GPU support before retrying. See `misc/ODM/ODM_OM_RUNBOOK.md`.
5. If a full run fails but you need a quick diagnostic, retry with `-Mode fast`.
6. If the TIFF exists but looks wrong, check for mixed dates, missing metadata, insufficient overlap, or a folder containing the wrong image set.
7. If disk space is low, review the cleanup behavior in `misc/ODM/lhc_sit_make_oms.ps1` or `misc/ODM/sv_make_oms.ps1` before running large batches.
