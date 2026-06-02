# Building an Orthomosaic

For this project, orthomosaics are built with the local ODM / NodeODM scripts rather than through the WebODM browser UI.

The main entry point is:

- `make_om.ps1`

It wraps:

- `run_nodeodm_orthomosaic.ps1`

The workflow is:

1. Put all images from one flight date in a single folder.
2. Check that the images already have usable GPS metadata in EXIF.
3. If they do not, prepare a converted copy with GPS copied from XMP to EXIF.
4. Run `make_om.ps1` on that image folder.
5. Wait for the final orthophoto to be written out.
6. Copy or rename the final TIFF into the cleaned orthomosaic input folder used later in the pipeline.

This setup is meant to produce one main output: `odm_orthophoto.tif`. That is the orthomosaic used later for crown detection, tracking, and phenology analysis.

Run the main command like this:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images"
```

By default, the wrapper chooses an output root automatically. If the image folder is inside a path containing `Raw_data`, it maps that to a matching `Processed\ODM` location. Otherwise it writes to `D:\Drone_Phenology_Monitoring\ODM_Output`.

If you want to choose the output folder and dataset name explicitly, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\outputs" `
  -DatasetName "site_date"
```

If the images need GPS copied from XMP to EXIF first, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -PrepareExifFromXmp
```

That uses `Drone-Phenology-Monitoring\src\utility\xmp_to_exif.py` and creates a converted copy before running ODM. Use it only when the images do not already have GPS in EXIF.

If you want a faster preview-style run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -Mode fast
```

Fast mode is useful when you want to check that a dataset processes correctly or when you need a quick preview. For the main analysis workflow, use the standard mode unless there is a reason to trade quality for speed.

The final orthomosaic is written to:

```text
<OutputRoot>\\<DatasetName>\\odm_orthophoto.tif
```

In normal use, `DatasetName` is either taken from the image folder name or passed explicitly with `-DatasetName`.

The default full-mode options currently used by the wrapper are:

```text
--skip-report
--build-overviews
--orthophoto-compression DEFLATE
--orthophoto-resolution 2
```

These settings keep the full reconstruction path and produce a compressed orthophoto at 2 cm per pixel. That is the mode intended for the main phenology workflow.

The fast mode adds:

```text
--fast-orthophoto
--skip-3dmodel
```

Before starting a run, check one or two sample images with ExifTool if you are not sure about the metadata:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "D:\path\to\image.JPG"
```

Interpret that like this:

1. If `GPSLatitude` and `GPSLongitude` exist in EXIF, run orthomosaic creation normally.
2. If only the XMP versions are present, run the wrapper with `-PrepareExifFromXmp`.

After the run completes, the main file to keep is `odm_orthophoto.tif`. That is the orthomosaic used in the rest of the project.

Troubleshooting:

1. If the run fails immediately, check that the image folder exists and actually contains supported image types such as `.jpg`, `.jpeg`, `.tif`, `.tiff`, or `.png`.
2. If ODM cannot use the image locations properly, check the EXIF GPS fields first. Missing EXIF GPS is one of the first things to rule out.
3. If GPU-backed processing is not working, test Docker GPU support with:

```bash
wsl docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

4. If that GPU test fails, fix Docker or WSL GPU support before retrying the orthomosaic run.
5. If you need a quicker diagnostic run, retry with `-Mode fast` to see whether the dataset itself is the issue or whether the full reconstruction path is what is failing.
6. If the output TIFF is created but looks wrong, check the input images for missing metadata, mixed dates, or a folder containing the wrong set of images.
