# ODM Quick Start

Use this if you only want the shortest working path.

For the full setup and troubleshooting details, see:

- [ODM_OM_RUNBOOK.md](d:\Drone_Phenology_Monitoring\ODM_OM_RUNBOOK.md)

## Standard Orthomosaic (Default)

This is the recommended default. Runs the full dense reconstruction pipeline at **2 cm/px**.

Settings applied: `--skip-report --build-overviews --orthophoto-compression DEFLATE --orthophoto-resolution 2`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images"
```

What it does:

1. starts the GPU NodeODM container
2. uploads the images
3. runs full dense reconstruction (GPU-accelerated with DSPSIFT + OpenMVS CUDA)
4. produces a 2 cm/px orthomosaic with DEFLATE compression and overviews
5. waits for completion
6. copies the final TIFF out to the output folder

Default output root:

1. if your folder path contains `Raw_data`, output goes to the matching `Processed\ODM`
2. otherwise output goes to `D:\Drone_Phenology_Monitoring\ODM_Output`

## Example

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\Gaurav2\LHC\07_03_26"
```

## Fast Orthomosaic (Quicker, Lower Quality)

Use this only if you need a quick preview. Skips dense reconstruction. Produces ~5 cm/px.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -Mode fast
```

## If Images Need XMP To EXIF Conversion

Only use this when GPS exists in XMP but not in EXIF.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -PrepareExifFromXmp
```

That creates a converted copy and runs ODM on the converted folder.

## Common Variants

Set a custom output folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\my_outputs\ODM"
```

Set a custom dataset name:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -DatasetName "plot_a_jan_2026"
```

## Output File

The final orthomosaic is copied to:

```text
<OutputRoot>\<DatasetName>\odm_orthophoto.tif
```

## Before You Run

1. Confirm the folder contains drone images.
2. Confirm the dataset already has EXIF GPS, or use `-PrepareExifFromXmp`.
3. Confirm GPU Docker support still works if you changed WSL or Docker.

GPU test command:

```bash
wsl docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Filer Quick Rules

For this machine, use the CSE filer through `nfs_mount_img` with NFS v3 and `nolock`.

Do this:

```powershell
wsl docker run --rm --privileged nfs_mount_img bash -c "mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer && ls /mnt/filer"
```

Do not assume a direct Windows mount or a plain WSL NFS workflow will be stable here.

For long multi-line filer commands from PowerShell, prefer piping a here-string into the container:

```powershell
@'
mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer
du -sh /mnt/filer/IITD_Drone_data
'@ | wsl docker run --rm --privileged -i nfs_mount_img bash
```

## Project Shortcuts

Use these scripts instead of rebuilding commands from scratch:

1. [make_om.ps1](d:\Drone_Phenology_Monitoring\make_om.ps1) for single orthomosaic generation
2. [lhc_sit_make_oms.ps1](d:\Drone_Phenology_Monitoring\lhc_sit_make_oms.ps1) for LHC and SIT backlog runs
3. [sv_make_oms.ps1](d:\Drone_Phenology_Monitoring\sv_make_oms.ps1) for Sanjay Van backlog runs
4. [copy_to_filer.ps1](d:\Drone_Phenology_Monitoring\copy_to_filer.ps1) for generic filer uploads
5. [lhc_sit_upload_oms.sh](d:\Drone_Phenology_Monitoring\lhc_sit_upload_oms.sh) and [sv_upload_oms.sh](d:\Drone_Phenology_Monitoring\sv_upload_oms.sh) for orthomosaic uploads
6. [sv_sync_input_oms.sh](d:\Drone_Phenology_Monitoring\sv_sync_input_oms.sh) to rebuild the local `input/input_om_sv` folder from the filer

## Local Analysis Folder Conventions

Use the cleaned folders under `Drone-Phenology-Monitoring/input/`:

1. `input_om_lhc`: `lhc_DD-MM-YY.tif`
2. `input_om_sit`: `sit_DD-MM-YY.tif` and `sit_DD-MM-YY_dateUnknown.tif` for placeholder-dated legacy files
3. `input_om_sv/spot_X`: `sv_spotX_DD-MM-YY.tif`

Do not rely on the older mixed-name folders when preparing analysis inputs.