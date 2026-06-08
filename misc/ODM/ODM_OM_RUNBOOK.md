# ODM Orthomosaic Runbook

This document explains the exact workflow that is working on this machine for generating an orthomosaic from a folder of drone nadir images.

It covers:

- what is installed and configured
- how metadata should be handled
- how to run ODM directly without the WebODM UI
- how to run WebODM with a GPU node
- where outputs are written
- how to verify whether a run is actually using the GPU
- common failure modes seen on this machine

This is not generic ODM documentation. It is the runbook for this machine and this workspace.

## 1. Working Setup On This Machine

The machine is currently configured as follows:

- Windows host
- WSL Ubuntu 22.04.5
- NVIDIA RTX A5500 visible inside WSL
- Docker engine running inside WSL
- NVIDIA container toolkit installed in WSL
- Docker daemon configured with IIT proxy settings
- WebODM can start with `--gpu`
- direct NodeODM GPU runs also work

Important detail:

- Native Windows Docker access from PowerShell was not the reliable path for this project.
- The stable execution path is WSL-backed Docker.

## 2. Recommended Way To Make Orthomosaics

For this project, the recommended path is:

1. Use the direct NodeODM GPU script.
2. Point it at a folder of images.
3. Let it upload the images into a GPU-enabled NodeODM container.
4. Wait for processing to finish.
5. Let it copy the final orthomosaic GeoTIFF back to a Windows folder.

This avoids unnecessary WebODM UI/API overhead when the only required product is the orthomosaic.

The main script is:

- [run_nodeodm_orthomosaic.ps1](d:\Drone_Phenology_Monitoring\run_nodeodm_orthomosaic.ps1)

For a shorter user-facing entry point, use:

- [make_om.ps1](d:\Drone_Phenology_Monitoring\make_om.ps1)

For the shortest instructions, see:

- [ODM_QUICKSTART.md](d:\Drone_Phenology_Monitoring\ODM_QUICKSTART.md)

## 3. Files Added Or Updated For This Workflow

The following local files are part of the working setup:

- [make_om.ps1](d:\Drone_Phenology_Monitoring\make_om.ps1)
- [ODM_QUICKSTART.md](d:\Drone_Phenology_Monitoring\ODM_QUICKSTART.md)
- [run_nodeodm_orthomosaic.ps1](d:\Drone_Phenology_Monitoring\run_nodeodm_orthomosaic.ps1)
- [setup_wsl_docker_gpu.sh](d:\Drone_Phenology_Monitoring\setup_wsl_docker_gpu.sh)
- [webodm.sh](d:\Drone_Phenology_Monitoring\WebODM\webodm.sh)
- [xmp_to_exif.py](d:\Drone_Phenology_Monitoring\Drone-Phenology-Monitoring\src\utility\xmp_to_exif.py)

What they are for:

- `make_om.ps1`: simplest user-facing wrapper for fast or full orthomosaic runs.
- `ODM_QUICKSTART.md`: short instructions for the common workflow.
- `run_nodeodm_orthomosaic.ps1`: recommended orthomosaic production command.
- `setup_wsl_docker_gpu.sh`: one-time WSL GPU and proxy setup script.
- `webodm.sh`: patched to prefer Docker Compose v2 over legacy v1.
- `xmp_to_exif.py`: copies GPS coordinates from XMP into EXIF if a dataset needs that conversion.

## 4. One-Time GPU Setup

This has already been completed on this machine.

It did the following:

1. configured Docker daemon proxy via IIT proxy
2. configured apt proxy in WSL
3. installed NVIDIA container toolkit
4. configured Docker runtime for NVIDIA GPUs
5. restarted Docker
6. verified GPU containers with `nvidia-smi`

The script used was:

- [setup_wsl_docker_gpu.sh](d:\Drone_Phenology_Monitoring\setup_wsl_docker_gpu.sh)

If you ever need to rerun it:

```bash
wsl bash /mnt/d/Drone_Phenology_Monitoring/setup_wsl_docker_gpu.sh '<your-sudo-password>'
```

After setup, this command should work:

```bash
wsl docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If that command fails, GPU-backed ODM runs are not ready.

## 5. Metadata Handling: EXIF Versus XMP

### 5.1 What ODM Needs

ODM reads camera GPS information from metadata. In practice, for your workflow, the useful question is whether the drone images already have usable GPS in EXIF.

If the GPS exists only in XMP and not in EXIF, some workflows can fail or behave inconsistently, and you may need the XMP-to-EXIF conversion step.

### 5.2 What Was True For The Tested Folder

For the tested folder:

- `D:\gaurav2\IIT_Delhi_Drone_Data\Raw_data\LHC\9_11_25`

the images already had usable EXIF GPS, so the conversion step was not required for that folder.

### 5.3 How To Check Metadata Before Running ODM

Use ExifTool on one or two sample images first.

Example from PowerShell:

```powershell
exiftool -GPSLatitude -GPSLongitude -XMP:GPSLatitude -XMP:GPSLongitude "D:\path\to\image.JPG"
```

Interpretation:

- If `GPSLatitude` and `GPSLongitude` exist in EXIF, ODM can use them directly.
- If only `XMP:GPSLatitude` and `XMP:GPSLongitude` exist, convert them to EXIF first.

### 5.4 When To Run XMP To EXIF Conversion

Run conversion only when:

1. EXIF GPS is missing
2. XMP GPS is present
3. you want a converted copy of the dataset with EXIF GPS written into the image files

Do not run conversion blindly on every dataset.

### 5.5 How To Run The Existing Conversion Utility

Script:

- [xmp_to_exif.py](d:\Drone_Phenology_Monitoring\Drone-Phenology-Monitoring\src\utility\xmp_to_exif.py)

This script:

1. copies images from a source folder to a target folder
2. reads XMP GPS with ExifTool
3. writes GPS into EXIF in the copied images

Usage:

```powershell
python "D:\Drone_Phenology_Monitoring\Drone-Phenology-Monitoring\src\utility\xmp_to_exif.py" "D:\source_folder" "D:\target_folder"
```

Then run ODM on the converted target folder, not on the original source folder.

Note:

- this script assumes `exiftool` is installed and available on PATH
- it writes `GPSLatitudeRef` and `GPSLongitudeRef` from the trailing compass letter in the XMP value

## 6. Recommended Direct GPU Orthomosaic Workflow

### 6.1 Script

Use:

- [run_nodeodm_orthomosaic.ps1](d:\Drone_Phenology_Monitoring\run_nodeodm_orthomosaic.ps1)

What it does:

1. validates the image folder
2. starts or recreates a GPU NodeODM container
3. uploads all supported images
4. creates and commits a NodeODM task
5. optionally waits for completion
6. copies out the final orthomosaic GeoTIFF

### 6.2 Supported Image Types

The script currently accepts:

- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.png`

### 6.3 Default Processing Mode

The default ODM options in the script are:

```text
--skip-report
--build-overviews
--orthophoto-compression DEFLATE
--orthophoto-resolution 2
```

These settings run the full dense reconstruction pipeline at **2 cm/px**:

- GPU-accelerated feature extraction via DSPSIFT
- Dense point cloud via OpenMVS DensifyPointCloud with CUDA
- Full mesh, texturing, and georeferencing
- DEFLATE-compressed GeoTIFF with overviews

### 6.4 Basic Command

Run from PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\run_nodeodm_orthomosaic.ps1" `
  -ImageFolder "D:\path\to\images" `
  -OutputRoot "D:\path\to\output" `
  -Wait `
  -ResetNode
```

### 6.5 Example Using The Tested Dataset

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\run_nodeodm_orthomosaic.ps1" `
  -ImageFolder "D:\gaurav2\IIT_Delhi_Drone_Data\Raw_data\LHC\9_11_25" `
  -OutputRoot "D:\gaurav2\IIT_Delhi_Drone_Data\Processed\ODM" `
  -DatasetName "LHC_9_11_25_gpu" `
  -Wait `
  -ResetNode
```

### 6.6 Output Location

For the tested run, the final GeoTIFF was written here:

- [odm_orthophoto.tif](d:\gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\LHC_9_11_25_gpu\odm_orthophoto.tif)

### 6.7 If You Want A Different Dataset Name

Use `-DatasetName`.

This affects:

- the task name in NodeODM
- the output subfolder under `-OutputRoot`

### 6.8 If You Do Not Want To Wait For Completion

Drop `-Wait`.

The task will still be created and started, but the script will return immediately after submission.

### 6.9 If You Do Not Want The TIFF Copied Out Automatically

Use:

```powershell
-SkipCopy
```

Then the output remains inside the container task directory until you manually copy it.

## 7. Standard Workflow Command

Use `make_om.ps1` (no `-Mode` flag needed — full is now the default):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images"
```

This runs the full dense reconstruction pipeline at **2 cm/px** orthophoto resolution with GPU acceleration.

If you need only a quick preview (lower quality, skips dense reconstruction):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images" `
  -Mode fast
```

**Important**: do not pass `-OdmArgs` as an array when calling `run_nodeodm_orthomosaic.ps1` via `powershell -File`. PowerShell's `-File` mode joins all array elements into one comma-separated string, breaking the option parsing. Always use `make_om.ps1` as the entry point, which internally uses splatting to pass arrays correctly.

## 8. Running Through WebODM

If you want the WebODM stack itself available with a GPU default node, the command is:

```bash
cd /mnt/d/Drone_Phenology_Monitoring/WebODM
./webodm.sh start --gpu --detached
```

What was required to make this work on this machine:

1. NVIDIA container toolkit in WSL
2. Docker daemon proxy configured
3. Compose v2 plugin installed
4. `webodm.sh` patched to prefer `docker compose` over legacy `docker-compose`
5. GPU node image available locally

### 8.1 What `--gpu` Actually Does

When GPU is detected, WebODM includes:

- [docker-compose.nodeodm.gpu.nvidia.yml](d:\Drone_Phenology_Monitoring\WebODM\docker-compose.nodeodm.gpu.nvidia.yml)

That file sets the default processing node image to:

- `webodm/nodeodm:gpu`

and requests an NVIDIA GPU device.

### 8.2 How To Verify WebODM Is Actually Running A GPU Node

Use:

```bash
wsl docker ps
wsl docker inspect <node-container-name>
wsl docker exec <node-container-name> nvidia-smi
```

On this machine, the active GPU node was verified from inside the container with `nvidia-smi`.

## 9. How To Verify Whether A Run Used GPU In Practice

There are two different questions:

1. Was the container GPU-capable?
2. Did the chosen ODM stages actually use the GPU significantly?

These are not the same thing.

### 9.1 GPU-Capable Container Check

Check:

```bash
wsl docker inspect <container> --format '{{.Config.Image}} {{json .HostConfig.DeviceRequests}}'
```

If the image is a GPU image and device requests include GPU capability, the container is correctly configured.

### 9.2 Inside-Container Check

```bash
wsl docker exec <container> nvidia-smi
```

If this works, GPU devices are visible inside the container.

### 9.3 Live Utilization Check

While the task is running:

```bash
wsl nvidia-smi
```

or:

```bash
wsl watch -n 1 nvidia-smi
```

Interpretation:

- If you used `--fast-orthophoto`, CPU-heavy behavior is normal.
- Low GPU utilization does not mean the setup is broken.
- It often means the chosen processing stages were not the GPU-heavy ones.

## 10. Exact Output You Should Expect

For the direct NodeODM script, the final orthomosaic file copied to Windows is:

```text
<OutputRoot>\<DatasetName>\odm_orthophoto.tif
```

For example:

- [odm_orthophoto.tif](d:\gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\LHC_9_11_25_gpu\odm_orthophoto.tif)

Inside the NodeODM task container, ODM’s native path is:

```text
/var/www/data/<task_uuid>/odm_orthophoto/odm_orthophoto.tif
```

## 11. Troubleshooting

### 11.1 GPU Container Runs Fail

Check this first:

```bash
wsl docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, fix Docker GPU support before touching ODM.

### 11.2 Docker Pulls Time Out

This machine needed daemon-level proxy configuration, not just shell environment variables.

The Docker daemon proxy is configured via:

- `/etc/systemd/system/docker.service.d/http-proxy.conf`

The apt proxy is configured via:

- `/etc/apt/apt.conf.d/95proxy`

### 11.3 `webodm.sh start --gpu` Fails With `ContainerConfig`

Cause:

- legacy `docker-compose` v1 bug with current Docker engine metadata

Fix used here:

1. install Docker Compose v2 plugin
2. patch [webodm.sh](d:\Drone_Phenology_Monitoring\WebODM\webodm.sh) to prefer `docker compose`

### 11.4 Windows-Mounted NodeODM Task Storage Fails

An earlier approach used a Windows bind mount for NodeODM task storage. On this machine that caused task creation / commit problems.

The stable workaround was:

1. keep task storage inside the container
2. copy out only the final orthophoto TIFF

That is exactly what [run_nodeodm_orthomosaic.ps1](d:\Drone_Phenology_Monitoring\run_nodeodm_orthomosaic.ps1) now does.

### 11.5 Dataset Has GPS In XMP But Not EXIF

Use:

- [xmp_to_exif.py](d:\Drone_Phenology_Monitoring\Drone-Phenology-Monitoring\src\utility\xmp_to_exif.py)

and run ODM on the converted output folder.

### 11.6 You Want To Watch ODM Progress

With the direct script and `-Wait`, progress is printed periodically.

Without `-Wait`, query NodeODM directly:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:3001/task/<uuid>/info?with_output=0"
Invoke-RestMethod -Uri "http://127.0.0.1:3001/task/<uuid>/output?line=0"
```

### 11.7 Output Is Low Resolution (5 cm/px Instead Of Expected Higher Resolution)

The default `--orthophoto-resolution` in ODM is `5` cm/px.

`make_om.ps1` with `-Mode full` sets it to **2 cm/px** via `--orthophoto-resolution 2`.

Important caveat: ODM caps resolution at the GSD estimate computed from your images. If the GSD estimate is coarser than 2 cm/px, ODM will silently use the GSD value instead.

Do not pass `-OdmArgs` to `run_nodeodm_orthomosaic.ps1` directly from the command line. PowerShell's `-File` execution mode joins array values into a single comma-separated string. Use `make_om.ps1` instead — it calls the runner via splatting (`@runParams`), which correctly passes the array.

If you need to add custom options, modify the `$odmArgs` array inside `make_om.ps1` directly.

## 12. Minimal Run Checklist

Before a new run:

1. confirm image folder exists
2. confirm there are supported image files inside it
3. confirm EXIF GPS exists, or convert XMP to EXIF first
4. confirm GPU container test works if you want GPU-backed processing
5. run the script
7. verify the TIFF exists at the expected output location

## 13. Most Common Commands

### 13.1 Standard GPU Orthomosaic (2 cm/px)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\Drone_Phenology_Monitoring\make_om.ps1" `
  -ImageFolder "D:\path\to\images"
```

### 13.2 Check GPU Container Support

```bash
wsl docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 13.3 Start WebODM With GPU Node

```bash
cd /mnt/d/Drone_Phenology_Monitoring/WebODM
./webodm.sh start --gpu --detached
```

### 13.4 Convert XMP GPS To EXIF GPS

```powershell
python "D:\Drone_Phenology_Monitoring\Drone-Phenology-Monitoring\src\utility\xmp_to_exif.py" "D:\source_folder" "D:\target_folder"
```

## 14. Final Recommendation

For this project, if the goal is simply to produce orthomosaic TIFFs from nadir-image folders, use the direct GPU NodeODM script rather than the WebODM UI.

## 15. Accessing The CSE Filer Reliably

The filer used in this project is:

- `cohesityiitd.cse.iitd.ac.in:/Aaditeswar`

On this machine, the stable path is not a direct Windows NFS mount and not a plain WSL NFS workflow.

What is reliable here:

1. run NFS operations inside the Docker image `nfs_mount_img`
2. mount the filer inside that container with `-t nfs -o vers=3,nolock`
3. bind local Windows folders into the container when copying data

Important practical detail:

- For multi-line bash scripts sent from PowerShell to WSL Docker, pipe the script body into `wsl docker run ... bash`.
- That avoids CRLF and quoting issues that caused brittle failures during this project.

### 15.1 Known-Good Mount Pattern

Use this exact mount option set:

```bash
mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer
```

The following patterns were used successfully during this project:

```powershell
wsl docker run --rm --privileged nfs_mount_img bash -c "mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer && ls /mnt/filer"
```

and, for longer scripts:

```powershell
@'
mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer
du -sh /mnt/filer/IITD_Drone_data
du -sh /mnt/filer/Sanjay_Van_Data
'@ | wsl docker run --rm --privileged -i nfs_mount_img bash
```

### 15.2 Reading From The Filer

Typical read-only checks that were used successfully:

```powershell
wsl docker run --rm --privileged nfs_mount_img bash -c "mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer && ls /mnt/filer/IITD_Drone_data/Raw_data/LHC"
```

```powershell
wsl docker run --rm --privileged nfs_mount_img bash -c "mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer && du -sh /mnt/filer/IITD_Drone_data /mnt/filer/Sanjay_Van_Data"
```

What to check first when looking for missing orthomosaics:

1. whether the expected date folder exists
2. whether it contains `odm_orthophoto.tif` at the root or inside `odm_orthophoto/`
3. whether the folder is a raw-image bundle only, with no orthomosaic output yet

That distinction mattered for Sanjay Van spot 2 and spot 3 on `10-08-25`: the bundle existed, but no orthomosaic had been produced.

### 15.3 Writing To The Filer

Use:

- [copy_to_filer.ps1](d:\Drone_Phenology_Monitoring\copy_to_filer.ps1)

This script is the stable upload path for arbitrary local folders because it:

1. restores `nfs_mount_img` from `nfs_mount_img.tar` if needed
2. bind-mounts the Windows source folder into the container at `/source`
3. mounts the filer inside the container
4. runs `rsync` directly to the filer destination

Example:

```powershell
.\copy_to_filer.ps1 `
  -LocalSource "D:\path\to\local_folder" `
  -FilerDest "IITD_Drone_data/Raw_data/SIT/09_05_26"
```

Use `-DryRun` before large transfers.

### 15.4 Site-Specific Upload Helpers

For orthomosaic uploads that match the project directory layout, use the existing helper scripts instead of ad-hoc commands:

- [lhc_sit_upload_oms.sh](d:\Drone_Phenology_Monitoring\lhc_sit_upload_oms.sh)
- [sv_upload_oms.sh](d:\Drone_Phenology_Monitoring\sv_upload_oms.sh)

These scripts already encode the intended filer destinations and skip files that are already present.

## 16. Project Naming And Sync Conventions

The current local analysis folders are intentionally cleaned up. Prefer these folders over older ad-hoc folders:

- `Drone-Phenology-Monitoring/input/input_om_lhc`
- `Drone-Phenology-Monitoring/input/input_om_sit`
- `Drone-Phenology-Monitoring/input/input_om_sv`

### 16.1 LHC Naming

Use:

- `lhc_DD-MM-YY.tif`

Example:

- `lhc_07-03-26.tif`

### 16.2 SIT Naming

Use:

- `sit_DD-MM-YY.tif`

If a file had no trustworthy acquisition date but still needed to remain in chronological analyses, use:

- `sit_DD-MM-YY_dateUnknown.tif`

Those `_dateUnknown` files are placeholder-dated legacy orthomosaics, not confirmed survey dates.

### 16.3 Sanjay Van Naming

Store files by spot in subfolders under `input/input_om_sv` and name them as:

- `sv_spotX_DD-MM-YY.tif`

Examples:

- `input/input_om_sv/spot_1/sv_spot1_31-01-26.tif`
- `input/input_om_sv/spot_4/sv_spot4_00-00-0000.tif`

### 16.4 SV Sync Rules

Use:

- [sv_sync_input_oms.sh](d:\Drone_Phenology_Monitoring\sv_sync_input_oms.sh)

What it currently does:

1. mounts the filer inside the NFS Docker container
2. walks `Sanjay_Van_Data/ortho_images/spot_1` through `spot_4`
3. accepts both new-style `YYMMDD` folders and older bundle folders with embedded `DD-MM-YY`
4. falls back to `00-00-0000` only when no parseable date exists
5. skips files that are already present locally

Do not flatten all SV orthomosaics into a single folder anymore. The per-spot structure is the intended layout.

## 17. Batch Processing And Cleanup Rules

The batch scripts that were validated in this project are:

- [lhc_sit_make_oms.ps1](d:\Drone_Phenology_Monitoring\lhc_sit_make_oms.ps1)
- [sv_make_oms.ps1](d:\Drone_Phenology_Monitoring\sv_make_oms.ps1)

Shared operating rules from those runs:

1. keep using `make_om.ps1` as the entry point
2. prefer `-Mode fast` for backlog-clearing batches unless full reconstruction is explicitly required
3. monitor free space before every run
4. remove NodeODM task data after each completed batch item
5. prune stopped containers and temporary artifacts between runs

Drive-space checks that were used successfully:

- keep at least about `20 GB` free on `C:`
- keep at least about `10 GB` free on the local data drive used by that dataset (`D:` or `E:` in these scripts)

## 18. Filer Audit Commands That Worked

For storage inspection on this machine, these commands were reliable:

```powershell
Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -gt 0 } | Select-Object Name, @{N='Total GB';E={[math]::Round(($_.Used+$_.Free)/1GB,1)}}, @{N='Used GB';E={[math]::Round($_.Used/1GB,1)}}, @{N='Free GB';E={[math]::Round($_.Free/1GB,1)}} | Format-Table -AutoSize
```

```powershell
wsl docker run --rm --privileged nfs_mount_img bash -c "mount -t nfs -o vers=3,nolock cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer && du -sh /mnt/filer/IITD_Drone_data/Raw_data/LHC /mnt/filer/IITD_Drone_data/Raw_data/SIT /mnt/filer/IITD_Drone_data/orthomosaics/LHC /mnt/filer/IITD_Drone_data/orthomosaics/SIT /mnt/filer/Sanjay_Van_Data/ortho_images/spot_1 /mnt/filer/Sanjay_Van_Data/ortho_images/spot_2 /mnt/filer/Sanjay_Van_Data/ortho_images/spot_3 /mnt/filer/Sanjay_Van_Data/ortho_images/spot_4 /mnt/filer/IITD_Drone_data /mnt/filer/Sanjay_Van_Data"
```

Important interpretation from the audit:

- `IITD_Drone_data` is small enough to reason about directly from raw-data and orthomosaic subfolders
- `Sanjay_Van_Data` is dominated by `raw_data`, especially `audio_raw`, not by the orthomosaic TIFFs
- a quick `du` on the per-spot orthomosaic folders is not representative of the full `Sanjay_Van_Data` footprint

Use WebODM only if you specifically need:

- the project/task UI
- task management through the browser
- node management through the web app
- other WebODM features beyond just orthomosaic generation

For batch orthomosaic production, the direct script is the simpler and more reproducible path on this machine.