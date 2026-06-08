<#
.SYNOPSIS
    Copy local data to the CSE NFS filer using Docker + rsync.

.DESCRIPTION
    Uses a committed Docker image (nfs_mount_img) that already has NFS client
    and rsync installed. A throwaway container is spun up with:
      - the local Windows source bind-mounted at /source (read-only)
      - NFS mounted at /mnt/filer inside the container
    rsync then copies /source/ -> filer. No docker cp involved.

    Why not plain WSL rsync: WSL's NFS client hangs on the mount handshake
    with this particular Cohesity NFS server (confirmed behaviour on this machine).
    The Docker container approach is the one that works.

    One-time setup (already done):
      docker start nfs_mount
      docker exec nfs_mount apt-get install -y rsync
      docker commit nfs_mount nfs_mount_img:latest

.PARAMETER LocalSource
    Windows path to the directory to copy.
    e.g.  D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT\09_05_26

.PARAMETER FilerDest
    Destination path relative to the filer root (script prepends /mnt/filer/).
    e.g.  IITD_Drone_data/Raw_data/SIT/09_05_26

.PARAMETER NfsMountPoint
    NFS server:export.  Default: cohesityiitd.cse.iitd.ac.in:/Aaditeswar

.PARAMETER DryRun
    Preview what would be transferred without writing anything.

.EXAMPLE
    .\copy_to_filer.ps1 `
        -LocalSource "D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT\09_05_26" `
        -FilerDest   "IITD_Drone_data/Raw_data/SIT/09_05_26"

.EXAMPLE
    .\copy_to_filer.ps1 `
        -LocalSource "D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT\09_05_26" `
        -FilerDest   "IITD_Drone_data/Raw_data/SIT/09_05_26" `
        -DryRun
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]  [string]$LocalSource,
    [Parameter(Mandatory=$true)]  [string]$FilerDest,
    [string]$NfsMountPoint = "cohesityiitd.cse.iitd.ac.in:/Aaditeswar",
    [string]$DockerImage   = "nfs_mount_img",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step { param([string]$m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   { param([string]$m) Write-Host "    OK: $m" -ForegroundColor Green }
function Warn { param([string]$m) Write-Host "    WARN: $m" -ForegroundColor Yellow }

# --- 1. Verify local source -------------------------------------------------
Step "Checking local source"
if (-not (Test-Path -LiteralPath $LocalSource)) { throw "Not found: $LocalSource" }
Ok $LocalSource

$filerPath = "/mnt/filer/$FilerDest".TrimEnd('/')
Write-Host "    Local  : $LocalSource"
Write-Host "    Filer  : $filerPath"

# --- 2. Ensure Docker image exists (auto-restore from backup tar if missing) -
Step "Checking Docker image: $DockerImage"
$img = docker images $DockerImage --format "{{.Repository}}" 2>&1
if ($img -notmatch $DockerImage) {
    $backupTar = Join-Path $PSScriptRoot "nfs_mount_img.tar"
    if (Test-Path $backupTar) {
        Warn "Image not found — restoring from backup tar"
        docker load -i $backupTar
    } else {
        throw @"
Image '$DockerImage' not found and no backup tar at $backupTar.
To recreate the image:
  1. docker start nfs_mount           (start the original container)
  2. docker exec nfs_mount apt-get install -y rsync
  3. docker commit nfs_mount nfs_mount_img:latest
  4. docker save nfs_mount_img -o nfs_mount_img.tar   (re-create backup)
"@
    }
}
Ok "Image ready"

# --- 3. rsync via throwaway privileged container ----------------------------
#   --privileged  : required for NFS mount inside container
#   -v source:/source:ro  : bind-mounts local Windows path (read-only)
#   NFS is mounted fresh inside the container, then rsync runs
#   Trailing slash on /source/ = copy contents (not the folder itself)
$rsyncFlags = if ($DryRun) { "--dry-run -avh --progress" } else { "-avh --progress --partial" }

Step "Running rsync$(if ($DryRun) {' (DRY RUN)'})"
if ($DryRun) { Warn "Nothing will be written" }
Write-Host ""

docker run --rm --privileged `
    -v "${LocalSource}:/source:ro" `
    $DockerImage `
    bash -c "mkdir -p '$filerPath' && mount -t nfs -o nolock,vers=3 '$NfsMountPoint' /mnt/filer && rsync $rsyncFlags /source/ '$filerPath/'"

if ($LASTEXITCODE -ne 0) { throw "Failed (exit $LASTEXITCODE)" }
Ok "Done"

# --- 4. Quick check ---------------------------------------------------------
Step "Destination listing"
docker run --rm --privileged $DockerImage `
    bash -c "mount -t nfs -o nolock,vers=3 '$NfsMountPoint' /mnt/filer 2>/dev/null; ls -lh '$filerPath' | head -20"
