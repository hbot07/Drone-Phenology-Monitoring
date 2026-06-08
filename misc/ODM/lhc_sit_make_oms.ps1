##############################################################
# lhc_sit_make_oms.ps1
# Batch-create missing LHC and SIT orthomosaics (fast 2cm mode)
# Monitors C: and D: drive space, cleans NodeODM tasks after each run
#
# Missing OMs to create:
#   LHC: 16_03_26 (35 images)
#   SIT: 9_11_25, 20_11_25, 28_01_26, 29_1_26, 20_02_26,
#        16_03_26 (80m flight)
#   Note: 17_03_26_50m excluded (experimental flight, not processed)
##############################################################

$makeOm   = "D:\Drone_Phenology_Monitoring\make_om.ps1"
$lhcBase  = "D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\LHC"
$sitBase  = "D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT"
$lhcOut   = "D:\Gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\lhc_processed"
$sitOut   = "D:\Gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\sit_processed"

# img       : source image folder
# outRoot   : output root (site-specific subfolder)
# name      : DatasetName -> output tif at outRoot\name\odm_orthophoto.tif
$runs = @(
    @{ img="$lhcBase\16_03_26";       outRoot=$lhcOut; name='LHC_16_03_26' },
    @{ img="$sitBase\9_11_25";        outRoot=$sitOut; name='SIT_9_11_25'  },
    @{ img="$sitBase\20_11_25";       outRoot=$sitOut; name='SIT_20_11_25' },
    @{ img="$sitBase\28_01_26";       outRoot=$sitOut; name='SIT_28_01_26' },
    @{ img="$sitBase\29_1_26";        outRoot=$sitOut; name='SIT_29_1_26'  },
    @{ img="$sitBase\20_02_26";       outRoot=$sitOut; name='SIT_20_02_26' },
    @{ img="$sitBase\16_03_26_80m";   outRoot=$sitOut; name='SIT_16_03_26' }
)

function Get-FreeGB {
    param([string]$Drive)
    return [math]::Round((Get-PSDrive $Drive -ErrorAction SilentlyContinue).Free / 1GB, 1)
}

function Remove-NodeODMTasks {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:3001/task/list" -Method Get `
            -ErrorAction Stop
        foreach ($t in $resp) {
            Invoke-RestMethod -Uri "http://127.0.0.1:3001/task/remove" -Method Post `
                -ContentType "application/json" `
                -Body "{`"uuid`":`"$($t.uuid)`"}" -ErrorAction SilentlyContinue | Out-Null
        }
        Write-Host "  Removed $($resp.Count) NodeODM task(s) from container" -ForegroundColor DarkCyan
    } catch {
        Write-Host "  Could not query NodeODM task list: $_" -ForegroundColor DarkGray
    }
}

function Invoke-PostRunCleanup {
    Write-Host "`n  [Cleanup] Removing NodeODM task data..." -ForegroundColor DarkCyan
    Remove-NodeODMTasks

    Write-Host "  [Cleanup] Docker prune (containers + build cache)..." -ForegroundColor DarkCyan
    wsl docker container prune -f 2>&1 | Out-Null

    Write-Host "  [Cleanup] Clearing Windows TEMP..." -ForegroundColor DarkCyan
    Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue

    $freeC = Get-FreeGB 'C'
    $freeD = Get-FreeGB 'D'
    Write-Host "  [Cleanup] Done. C: ${freeC}GB free | D: ${freeD}GB free" -ForegroundColor DarkCyan
}

# ---- Skip runs whose output tif already exists ----
$pendingRuns = @()
foreach ($r in $runs) {
    $outTif = Join-Path $r.outRoot "$($r.name)\odm_orthophoto.tif"
    if (Test-Path $outTif) {
        Write-Host "SKIP $($r.name) - orthophoto already exists" -ForegroundColor DarkGray
    } else {
        $pendingRuns += $r
    }
}

if ($pendingRuns.Count -eq 0) {
    Write-Host "`nAll orthomosaics already exist. Nothing to do." -ForegroundColor Green
    exit 0
}

Write-Host "`n$($pendingRuns.Count) run(s) to process: $($pendingRuns.name -join ', ')" -ForegroundColor Cyan

# ---- Main loop ----
$first     = $true
$completed = [System.Collections.Generic.List[string]]::new()
$failed    = [System.Collections.Generic.List[string]]::new()

foreach ($r in $pendingRuns) {
    $freeC = Get-FreeGB 'C'
    $freeD = Get-FreeGB 'D'
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host "  C: ${freeC}GB free | D: ${freeD}GB free" -ForegroundColor Yellow

    if ($freeC -lt 20) {
        Write-Host "ABORT: C: drive has only ${freeC}GB free (need >= 20GB). Stopping." -ForegroundColor Red
        break
    }
    if ($freeD -lt 10) {
        Write-Host "ABORT: D: drive has only ${freeD}GB free (need >= 10GB). Stopping." -ForegroundColor Red
        break
    }

    Write-Host "=== Starting FAST 2cm: $($r.name) ===" -ForegroundColor Cyan

    if ($first) {
        & $makeOm -ImageFolder $r.img -OutputRoot $r.outRoot -DatasetName $r.name -Mode fast
        $first = $false
    } else {
        & $makeOm -ImageFolder $r.img -OutputRoot $r.outRoot -DatasetName $r.name -Mode fast -NoResetNode
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "=== FAILED: $($r.name) (exit $LASTEXITCODE) ===" -ForegroundColor Red
        $failed.Add($r.name)
    } else {
        Write-Host "=== DONE: $($r.name) ===" -ForegroundColor Green
        $completed.Add($r.name)
    }

    Invoke-PostRunCleanup
}

# ---- Summary ----
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "BATCH COMPLETE" -ForegroundColor Cyan
Write-Host "  Completed ($($completed.Count)): $($completed -join ', ')" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "  Failed    ($($failed.Count)): $($failed -join ', ')" -ForegroundColor Red
}
Write-Host "================================================================" -ForegroundColor Cyan
