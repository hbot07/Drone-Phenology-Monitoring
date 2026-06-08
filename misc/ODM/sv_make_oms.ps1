##############################################################
# sv_make_oms.ps1
# Batch-create missing Sanjay Van orthomosaics (fast 2cm mode)
# Monitors C: drive space, cleans NodeODM tasks after each run
##############################################################

$makeOm = "D:\Drone_Phenology_Monitoring\make_om.ps1"

# ---- Runs: only dates with raw data but no existing OM ----
$runs = @(
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_1\250911'; name='spot_1_250911' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_1\251130'; name='spot_1_251130' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_1\260131'; name='spot_1_260131' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_2\250911'; name='spot_2_250911' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_2\251130'; name='spot_2_251130' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_2\260131'; name='spot_2_260131' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_3\251114'; name='spot_3_251114' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_3\251130'; name='spot_3_251130' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_3\260131'; name='spot_3_260131' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_4\251109'; name='spot_4_251109' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_4\251114'; name='spot_4_251114' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_4\251130'; name='spot_4_251130' },
    @{ img='E:\Sanjay_van_data\raw_data\images_raw\spot_4\260131'; name='spot_4_260131' }
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
        Write-Host "  Could not query NodeODM task list (node may be stopped): $_" -ForegroundColor DarkGray
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
    $freeE = Get-FreeGB 'E'
    Write-Host "  [Cleanup] Done. C: ${freeC}GB free | E: ${freeE}GB free" -ForegroundColor DarkCyan
}

# ---- Skip runs whose output already exists ----
$pendingRuns = @()
foreach ($r in $runs) {
    $outTif = "E:\Sanjay_van_data\Processed\ODM\$($r.name)\odm_orthophoto.tif"
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
$first = $true
$completed = [System.Collections.Generic.List[string]]::new()
$failed    = [System.Collections.Generic.List[string]]::new()

foreach ($r in $pendingRuns) {
    $freeC = Get-FreeGB 'C'
    $freeE = Get-FreeGB 'E'
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host "  C: ${freeC}GB free | E: ${freeE}GB free" -ForegroundColor Yellow

    if ($freeC -lt 20) {
        Write-Host "ABORT: C: drive has only ${freeC}GB free (need >= 20GB). Stopping." -ForegroundColor Red
        break
    }

    Write-Host "=== Starting FAST 2cm: $($r.name) ===" -ForegroundColor Cyan

    if ($first) {
        & $makeOm -ImageFolder $r.img -DatasetName $r.name -Mode fast
        $first = $false
    } else {
        & $makeOm -ImageFolder $r.img -DatasetName $r.name -Mode fast -NoResetNode
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
