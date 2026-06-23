[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunsCsv,

    [string]$MakeOmPath,

    [ValidateSet("fast", "full")]
    [string]$DefaultMode = "full",

    [switch]$NoSkipExisting,

    [string[]]$MinFreeGB = @(),

    [switch]$CleanupNodeTasks,

    [switch]$DockerPrune,

    [switch]$ClearTemp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-NormalizedPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [switch]$CreateDirectory
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        if (-not $CreateDirectory) {
            throw "Path does not exist: $Path"
        }
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Convert-ToBool {
    param(
        [object]$Value,
        [bool]$Default = $false
    )

    if ($null -eq $Value) {
        return $Default
    }

    $text = "$Value".Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $Default
    }

    return @("1", "true", "yes", "y") -contains $text.ToLowerInvariant()
}

function Get-FreeGB {
    param([string]$Drive)
    $driveInfo = Get-PSDrive $Drive -ErrorAction SilentlyContinue
    if ($null -eq $driveInfo) {
        throw "Drive not found: $Drive"
    }
    return [math]::Round($driveInfo.Free / 1GB, 1)
}

function Convert-MinFreeSpec {
    param([string[]]$Specs)

    $parsed = @{}
    foreach ($spec in $Specs) {
        if ([string]::IsNullOrWhiteSpace($spec)) {
            continue
        }
        if ($spec -notmatch '^(?<drive>[A-Za-z]):?=(?<gb>[0-9]+(?:\.[0-9]+)?)$') {
            throw "Invalid -MinFreeGB entry '$spec'. Use DRIVE=GB, for example C=20."
        }
        $parsed[$Matches.drive.ToUpperInvariant()] = [double]$Matches.gb
    }
    return $parsed
}

function Assert-MinFreeSpace {
    param([hashtable]$MinFree)

    foreach ($drive in $MinFree.Keys) {
        $free = Get-FreeGB -Drive $drive
        $required = [double]$MinFree[$drive]
        Write-Host "  $drive`: ${free}GB free (required: ${required}GB)"
        if ($free -lt $required) {
            throw "$drive`: has only ${free}GB free; required >= ${required}GB."
        }
    }
}

function Remove-NodeODMTasks {
    param([int]$NodePort = 3001)

    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$NodePort/task/list" -Method Get -ErrorAction Stop
        foreach ($task in $resp) {
            Invoke-RestMethod -Uri "http://127.0.0.1:$NodePort/task/remove" -Method Post `
                -ContentType "application/json" `
                -Body "{`"uuid`":`"$($task.uuid)`"}" `
                -ErrorAction SilentlyContinue | Out-Null
        }
        Write-Host "  Removed $($resp.Count) NodeODM task(s)."
    } catch {
        Write-Host "  Could not query NodeODM task list: $_" -ForegroundColor DarkGray
    }
}

function Invoke-OptionalCleanup {
    param([int]$NodePort = 3001)

    if ($CleanupNodeTasks) {
        Write-Host "  [Cleanup] Removing NodeODM task data..."
        Remove-NodeODMTasks -NodePort $NodePort
    }

    if ($DockerPrune) {
        Write-Host "  [Cleanup] Docker container/build-cache prune..."
        & wsl docker container prune -f 2>&1 | Out-Null
        & wsl docker builder prune -f 2>&1 | Out-Null
    }

    if ($ClearTemp) {
        Write-Host "  [Cleanup] Clearing Windows TEMP..."
        Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$scriptDir = Split-Path -Parent $PSCommandPath
if ([string]::IsNullOrWhiteSpace($MakeOmPath)) {
    $MakeOmPath = Join-Path $scriptDir "make_om.ps1"
}

$resolvedRunsCsv = Resolve-NormalizedPath -Path $RunsCsv
$resolvedMakeOm = Resolve-NormalizedPath -Path $MakeOmPath
$minFreeByDrive = Convert-MinFreeSpec -Specs $MinFreeGB

$runs = @(Import-Csv -LiteralPath $resolvedRunsCsv)
if ($runs.Count -eq 0) {
    throw "No rows found in CSV: $resolvedRunsCsv"
}

foreach ($requiredColumn in @("image_folder", "dataset_name")) {
    if (-not ($runs[0].PSObject.Properties.Name -contains $requiredColumn)) {
        throw "CSV must contain a '$requiredColumn' column."
    }
}

$completed = [System.Collections.Generic.List[string]]::new()
$failed = [System.Collections.Generic.List[string]]::new()
$firstRun = $true

Write-Host "Batch CSV : $resolvedRunsCsv"
Write-Host "make_om   : $resolvedMakeOm"
Write-Host "Run count : $($runs.Count)"

foreach ($run in $runs) {
    $datasetName = "$($run.dataset_name)".Trim()
    $imageFolder = "$($run.image_folder)".Trim()
    $outputRoot = if ($run.PSObject.Properties.Name -contains "output_root") { "$($run.output_root)".Trim() } else { "" }
    $mode = if ($run.PSObject.Properties.Name -contains "mode" -and -not [string]::IsNullOrWhiteSpace("$($run.mode)")) { "$($run.mode)".Trim() } else { $DefaultMode }
    $prepareExif = if ($run.PSObject.Properties.Name -contains "prepare_exif_from_xmp") { Convert-ToBool $run.prepare_exif_from_xmp } else { $false }
    $preparedImageFolder = if ($run.PSObject.Properties.Name -contains "prepared_image_folder") { "$($run.prepared_image_folder)".Trim() } else { "" }

    if ([string]::IsNullOrWhiteSpace($datasetName)) {
        throw "CSV row has an empty dataset_name."
    }
    if ([string]::IsNullOrWhiteSpace($imageFolder)) {
        throw "CSV row '$datasetName' has an empty image_folder."
    }

    $resolvedImageFolder = Resolve-NormalizedPath -Path $imageFolder
    $resolvedOutputRoot = ""
    if (-not [string]::IsNullOrWhiteSpace($outputRoot)) {
        $resolvedOutputRoot = Resolve-NormalizedPath -Path $outputRoot -CreateDirectory
    }

    $skipExistingForRow = -not $NoSkipExisting
    if ($run.PSObject.Properties.Name -contains "skip_existing") {
        $skipExistingForRow = Convert-ToBool $run.skip_existing -Default $skipExistingForRow
    }

    if ($skipExistingForRow -and -not [string]::IsNullOrWhiteSpace($resolvedOutputRoot)) {
        $outTif = Join-Path $resolvedOutputRoot "$datasetName\odm_orthophoto.tif"
        if (Test-Path -LiteralPath $outTif) {
            Write-Host "SKIP $datasetName - orthophoto already exists: $outTif" -ForegroundColor DarkGray
            continue
        }
    }

    Write-Host ""
    Write-Host "================================================================"
    Write-Host "Starting: $datasetName"
    Write-Host "Images  : $resolvedImageFolder"
    if (-not [string]::IsNullOrWhiteSpace($resolvedOutputRoot)) {
        Write-Host "Output  : $resolvedOutputRoot"
    }
    Write-Host "Mode    : $mode"

    try {
        Assert-MinFreeSpace -MinFree $minFreeByDrive

        $params = @{
            ImageFolder = $resolvedImageFolder
            DatasetName = $datasetName
            Mode = $mode
        }
        if (-not [string]::IsNullOrWhiteSpace($resolvedOutputRoot)) {
            $params.OutputRoot = $resolvedOutputRoot
        }
        if ($prepareExif) {
            $params.PrepareExifFromXmp = $true
        }
        if (-not [string]::IsNullOrWhiteSpace($preparedImageFolder)) {
            $params.PreparedImageFolder = $preparedImageFolder
        }
        if (-not $firstRun) {
            $params.NoResetNode = $true
        }

        & $resolvedMakeOm @params
        if ($LASTEXITCODE -ne 0) {
            throw "make_om.ps1 exited with code $LASTEXITCODE"
        }

        $completed.Add($datasetName)
        $firstRun = $false
        Write-Host "DONE: $datasetName" -ForegroundColor Green
    } catch {
        $failed.Add($datasetName)
        Write-Host "FAILED: $datasetName - $_" -ForegroundColor Red
    } finally {
        Invoke-OptionalCleanup
    }
}

Write-Host ""
Write-Host "================================================================"
Write-Host "BATCH COMPLETE"
Write-Host "Completed ($($completed.Count)): $($completed -join ', ')" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host "Failed    ($($failed.Count)): $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
