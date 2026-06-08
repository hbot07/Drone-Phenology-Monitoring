[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageFolder,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string]$DatasetName,

    [string]$Image = "opendronemap/odm",

    [switch]$Gpu,

    [switch]$DryRun,

    [string[]]$OdmArgs = @(
        "--fast-orthophoto",
        "--skip-3dmodel",
        "--skip-report",
        "--build-overviews",
        "--orthophoto-compression",
        "DEFLATE"
    )
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

function Convert-ToWslPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath
    )

    if ($WindowsPath -match '^[A-Za-z]:\\') {
        $drive = $WindowsPath.Substring(0, 1).ToLowerInvariant()
        $suffix = $WindowsPath.Substring(2).Replace('\', '/')
        return "/mnt/$drive$suffix"
    }

    if ($WindowsPath -match '^/mnt/[a-z]/') {
        return $WindowsPath
    }

    throw "Only local drive paths are supported right now: $WindowsPath"
}

function Invoke-WslCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )

    & wsl.exe @Arguments
    if (-not $AllowFailure -and $LASTEXITCODE -ne 0) {
        throw "WSL command failed: wsl $($Arguments -join ' ')"
    }
}

function Get-SafeDatasetName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $safe = $Name -replace '[^A-Za-z0-9_-]', '_'
    $safe = $safe.Trim('_')

    if ([string]::IsNullOrWhiteSpace($safe)) {
        throw "Could not derive a valid dataset name from '$Name'."
    }

    return $safe
}

$resolvedImageFolder = Resolve-NormalizedPath -Path $ImageFolder
$resolvedOutputRoot = Resolve-NormalizedPath -Path $OutputRoot -CreateDirectory

$imageCount = @(Get-ChildItem -LiteralPath $resolvedImageFolder -File |
    Where-Object { $_.Extension -match '^(?i)\.(jpg|jpeg|tif|tiff|png)$' }).Count

if ($imageCount -eq 0) {
    throw "No supported image files were found in $resolvedImageFolder"
}

if ([string]::IsNullOrWhiteSpace($DatasetName)) {
    $DatasetName = Split-Path -Leaf $resolvedImageFolder
}

$safeDatasetName = Get-SafeDatasetName -Name $DatasetName
$datasetWindowsPath = Join-Path $resolvedOutputRoot $safeDatasetName

if (-not (Test-Path -LiteralPath $datasetWindowsPath)) {
    New-Item -ItemType Directory -Path $datasetWindowsPath -Force | Out-Null
}

$wslImageFolder = Convert-ToWslPath -WindowsPath $resolvedImageFolder
$wslDatasetPath = Convert-ToWslPath -WindowsPath $datasetWindowsPath

Invoke-WslCommand -Arguments @("docker", "version")
Invoke-WslCommand -Arguments @("mkdir", "-p", $wslDatasetPath)

$containerDatasetPath = "/datasets/$safeDatasetName"

$dockerArgs = @(
    "docker",
    "run",
    "--rm",
    "--mount", "type=bind,src=$wslDatasetPath,dst=$containerDatasetPath",
    "--mount", "type=bind,src=$wslImageFolder,dst=$containerDatasetPath/images,readonly"
)

if ($Gpu) {
    $dockerArgs += @("--gpus", "all")
}

$dockerArgs += @(
    $Image,
    "--project-path", "/datasets",
    $safeDatasetName
)

$dockerArgs += $OdmArgs

Write-Host "Image folder    : $resolvedImageFolder"
Write-Host "Output dataset  : $datasetWindowsPath"
Write-Host "Dataset name    : $safeDatasetName"
Write-Host "WSL image path  : $wslImageFolder"
Write-Host "WSL output path : $wslDatasetPath"
Write-Host "Expected output : $datasetWindowsPath\odm_orthophoto\odm_orthophoto.tif"
Write-Host ""
Write-Host "Command: wsl $($dockerArgs -join ' ')"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run only. No container was started."
    exit 0
}

Invoke-WslCommand -Arguments $dockerArgs

Write-Host ""
Write-Host "Orthomosaic finished."
Write-Host "GeoTIFF: $datasetWindowsPath\odm_orthophoto\odm_orthophoto.tif"