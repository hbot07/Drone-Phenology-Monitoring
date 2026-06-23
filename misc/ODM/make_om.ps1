[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageFolder,

    [string]$OutputRoot,

    [string]$DatasetName,

    [ValidateSet("fast", "full")]
    [string]$Mode = "full",

    [switch]$PrepareExifFromXmp,

    [string]$PreparedImageFolder,

    [switch]$NoResetNode
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

function Get-DefaultOutputRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResolvedImageFolder,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    if ($ResolvedImageFolder -match '^(?<prefix>[A-Za-z]:\\.*?\\)Raw_data\\') {
        return Join-Path $Matches.prefix "Processed\ODM"
    }

    return Join-Path $ProjectRoot "ODM_Output"
}

function Find-RepositoryRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StartDirectory
    )

    $current = Resolve-Path -LiteralPath $StartDirectory
    while ($null -ne $current) {
        $candidate = Join-Path $current.Path "src\utility\xmp_to_exif.py"
        if (Test-Path -LiteralPath $candidate) {
            return $current.Path
        }

        $parent = Split-Path -Parent $current.Path
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current.Path) {
            break
        }
        $current = Resolve-Path -LiteralPath $parent
    }

    throw "Could not locate repository root from: $StartDirectory"
}

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Find-RepositoryRoot -StartDirectory $scriptDir
$resolvedImageFolder = Resolve-NormalizedPath -Path $ImageFolder

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Get-DefaultOutputRoot -ResolvedImageFolder $resolvedImageFolder -ProjectRoot $scriptDir
}
$resolvedOutputRoot = Resolve-NormalizedPath -Path $OutputRoot -CreateDirectory

if ([string]::IsNullOrWhiteSpace($DatasetName)) {
    $DatasetName = Split-Path -Leaf $resolvedImageFolder
}

$workingImageFolder = $resolvedImageFolder

if ($PrepareExifFromXmp) {
    if ([string]::IsNullOrWhiteSpace($PreparedImageFolder)) {
        $PreparedImageFolder = Join-Path (Split-Path -Parent $resolvedImageFolder) ((Split-Path -Leaf $resolvedImageFolder) + "_exif")
    }

    $xmpScript = Join-Path $repoRoot "src\utility\xmp_to_exif.py"
    if (-not (Test-Path -LiteralPath $xmpScript)) {
        throw "Could not find XMP to EXIF conversion script: $xmpScript"
    }

    Write-Host "Preparing EXIF GPS copy in: $PreparedImageFolder"
    & python $xmpScript $resolvedImageFolder $PreparedImageFolder
    if ($LASTEXITCODE -ne 0) {
        throw "XMP to EXIF conversion failed."
    }

    $workingImageFolder = Resolve-NormalizedPath -Path $PreparedImageFolder
}

$odmArgs = switch ($Mode) {
    "fast" {
        @(
            "--fast-orthophoto",
            "--skip-3dmodel",
            "--skip-report",
            "--build-overviews",
            "--orthophoto-compression",
            "DEFLATE",
            "--orthophoto-resolution",
            "2"
        )
    }
    "full" {
        @(
            "--skip-report",
            "--build-overviews",
            "--orthophoto-compression",
            "DEFLATE",
            "--orthophoto-resolution",
            "2"
        )
    }
    default {
        @(
            "--skip-report",
            "--build-overviews",
            "--orthophoto-compression",
            "DEFLATE",
            "--orthophoto-resolution",
            "2"
        )
    }
}

$runner = Join-Path $scriptDir "run_nodeodm_orthomosaic.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Could not find NodeODM orthomosaic runner: $runner"
}

Write-Host "Image folder : $workingImageFolder"
Write-Host "Output root  : $resolvedOutputRoot"
Write-Host "Dataset name : $DatasetName"
Write-Host "Mode         : $Mode"

$runParams = @{
    ImageFolder = $workingImageFolder
    OutputRoot = $resolvedOutputRoot
    DatasetName = $DatasetName
    Wait = $true
    OdmArgs = $odmArgs
}

if (-not $NoResetNode) {
    $runParams.ResetNode = $true
}

& $runner @runParams
if ($LASTEXITCODE -ne 0) {
    throw "Orthomosaic generation failed."
}
