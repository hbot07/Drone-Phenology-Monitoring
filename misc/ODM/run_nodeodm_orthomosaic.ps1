[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageFolder,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string]$DatasetName,

    [string]$NodeContainerName = "nodeodm-gpu",

    [int]$NodePort = 3001,

    [string]$NodeImage = "opendronemap/nodeodm:gpu",

    [switch]$ResetNode,

    [switch]$Wait,

    [int]$PollSeconds = 30,

    [switch]$SkipCopy,

    [string[]]$OdmArgs = @(
        "--skip-report",
        "--build-overviews",
        "--orthophoto-compression",
        "DEFLATE",
        "--orthophoto-resolution",
        "2"
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

function Convert-OdmArgsToOptionsJson {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $options = New-Object System.Collections.Generic.List[object]
    $index = 0

    while ($index -lt $Arguments.Count) {
        $arg = $Arguments[$index]
        if (-not $arg.StartsWith("--")) {
            throw "Expected an ODM option name starting with --, got '$arg'"
        }

        $name = $arg.Substring(2)
        $value = $true

        if (($index + 1) -lt $Arguments.Count -and -not $Arguments[$index + 1].StartsWith("--")) {
            $raw = $Arguments[$index + 1]
            $numericVal = 0.0
            if ([double]::TryParse($raw, [System.Globalization.NumberStyles]::Number,
                    [System.Globalization.CultureInfo]::InvariantCulture, [ref]$numericVal)) {
                $value = if ($numericVal -eq [math]::Truncate($numericVal)) { [int]$numericVal } else { $numericVal }
            } else {
                $value = $raw
            }
            $index += 1
        }

        $options.Add([pscustomobject]@{
            name = $name
            value = $value
        })

        $index += 1
    }

    return ($options | ConvertTo-Json -Compress)
}

function Ensure-NodeRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ContainerName,
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [Parameter(Mandatory = $true)]
        [string]$Image,
        [switch]$Reset
    )

    if ($Reset) {
        Invoke-WslCommand -Arguments @("docker", "rm", "-f", $ContainerName) -AllowFailure
    }

    $containerQuery = & wsl.exe docker ps -aq --filter "name=^${ContainerName}$"
    $containerId = if ($null -eq $containerQuery) { "" } else { ($containerQuery | Out-String).Trim() }
    if ([string]::IsNullOrWhiteSpace($containerId)) {
        $runArgs = @("docker", "run", "-d", "--name", $ContainerName, "-p", "$Port`:3000")
        if ($Image -match ':(gpu|gpu\.|.*gpu.*)$') {
            $runArgs += @("--gpus", "all")
        }
        $runArgs += $Image
        Invoke-WslCommand -Arguments $runArgs
    } else {
        $runningQuery = & wsl.exe docker inspect -f "{{.State.Running}}" $ContainerName
        $isRunning = if ($null -eq $runningQuery) { "" } else { ($runningQuery | Out-String).Trim() }
        if ($isRunning -ne "true") {
            Invoke-WslCommand -Arguments @("docker", "start", $ContainerName)
        }
    }

    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline) {
        try {
            $info = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/info"
            if ($info.version) {
                return
            }
        } catch {
        }

        Start-Sleep -Seconds 2
    }

    throw "NodeODM did not become ready on port $Port"
}

function Copy-OrthophotoFromContainer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ContainerName,
        [Parameter(Mandatory = $true)]
        [string]$TaskUuid,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    $wslDestination = Convert-ToWslPath -WindowsPath $DestinationPath
    Invoke-WslCommand -Arguments @(
        "docker", "cp",
        "${ContainerName}:/var/www/data/$TaskUuid/odm_orthophoto/odm_orthophoto.tif",
        $wslDestination
    )
}

$resolvedImageFolder = Resolve-NormalizedPath -Path $ImageFolder
$resolvedOutputRoot = Resolve-NormalizedPath -Path $OutputRoot -CreateDirectory

$images = @(Get-ChildItem -LiteralPath $resolvedImageFolder -File |
    Where-Object { $_.Extension -match '^(?i)\.(jpg|jpeg|tif|tiff|png)$' } |
    Sort-Object Name)

if ($images.Count -eq 0) {
    throw "No supported image files were found in $resolvedImageFolder"
}

if ([string]::IsNullOrWhiteSpace($DatasetName)) {
    $DatasetName = Split-Path -Leaf $resolvedImageFolder
}

$safeDatasetName = Get-SafeDatasetName -Name $DatasetName
$taskOutputDir = Join-Path $resolvedOutputRoot $safeDatasetName
if (-not (Test-Path -LiteralPath $taskOutputDir)) {
    New-Item -ItemType Directory -Path $taskOutputDir -Force | Out-Null
}

$orthophotoPath = Join-Path $taskOutputDir "odm_orthophoto.tif"
$optionsJson = Convert-OdmArgsToOptionsJson -Arguments $OdmArgs
Write-Host "Options JSON   : $optionsJson"

Ensure-NodeRunning -ContainerName $NodeContainerName -Port $NodePort -Image $NodeImage -Reset:$ResetNode

Add-Type -AssemblyName System.Net.Http
$client = [System.Net.Http.HttpClient]::new()

$initForm = [System.Net.Http.MultipartFormDataContent]::new()
$initForm.Add([System.Net.Http.StringContent]::new($safeDatasetName), 'name')
$initForm.Add([System.Net.Http.StringContent]::new($optionsJson), 'options')

$initResponse = $client.PostAsync("http://127.0.0.1:$NodePort/task/new/init", $initForm).Result
$initBody = $initResponse.Content.ReadAsStringAsync().Result
if (-not $initResponse.IsSuccessStatusCode) {
    throw "Task initialization failed: $initBody"
}

$taskUuid = ((ConvertFrom-Json $initBody).uuid)
if ([string]::IsNullOrWhiteSpace($taskUuid)) {
    throw "Task initialization did not return a UUID. Response: $initBody"
}

Write-Host "Task UUID      : $taskUuid"
Write-Host "Image count    : $($images.Count)"
Write-Host "Node container : $NodeContainerName"
Write-Host "Node port      : $NodePort"

$index = 0
foreach ($image in $images) {
    $index += 1
    $uploadForm = [System.Net.Http.MultipartFormDataContent]::new()
    $stream = [System.IO.File]::OpenRead($image.FullName)

    try {
        $content = [System.Net.Http.StreamContent]::new($stream)
        $uploadForm.Add($content, 'images', $image.Name)

        $uploadResponse = $client.PostAsync("http://127.0.0.1:$NodePort/task/new/upload/$taskUuid", $uploadForm).Result
        $uploadBody = $uploadResponse.Content.ReadAsStringAsync().Result
        if (-not $uploadResponse.IsSuccessStatusCode) {
            throw "Upload failed for $($image.Name): $uploadBody"
        }

        if (($index % 10) -eq 0 -or $index -eq $images.Count) {
            Write-Host ("Uploaded {0}/{1}: {2}" -f $index, $images.Count, $image.Name)
        }
    } finally {
        $stream.Dispose()
        $uploadForm.Dispose()
    }
}

$commitResponse = $client.PostAsync("http://127.0.0.1:$NodePort/task/new/commit/$taskUuid", $null).Result
$commitBody = $commitResponse.Content.ReadAsStringAsync().Result
if (-not $commitResponse.IsSuccessStatusCode) {
    throw "Task commit failed: $commitBody"
}

Write-Host "Task committed : $taskUuid"
Write-Host "Task info URL  : http://127.0.0.1:$NodePort/task/$taskUuid/info"

if (-not $Wait) {
    exit 0
}

if ($PollSeconds -lt 5) {
    throw "PollSeconds should be at least 5 seconds."
}

$consecutiveErrors = 0
while ($true) {
    try {
        $info = Invoke-RestMethod -Uri "http://127.0.0.1:$NodePort/task/$taskUuid/info?with_output=0" -ErrorAction Stop
        $consecutiveErrors = 0
    } catch {
        $consecutiveErrors++
        Write-Host "Poll error ($consecutiveErrors/5): $($_.Exception.Message)"
        if ($consecutiveErrors -ge 5) {
            throw "Task polling failed after 5 consecutive errors - container may have been reset."
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if ($null -eq $info) {
        $consecutiveErrors++
        Write-Host "Null response ($consecutiveErrors/5) - retrying..."
        if ($consecutiveErrors -ge 5) {
            throw "Task polling failed after 5 consecutive empty responses - container may have been reset."
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if (-not ($info.PSObject.Properties['status']) -or $null -eq $info.status) {
        $consecutiveErrors++
        Write-Host "Missing status ($consecutiveErrors/5) - retrying..."
        if ($consecutiveErrors -ge 5) {
            throw "Task polling failed after 5 consecutive missing-status responses - container may have been reset."
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $statusCode = [int]$info.status.code
    $progress = [double]$info.progress

    Write-Host ("Status {0}, progress {1:N1}%%" -f $statusCode, $progress)

    if ($statusCode -eq 40) {
        if (-not $SkipCopy) {
            Copy-OrthophotoFromContainer -ContainerName $NodeContainerName -TaskUuid $taskUuid -DestinationPath $orthophotoPath
            Write-Host "Orthophoto copied to: $orthophotoPath"
        }

        exit 0
    }

    if ($statusCode -eq 30) {
        $errorMessage = $info.status.errorMessage
        throw "Task failed: $errorMessage"
    }

    if ($statusCode -eq 50) {
        throw "Task was canceled."
    }

    Start-Sleep -Seconds $PollSeconds
}