# run_batch_parallel.ps1 - queue all 6 remaining OM jobs to NodeODM
param([switch]$NoReset)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$datasets = @(
    @{ img = "D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT\15_04_26";  name = "SIT_15_04_26"  },
    @{ img = "D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT\09_05_26";  name = "SIT_09_05_26"  },
    @{ img = "E:\Sanjay_van_data\raw_data\images_raw\spot_1\260510";   name = "spot_1_260510" },
    @{ img = "E:\Sanjay_van_data\raw_data\images_raw\spot_2\260510";   name = "spot_2_260510" },
    @{ img = "E:\Sanjay_van_data\raw_data\images_raw\spot_3\260510";   name = "spot_3_260510" },
    @{ img = "E:\Sanjay_van_data\raw_data\images_raw\spot_4\260510";   name = "spot_4_260510" }
)
if (-not $NoReset) {
    Write-Host "Resetting nodeodm-gpu..." -ForegroundColor Yellow
    & wsl.exe docker rm -f nodeodm-gpu 2>$null
    & wsl.exe docker run -d --name nodeodm-gpu -p 3001:3000 --gpus all opendronemap/nodeodm:gpu
    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline) {
        try { $chk = Invoke-RestMethod -Uri "http://127.0.0.1:3001/info" -ErrorAction Stop
              if ($chk.version) { Write-Host "NodeODM ready." -ForegroundColor Green; break } } catch {}
        Start-Sleep -Seconds 3
    }
}
$make = "D:\Drone_Phenology_Monitoring\make_om.ps1"
$jobs = [System.Collections.Generic.List[object]]::new()
foreach ($r in $datasets) {
    $j = Start-Job -Name $r.name -ScriptBlock {
        param($img, $name, $s)
        & powershell -NoProfile -ExecutionPolicy Bypass -File $s -ImageFolder $img -DatasetName $name -NoResetNode
        exit $LASTEXITCODE
    } -ArgumentList $r.img, $r.name, $make
    $jobs.Add($j)
    Write-Host ("  Queued job {0}: {1}" -f $j.Id, $r.name) -ForegroundColor Cyan
}
Write-Host ("All {0} jobs running. Monitoring..." -f $jobs.Count) -ForegroundColor Green
$seen = [System.Collections.Generic.HashSet[string]]::new()
while ($true) {
    Start-Sleep -Seconds 60
    foreach ($j in $jobs) {
        foreach ($line in ($j | Receive-Job -Keep 2>&1)) {
            if ($seen.Add("$($j.Name)|$line")) { Write-Host ("[{0}] {1}" -f $j.Name, $line) }
        }
    }
    $run  = @($jobs | Where-Object { $_.State -eq "Running"   }).Count
    $done = @($jobs | Where-Object { $_.State -eq "Completed" }).Count
    $fail = @($jobs | Where-Object { $_.State -eq "Failed"    }).Count
    Write-Host ("{0}  running={1} done={2} failed={3}" -f (Get-Date -Format "HH:mm:ss"),$run,$done,$fail) -ForegroundColor DarkCyan
    if ($run -eq 0) { break }
}
Write-Host "`n=== RESULTS ===" -ForegroundColor White
foreach ($j in $jobs) {
    if ($j.State -eq "Completed") {
        Write-Host ("[OK]   {0}" -f $j.Name) -ForegroundColor Green
    } else {
        Write-Host ("[FAIL] {0} [{1}]" -f $j.Name, $j.State) -ForegroundColor Red
        $j | Receive-Job 2>&1 | ForEach-Object { Write-Host ("       $_") -ForegroundColor DarkRed }
    }
}
$paths = @(
    "D:\Gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\SIT_15_04_26\odm_orthophoto.tif",
    "D:\Gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\SIT_09_05_26\odm_orthophoto.tif",
    "E:\Sanjay_van_data\Processed\ODM\spot_1_260510\odm_orthophoto.tif",
    "E:\Sanjay_van_data\Processed\ODM\spot_2_260510\odm_orthophoto.tif",
    "E:\Sanjay_van_data\Processed\ODM\spot_3_260510\odm_orthophoto.tif",
    "E:\Sanjay_van_data\Processed\ODM\spot_4_260510\odm_orthophoto.tif"
)
foreach ($p in $paths) {
    $tag = if (Test-Path -LiteralPath $p) { "[OK]" } else { "[MISSING]" }
    Write-Host ("  {0}  {1}" -f $tag, $p)
}
