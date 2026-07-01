param(
    [int]$Steps = 806400,
    [int]$Seed = 1,
    [int[]]$DescriptorDims = @(16, 32, 64, 128),
    [int]$MaxParallel = 2,
    [int]$StaggerSeconds = 180,
    [string]$ExperimentPrefix = "resetperm806_flatdesc"
)

$ErrorActionPreference = "Stop"

if ($MaxParallel -lt 1 -or $MaxParallel -gt 4) {
    throw "MaxParallel must be between 1 and 4"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_v6_resetperm_flat_descriptor_seed1_1500k.ps1"
$LogDir = Join-Path $RepoRoot "training_logs"
$ManifestPath = Join-Path $LogDir "$ExperimentPrefix.seed$Seed.manifest.json"

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "runner script missing: $Runner"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $RepoRoot

$Jobs = @(
    $DescriptorDims | ForEach-Object {
        [pscustomobject]@{
            Dim = [int]$_
            Experiment = "${ExperimentPrefix}_dim${_}_flatcrit_seed${Seed}"
        }
    }
)

[ordered]@{
    purpose = "reset-permutation flat-descriptor dimension sweep"
    seed = $Seed
    steps = $Steps
    descriptor_dims = $DescriptorDims
    max_parallel = $MaxParallel
    uses_runner = $Runner
    experiments = @($Jobs | ForEach-Object { $_.Experiment })
    started_at = (Get-Date -Format o)
} | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $ManifestPath -Encoding UTF8

$Queue = [System.Collections.Queue]::new()
foreach ($Job in $Jobs) {
    $Queue.Enqueue($Job)
}
$Running = @()

function Start-DimRun {
    param([pscustomobject]$Job)

    $WrapperOut = Join-Path $LogDir "$($Job.Experiment).wrapper.log"
    $WrapperErr = Join-Path $LogDir "$($Job.Experiment).wrapper.err.log"
    Remove-Item -LiteralPath $WrapperOut, $WrapperErr -Force `
        -ErrorAction SilentlyContinue

    $Args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Runner,
        "-Seed", "$Seed",
        "-Steps", "$Steps",
        "-DescriptorDim", "$($Job.Dim)",
        "-ExperimentPrefix", "${ExperimentPrefix}_dim$($Job.Dim)_flatcrit"
    )
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $Args `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $WrapperOut `
        -RedirectStandardError $WrapperErr `
        -PassThru
    Set-Content -LiteralPath (Join-Path $LogDir "$($Job.Experiment).wrapper.pid") `
        -Value $Process.Id
    Write-Host "[$(Get-Date -Format o)] started $($Job.Experiment) pid=$($Process.Id)"
    return [pscustomobject]@{
        Job = $Job
        Process = $Process
    }
}

while ($Queue.Count -gt 0 -or $Running.Count -gt 0) {
    while ($Queue.Count -gt 0 -and $Running.Count -lt $MaxParallel) {
    $Running += Start-DimRun -Job $Queue.Dequeue()
        if (
            $StaggerSeconds -gt 0 -and
            $Queue.Count -gt 0 -and
            $Running.Count -lt $MaxParallel
        ) {
            Start-Sleep -Seconds $StaggerSeconds
        }
    }

    Start-Sleep -Seconds 60

    $StillRunning = @()
    foreach ($Item in $Running) {
        $Item.Process.Refresh()
        if ($Item.Process.HasExited) {
            if ($Item.Process.ExitCode -ne 0) {
                throw "$($Item.Job.Experiment) failed with exit code $($Item.Process.ExitCode)"
            }
            Write-Output "[$(Get-Date -Format o)] completed $($Item.Job.Experiment)"
        } else {
            $StillRunning += $Item
        }
    }
    $Running = $StillRunning
}

Write-Output "[$(Get-Date -Format o)] descriptor dimension sweep complete"
