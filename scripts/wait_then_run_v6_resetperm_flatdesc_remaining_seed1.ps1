param(
    [string[]]$WaitExperiments = @(
        "resetperm806_flatdesc_dim16_flatcrit_seed1",
        "resetperm806_flatdesc_dim32_flatcrit_seed1"
    ),
    [int[]]$DescriptorDims = @(64, 128),
    [int]$Steps = 806400,
    [int]$Seed = 1,
    [int]$MaxParallel = 2,
    [string]$ExperimentPrefix = "resetperm806_flatdesc"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot "training_logs"
$SweepScript = Join-Path $PSScriptRoot `
    "run_v6_resetperm_flat_descriptor_dim_sweep_seed1.ps1"

foreach ($Experiment in $WaitExperiments) {
    $PidPath = Join-Path $LogDir "$Experiment.wrapper.pid"
    while (-not (Test-Path -LiteralPath $PidPath)) {
        Start-Sleep -Seconds 30
    }
    $ProcessId = [int](Get-Content -LiteralPath $PidPath -Raw)
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 300
    }
}

& $SweepScript `
    -Steps $Steps `
    -Seed $Seed `
    -DescriptorDims $DescriptorDims `
    -MaxParallel $MaxParallel `
    -ExperimentPrefix $ExperimentPrefix
