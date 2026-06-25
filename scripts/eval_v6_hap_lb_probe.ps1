param(
    [int]$WaitForPid = 0,
    [int[]]$Seeds = @(1, 2, 3),
    [int]$Episodes = 24,
    [int]$EvalSeed = 1000,
    [string]$ExperimentPrefix = "v6_hap_lb_probe_seed"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$ResultsRoot = Join-Path $RepoRoot "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"

if ($WaitForPid -gt 0) {
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}

Set-Location -LiteralPath $RepoRoot

foreach ($Seed in $Seeds) {
    $Experiment = "$ExperimentPrefix$Seed"
    $ExperimentDir = Join-Path $ResultsRoot $Experiment
    $RunDir = Get-ChildItem -LiteralPath $ExperimentDir -Directory |
        Where-Object { $_.Name -match "^run\d+$" } |
        Sort-Object { [int]$_.Name.Substring(3) } |
        Select-Object -Last 1

    if ($null -eq $RunDir) {
        throw "No completed run directory found for $Experiment"
    }

    $ModelDir = Join-Path $RunDir.FullName "models"
    if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "actor.pt"))) {
        throw "No actor checkpoint found in $ModelDir"
    }

    $EvalLog = Join-Path $LogDir "$Experiment.eval.log"
    & $Python -u -m onpolicy.scripts.eval.eval_mec `
        --env_name MEC `
        --mec_eval_controller policy `
        --model_dir $ModelDir `
        --seed $EvalSeed `
        --mec_eval_episodes $Episodes 2>&1 |
        Tee-Object -FilePath $EvalLog

    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $Experiment"
    }
}

$HeuristicLog = Join-Path $LogDir "v6_hap_loadbearing_heuristic.eval.log"
& $Python -u -m onpolicy.scripts.eval.eval_mec `
    --env_name MEC `
    --mec_scenario v6_hap_loadbearing `
    --mec_eval_controller heuristic `
    --seed $EvalSeed `
    --mec_eval_episodes $Episodes 2>&1 |
    Tee-Object -FilePath $HeuristicLog

if ($LASTEXITCODE -ne 0) {
    throw "Heuristic evaluation failed"
}
