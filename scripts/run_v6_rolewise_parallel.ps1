param(
    [int]$Steps = 512000,
    [int[]]$Seeds = @(1, 2, 3),
    [string]$ExperimentPrefix = "v6_hap_lb_rolewise_seed"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$ResultsRoot = Join-Path $RepoRoot (
    "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"
)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $RepoRoot

$Runs = @()
foreach ($Seed in $Seeds) {
    $Experiment = "$ExperimentPrefix$Seed"
    $StdoutPath = Join-Path $LogDir "$Experiment.log"
    $StderrPath = Join-Path $LogDir "$Experiment.err.log"
    $TrainArgs = @(
        "-u",
        "-m", "onpolicy.scripts.train.train_mec",
        "--env_name", "MEC",
        "--algorithm_name", "mappo",
        "--experiment_name", $Experiment,
        "--mec_scenario", "v6_hap_loadbearing",
        "--seed", "$Seed",
        "--n_rollout_threads", "16",
        "--n_training_threads", "2",
        "--n_eval_rollout_threads", "8",
        "--episode_length", "200",
        "--num_env_steps", "$Steps",
        "--ppo_epoch", "5",
        "--num_mini_batch", "1",
        "--hidden_size", "128",
        "--layer_N", "2",
        "--use_entropy_anneal",
        "--mec_logstd_init", "-1.9",
        "--entropy_coef", "0.003",
        "--lr", "5e-4",
        "--critic_lr", "5e-4",
        "--gamma", "0.99",
        "--mec_rolewise_loss",
        "--log_interval", "5",
        "--save_interval", "20",
        "--save_step_checkpoints",
        "--use_eval",
        "--eval_interval", "20",
        "--eval_episodes", "24",
        "--eval_seed", "1000",
        "--use_wandb"
    )
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $TrainArgs `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru
    Write-Output (
        "[$(Get-Date -Format o)] Started $Experiment PID=$($Process.Id)"
    )
    $Runs += [pscustomobject]@{
        Experiment = $Experiment
        Process = $Process
        Stdout = $StdoutPath
        Stderr = $StderrPath
    }
}

$Failed = @()
foreach ($Run in $Runs) {
    $Run.Process.WaitForExit()
    $Run.Process.Refresh()
    $ModelDir = Join-Path $ResultsRoot (
        "$($Run.Experiment)\run1\models"
    )
    $Complete = (
        (Test-Path -LiteralPath (Join-Path $ModelDir "actor.pt")) -and
        (Test-Path -LiteralPath (
            Join-Path $ModelDir "best\best_checkpoint.json"
        ))
    )
    $ErrorBytes = (Get-Item -LiteralPath $Run.Stderr).Length
    Write-Output (
        "[$(Get-Date -Format o)] Finished $($Run.Experiment) " +
        "exit=$($Run.Process.ExitCode) complete=$Complete " +
        "stderr_bytes=$ErrorBytes"
    )
    if (-not $Complete -or $ErrorBytes -gt 0) {
        $Failed += $Run.Experiment
    }
}

if ($Failed.Count -gt 0) {
    throw "Training failed: $($Failed -join ', ')"
}
