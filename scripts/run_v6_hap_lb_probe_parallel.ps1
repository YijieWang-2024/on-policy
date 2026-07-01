param(
    [int]$Steps = 512000,
    [int[]]$Seeds = @(1, 2, 3),
    [string]$ExperimentPrefix = "v6_hap_lb_diag160_seed"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "marl environment Python was not found at $Python"
}

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
        "--log_interval", "5",
        "--save_interval", "25",
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

    Set-Content -LiteralPath (Join-Path $LogDir "$Experiment.pid") -Value $Process.Id
    Write-Output "[$(Get-Date -Format o)] Started $Experiment PID=$($Process.Id)"
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
    $ExitCode = $Run.Process.ExitCode
    Write-Output "[$(Get-Date -Format o)] Finished $($Run.Experiment) exit=$ExitCode"
    if ($ExitCode -ne 0) {
        $Failed += $Run.Experiment
    }
}

if ($Failed.Count -gt 0) {
    throw "Training failed: $($Failed -join ', ')"
}

& (Join-Path $PSScriptRoot "eval_v6_hap_lb_probe.ps1") `
    -ExperimentPrefix $ExperimentPrefix `
    -Seeds $Seeds

if ($LASTEXITCODE -ne 0) {
    throw "Evaluation pipeline failed with exit code $LASTEXITCODE"
}
