param(
    [int]$Steps = 320000,
    [int[]]$Seeds = @(1, 2, 3)
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

foreach ($Seed in $Seeds) {
    $Experiment = "v6_hap_lb_probe_seed$Seed"
    $LogPath = Join-Path $LogDir "$Experiment.log"

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

    "[$(Get-Date -Format o)] Starting $Experiment ($Steps steps)" |
        Tee-Object -FilePath $LogPath

    & $Python @TrainArgs 2>&1 |
        Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE

    "[$(Get-Date -Format o)] Finished $Experiment with exit code $ExitCode" |
        Tee-Object -FilePath $LogPath -Append

    if ($ExitCode -ne 0) {
        throw "$Experiment failed with exit code $ExitCode"
    }
}
