param(
    [int]$Seed = 1,
    [int]$Steps = 1500000,
    [int]$EpisodeLength = 350,
    [int]$RolloutThreads = 16,
    [int]$ValidationEpisodes = 24,
    [int]$ValidationSeed = 1000,
    [int]$TestEpisodes = 24,
    [int]$TestSeed = 100000,
    [int]$TestSeedStride = 13,
    [int]$DescriptorDim = 256,
    [string]$ExperimentPrefix = "resetperm1500_flat_descriptor_flatcrit"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$EvalDir = Join-Path $RepoRoot "eval_outputs\resetperm_flat_descriptor_1500k"
$ResultsRoot = Join-Path $RepoRoot (
    "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"
)
$Experiment = "${ExperimentPrefix}_seed${Seed}"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "marl environment Python was not found at $Python"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
Set-Location -LiteralPath $RepoRoot

function Get-LatestRunDir {
    param([string]$ExperimentName)
    $ExperimentDir = Join-Path $ResultsRoot $ExperimentName
    if (-not (Test-Path -LiteralPath $ExperimentDir)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $ExperimentDir -Directory |
        Where-Object { $_.Name -match "^run\d+$" } |
        Sort-Object { [int]$_.Name.Substring(3) } |
        Select-Object -Last 1
}

function Get-BestDir {
    param([string]$ExperimentName)
    $RunDir = Get-LatestRunDir -ExperimentName $ExperimentName
    if ($null -eq $RunDir) {
        return $null
    }
    $BestDir = Join-Path $RunDir.FullName "models\best"
    if (
        (Test-Path -LiteralPath (Join-Path $BestDir "actor.pt")) -and
        (Test-Path -LiteralPath (Join-Path $BestDir "best_checkpoint.json"))
    ) {
        return $BestDir
    }
    return $null
}

function Get-RunReachedTarget {
    param([string]$ExperimentName)
    $LogPath = Join-Path $LogDir "$ExperimentName.log"
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $false
    }
    $Text = Get-Content -LiteralPath $LogPath -Raw
    $Matches = [regex]::Matches(
        $Text,
        "total num timesteps\s+(\d+)/$Steps"
    )
    if ($Matches.Count -eq 0) {
        return $false
    }
    $MaxStep = 0
    foreach ($Match in $Matches) {
        $Value = [int]$Match.Groups[1].Value
        if ($Value -gt $MaxStep) {
            $MaxStep = $Value
        }
    }
    $Tolerance = 6 * $RolloutThreads * $EpisodeLength
    return ($MaxStep -ge ($Steps - $Tolerance))
}

$TrainLog = Join-Path $LogDir "$Experiment.log"
$TrainErr = Join-Path $LogDir "$Experiment.err.log"
Remove-Item -LiteralPath $TrainLog, $TrainErr -Force `
    -ErrorAction SilentlyContinue

$TrainArgs = @(
    "-u",
    "-m", "onpolicy.scripts.train.train_mec",
    "--env_name", "MEC",
    "--algorithm_name", "mappo",
    "--mec_scenario", "v6_hap_loadbearing",
    "--mec_episode_horizon", "$EpisodeLength",
    "--n_rollout_threads", "$RolloutThreads",
    "--n_training_threads", "2",
    "--n_eval_rollout_threads", "8",
    "--episode_length", "$EpisodeLength",
    "--num_env_steps", "$Steps",
    "--ppo_epoch", "5",
    "--num_mini_batch", "1",
    "--hidden_size", "128",
    "--layer_N", "2",
    "--use_entropy_anneal",
    "--mec_logstd_init", "-1.2",
    "--entropy_coef", "0.003",
    "--lr", "5e-4",
    "--critic_lr", "5e-4",
    "--gamma", "0.99",
    "--mec_rolewise_loss",
    "--log_interval", "6",
    "--save_interval", "18",
    "--save_step_checkpoints",
    "--use_eval",
    "--eval_interval", "18",
    "--eval_episodes", "$ValidationEpisodes",
    "--eval_seed", "$ValidationSeed",
    "--test_episodes", "$TestEpisodes",
    "--test_seed", "$TestSeed",
    "--test_seed_stride", "$TestSeedStride",
    "--use_wandb",
    "--experiment_name", $Experiment,
    "--seed", "$Seed",
    "--mec_policy_arch", "flat_descriptor",
    "--mec_flat_descriptor_dim", "$DescriptorDim",
    "--mec_critic_arch", "flat"
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $TrainArgs `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $TrainLog `
    -RedirectStandardError $TrainErr `
    -PassThru
Set-Content -LiteralPath (Join-Path $LogDir "$Experiment.pid") `
    -Value $Process.Id
Write-Output "[$(Get-Date -Format o)] started training $Experiment pid=$($Process.Id)"
$Process.WaitForExit()
$Process.Refresh()

if ($Process.ExitCode -ne 0) {
    throw "Training failed for $Experiment with exit code $($Process.ExitCode)"
}
if ((Test-Path -LiteralPath $TrainErr) -and
    ((Get-Item -LiteralPath $TrainErr).Length -gt 0)) {
    throw "Training stderr is non-empty for $Experiment"
}
if (-not (Get-RunReachedTarget -ExperimentName $Experiment)) {
    throw "Training log did not reach target steps for $Experiment"
}

$BestDir = Get-BestDir -ExperimentName $Experiment
if ($null -eq $BestDir) {
    throw "Best checkpoint missing for $Experiment"
}

$EvalLog = Join-Path $LogDir "$Experiment.heldout.log"
$EvalErr = Join-Path $LogDir "$Experiment.heldout.err.log"
$EvalJson = Join-Path $EvalDir "$Experiment.json"
Remove-Item -LiteralPath $EvalLog, $EvalErr, $EvalJson -Force `
    -ErrorAction SilentlyContinue

$EvalArgs = @(
    "-u",
    "-m", "onpolicy.scripts.eval.eval_mec",
    "--env_name", "MEC",
    "--mec_eval_controller", "policy",
    "--model_dir", $BestDir,
    "--mec_eval_seed", "$TestSeed",
    "--mec_eval_seed_stride", "$TestSeedStride",
    "--mec_eval_episodes", "$TestEpisodes",
    "--mec_eval_output", $EvalJson,
    "--mec_eval_skip_baselines"
)

$EvalProcess = Start-Process `
    -FilePath $Python `
    -ArgumentList $EvalArgs `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $EvalLog `
    -RedirectStandardError $EvalErr `
    -PassThru
Write-Output "[$(Get-Date -Format o)] started held-out eval $Experiment pid=$($EvalProcess.Id)"
$EvalProcess.WaitForExit()
$EvalProcess.Refresh()

if ($EvalProcess.ExitCode -ne 0) {
    throw "Held-out eval failed for $Experiment with exit code $($EvalProcess.ExitCode)"
}
if ((Test-Path -LiteralPath $EvalErr) -and
    ((Get-Item -LiteralPath $EvalErr).Length -gt 0)) {
    throw "Held-out eval stderr is non-empty for $Experiment"
}
if (-not (Test-Path -LiteralPath $EvalJson)) {
    throw "Held-out eval output missing for $Experiment"
}

$Data = Get-Content -LiteralPath $EvalJson -Raw | ConvertFrom-Json
$Cost = $Data.metrics.train / $Data.episode_horizon
$FreezeCost = $Data.freeze_hap_metrics.train / $Data.episode_horizon
$FreezePct = 100.0 * ($FreezeCost / [Math]::Max($Cost, 1e-9) - 1.0)
$AcceptPct = 100.0 * $Data.metrics.A / (
    $Data.metrics.A + $Data.metrics.U
)
$Best = Get-Content `
    -LiteralPath (Join-Path $BestDir "best_checkpoint.json") `
    -Raw | ConvertFrom-Json

$Summary = [pscustomobject][ordered]@{
    name = "Flat descriptor actor + Flat critic, reset permutation seed $Seed"
    experiment = $Experiment
    seed = $Seed
    descriptor_dim = $DescriptorDim
    best_validation = $Best.eval_reward
    selected_step = $Best.total_num_steps
    cost_per_slot = $Cost
    acceptance_pct = $AcceptPct
    w1_m = $Data.metrics.w1
    overflow_per_episode = $Data.metrics.ovf
    source_cost = $Data.metrics.src
    hap_freeze_pct = $FreezePct
    output = $EvalJson
}

$SummaryPath = Join-Path $EvalDir "$Experiment.summary.json"
$Summary | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $SummaryPath -Encoding UTF8
Write-Output "[$(Get-Date -Format o)] summary written to $SummaryPath"
$Summary | Format-List
