param(
    [int]$Seed = 1,
    [int]$Steps = 1500000,
    [int]$EpisodeLength = 350,
    [int]$RolloutThreads = 16,
    [int]$ValidationEpisodes = 24,
    [int]$ValidationSeed = 1000,
    [int]$TestEpisodes = 24,
    [int]$TestSeed = 100000,
    [int]$TestSeedStride = 13
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$EvalDir = Join-Path $RepoRoot "eval_outputs\resetperm_mean_critic_isolation_1500k"
$ResultsRoot = Join-Path $RepoRoot (
    "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"
)

if (-not (Test-Path -LiteralPath $Python)) {
    throw "marl environment Python was not found at $Python"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
Set-Location -LiteralPath $RepoRoot

$CommonTrainArgs = @(
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
    "--seed", "$Seed"
)

$Jobs = @(
    [pscustomobject]@{
        Name = "Mean actor + separate Set critic"
        Experiment = "resetperm1500_mean_actor_setcrit_separate_seed$Seed"
        ExtraArgs = @(
            "--mec_policy_arch", "mean",
            "--mec_critic_arch", "set",
            "--mec_set_critic_encoder", "separate"
        )
    },
    [pscustomobject]@{
        Name = "Mean actor + Flat critic"
        Experiment = "resetperm1500_mean_actor_flatcrit_seed$Seed"
        ExtraArgs = @(
            "--mec_policy_arch", "mean",
            "--mec_critic_arch", "flat"
        )
    }
)

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

function Start-Train {
    param([pscustomobject]$Job)
    $TrainLog = Join-Path $LogDir "$($Job.Experiment).log"
    $TrainErr = Join-Path $LogDir "$($Job.Experiment).err.log"
    Remove-Item -LiteralPath $TrainLog, $TrainErr -Force `
        -ErrorAction SilentlyContinue
    $Args = $CommonTrainArgs + @(
        "--experiment_name", $Job.Experiment
    ) + $Job.ExtraArgs
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Args `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TrainLog `
        -RedirectStandardError $TrainErr `
        -PassThru
    Set-Content -LiteralPath (Join-Path $LogDir "$($Job.Experiment).pid") `
        -Value $Process.Id
    $Job | Add-Member -NotePropertyName Process -NotePropertyValue $Process
    Write-Output "[$(Get-Date -Format o)] started $($Job.Name) pid=$($Process.Id)"
}

function Assert-TrainOk {
    param([pscustomobject]$Job)
    $Process = $Job.Process
    $Process.Refresh()
    if ($null -ne $Process.ExitCode -and $Process.ExitCode -ne 0) {
        throw "Training failed for $($Job.Experiment) exit=$($Process.ExitCode)"
    }
    $TrainErr = Join-Path $LogDir "$($Job.Experiment).err.log"
    if ((Test-Path -LiteralPath $TrainErr) -and
        ((Get-Item -LiteralPath $TrainErr).Length -gt 0)) {
        throw "Training stderr is non-empty for $($Job.Experiment)"
    }
    if (-not (Get-RunReachedTarget -ExperimentName $Job.Experiment)) {
        throw "Training log did not reach target steps for $($Job.Experiment)"
    }
}

function Run-HeldOutEval {
    param([pscustomobject]$Job)
    $BestDir = Get-BestDir -ExperimentName $Job.Experiment
    if ($null -eq $BestDir) {
        throw "Best checkpoint missing for $($Job.Experiment)"
    }

    $EvalLog = Join-Path $LogDir "$($Job.Experiment).heldout.log"
    $EvalErr = Join-Path $LogDir "$($Job.Experiment).heldout.err.log"
    $EvalJson = Join-Path $EvalDir "$($Job.Experiment).json"
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
    Write-Host "[$(Get-Date -Format o)] started eval $($Job.Experiment) pid=$($EvalProcess.Id)"
    $EvalProcess.WaitForExit()
    $EvalProcess.Refresh()

    if ($null -ne $EvalProcess.ExitCode -and $EvalProcess.ExitCode -ne 0) {
        throw "Held-out eval failed for $($Job.Experiment) exit=$($EvalProcess.ExitCode)"
    }
    if ((Test-Path -LiteralPath $EvalErr) -and
        ((Get-Item -LiteralPath $EvalErr).Length -gt 0)) {
        throw "Held-out eval stderr is non-empty for $($Job.Experiment)"
    }
    if (-not (Test-Path -LiteralPath $EvalJson)) {
        throw "Held-out eval output missing for $($Job.Experiment)"
    }

    $Data = Get-Content -LiteralPath $EvalJson -Raw | ConvertFrom-Json
    $Best = Get-Content `
        -LiteralPath (Join-Path $BestDir "best_checkpoint.json") `
        -Raw | ConvertFrom-Json
    $Cost = $Data.metrics.train / $Data.episode_horizon
    $FreezeCost = $Data.freeze_hap_metrics.train / $Data.episode_horizon
    $FreezePct = 100.0 * ($FreezeCost / [Math]::Max($Cost, 1e-9) - 1.0)
    $AcceptPct = 100.0 * $Data.metrics.A / (
        $Data.metrics.A + $Data.metrics.U
    )

    return [pscustomobject][ordered]@{
        name = $Job.Name
        experiment = $Job.Experiment
        seed = $Seed
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
}

foreach ($Job in $Jobs) {
    Start-Train -Job $Job
}

foreach ($Job in $Jobs) {
    $Job.Process.WaitForExit()
    Assert-TrainOk -Job $Job
}

$Summaries = @()
foreach ($Job in $Jobs) {
    $Summaries += Run-HeldOutEval -Job $Job
}

$SummaryPath = Join-Path $EvalDir "resetperm1500_mean_critic_isolation_seed$Seed.summary.json"
$Summaries | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $SummaryPath -Encoding UTF8
Write-Output "[$(Get-Date -Format o)] summary written to $SummaryPath"
$Summaries | Format-Table `
    name, experiment, best_validation, selected_step, cost_per_slot, `
    acceptance_pct, w1_m, hap_freeze_pct -AutoSize
