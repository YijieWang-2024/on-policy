param(
    [int]$Steps = 403200,
    [int]$EpisodeLength = 350,
    [int]$Seed = 1,
    [int]$RolloutThreads = 16,
    [int]$ValidationEpisodes = 24,
    [int]$ValidationSeed = 1000,
    [int]$TestEpisodes = 24,
    [int]$TestSeed = 100000,
    [int]$TestSeedStride = 13,
    [int]$MaxParallel = 3,
    [string]$ExperimentPrefix = "diag400_desc"
)

$ErrorActionPreference = "Stop"

if ($MaxParallel -lt 1 -or $MaxParallel -gt 3) {
    throw "MaxParallel must be between 1 and 3 on this workstation"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$EvalDir = Join-Path $RepoRoot "eval_outputs\descriptor_diagnostics_400k"
$ResultsRoot = Join-Path $RepoRoot (
    "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"
)
$ManifestPath = Join-Path $LogDir "$ExperimentPrefix.manifest.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "marl environment Python was not found at $Python"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
Set-Location -LiteralPath $RepoRoot

$Jobs = @(
    [pscustomobject]@{
        Name = "sortflat_public_descriptor"
        Experiment = "${ExperimentPrefix}_sortflat_seed${Seed}"
        Args = @("--mec_policy_arch", "sort_flat")
    },
    [pscustomobject]@{
        Name = "set_actor_flat_critic"
        Experiment = "${ExperimentPrefix}_setactor_flatcritic_seed${Seed}"
        Args = @(
            "--mec_policy_arch", "set",
            "--mec_set_encoder_type", "mean_pool",
            "--mec_critic_arch", "flat"
        )
    },
    [pscustomobject]@{
        Name = "flat_actor_set_critic"
        Experiment = "${ExperimentPrefix}_flatactor_setcritic_seed${Seed}"
        Args = @(
            "--mec_policy_arch", "flat",
            "--mec_critic_arch", "set",
            "--mec_set_encoder_type", "mean_pool",
            "--mec_set_critic_encoder", "separate"
        )
    }
)

$Manifest = [ordered]@{
    scenario = "v6_hap_loadbearing"
    purpose = "diagnose public descriptor/readout and actor-critic coupling"
    episode_length = $EpisodeLength
    steps = $Steps
    seed = $Seed
    rollout_threads = $RolloutThreads
    validation_seed = $ValidationSeed
    validation_episodes = $ValidationEpisodes
    test_seed = $TestSeed
    test_seed_stride = $TestSeedStride
    test_episodes = $TestEpisodes
    max_parallel = $MaxParallel
    experiments = @($Jobs | ForEach-Object {
        [ordered]@{
            name = $_.Name
            experiment = $_.Experiment
            args = $_.Args
        }
    })
    started_at = (Get-Date -Format o)
}
$Manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $ManifestPath -Encoding UTF8

function Start-TrainingRun {
    param([pscustomobject]$Job)

    $StdoutPath = Join-Path $LogDir "$($Job.Experiment).log"
    $StderrPath = Join-Path $LogDir "$($Job.Experiment).err.log"
    $TrainArgs = @(
        "-u",
        "-m", "onpolicy.scripts.train.train_mec",
        "--env_name", "MEC",
        "--algorithm_name", "mappo",
        "--experiment_name", $Job.Experiment,
        "--mec_scenario", "v6_hap_loadbearing",
        "--mec_episode_horizon", "$EpisodeLength",
        "--seed", "$Seed",
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
        "--use_wandb"
    ) + $Job.Args

    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $TrainArgs `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru
    Set-Content `
        -LiteralPath (Join-Path $LogDir "$($Job.Experiment).pid") `
        -Value $Process.Id
    Write-Host (
        "[$(Get-Date -Format o)] Started $($Job.Experiment) " +
        "PID=$($Process.Id)"
    )
    return [pscustomobject]@{
        Job = $Job
        Process = $Process
        Stdout = $StdoutPath
        Stderr = $StderrPath
    }
}

$Active = @()
for ($Start = 0; $Start -lt $Jobs.Count; $Start += $MaxParallel) {
    $End = [Math]::Min($Start + $MaxParallel - 1, $Jobs.Count - 1)
    foreach ($Job in $Jobs[$Start..$End]) {
        $Active += Start-TrainingRun -Job $Job
    }

    foreach ($Run in $Active) {
        $Run.Process.WaitForExit()
        $Run.Process.Refresh()
        $ExperimentDir = Join-Path $ResultsRoot $Run.Job.Experiment
        $RunDir = Get-ChildItem -LiteralPath $ExperimentDir -Directory |
            Where-Object { $_.Name -match "^run\d+$" } |
            Sort-Object { [int]$_.Name.Substring(3) } |
            Select-Object -Last 1
        $BestDir = if ($null -eq $RunDir) {
            $null
        } else {
            Join-Path $RunDir.FullName "models\best"
        }
        $ExitOk = (
            $Run.Process.ExitCode -eq 0 -or
            $null -eq $Run.Process.ExitCode
        )
        $Complete = (
            $ExitOk -and
            $null -ne $BestDir -and
            (Test-Path -LiteralPath (Join-Path $BestDir "actor.pt")) -and
            (Test-Path -LiteralPath (
                Join-Path $BestDir "best_checkpoint.json"
            ))
        )
        $ErrorBytes = (Get-Item -LiteralPath $Run.Stderr).Length
        Write-Output (
            "[$(Get-Date -Format o)] Finished $($Run.Job.Experiment) " +
            "exit=$($Run.Process.ExitCode) complete=$Complete " +
            "stderr_bytes=$ErrorBytes"
        )
        if (-not $Complete -or $ErrorBytes -gt 0) {
            throw "Training failed: $($Run.Job.Experiment)"
        }
        $Run | Add-Member -NotePropertyName BestDir -NotePropertyValue $BestDir
    }
    $Active = @()
}

$Evaluations = @()
foreach ($Job in $Jobs) {
    $ExperimentDir = Join-Path $ResultsRoot $Job.Experiment
    $RunDir = Get-ChildItem -LiteralPath $ExperimentDir -Directory |
        Where-Object { $_.Name -match "^run\d+$" } |
        Sort-Object { [int]$_.Name.Substring(3) } |
        Select-Object -Last 1
    $BestDir = Join-Path $RunDir.FullName "models\best"
    $EvalLog = Join-Path $LogDir "$($Job.Experiment).heldout.log"
    $EvalErr = Join-Path $LogDir "$($Job.Experiment).heldout.err.log"
    $EvalJson = Join-Path $EvalDir "$($Job.Experiment).json"
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
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $EvalArgs `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $EvalLog `
        -RedirectStandardError $EvalErr `
        -PassThru
    Write-Host (
        "[$(Get-Date -Format o)] Started held-out " +
        "$($Job.Experiment) PID=$($Process.Id)"
    )
    $Evaluations += [pscustomobject]@{
        Job = $Job
        Process = $Process
        Output = $EvalJson
        Stderr = $EvalErr
    }
}

foreach ($Evaluation in $Evaluations) {
    $Evaluation.Process.WaitForExit()
    $Evaluation.Process.Refresh()
    $ErrorBytes = (Get-Item -LiteralPath $Evaluation.Stderr).Length
    $ExitOk = (
        $Evaluation.Process.ExitCode -eq 0 -or
        $null -eq $Evaluation.Process.ExitCode
    )
    if (
        -not $ExitOk -or
        -not (Test-Path -LiteralPath $Evaluation.Output) -or
        $ErrorBytes -gt 0
    ) {
        throw "Held-out evaluation failed for $($Evaluation.Job.Experiment)"
    }
}

$Summary = @()
foreach ($Evaluation in $Evaluations) {
    $Data = Get-Content -LiteralPath $Evaluation.Output -Raw |
        ConvertFrom-Json
    $Cost = $Data.metrics.train / $Data.episode_horizon
    $FreezeCost = $Data.freeze_hap_metrics.train / $Data.episode_horizon
    $FreezePct = 100.0 * ($FreezeCost / [Math]::Max($Cost, 1e-9) - 1.0)
    $AcceptPct = 100.0 * $Data.metrics.A / (
        $Data.metrics.A + $Data.metrics.U
    )
    $Summary += [ordered]@{
        name = $Evaluation.Job.Name
        experiment = $Evaluation.Job.Experiment
        cost_per_slot = $Cost
        acceptance_pct = $AcceptPct
        w1_m = $Data.metrics.w1
        hap_freeze_pct = $FreezePct
        output = $Evaluation.Output
    }
}

$SummaryPath = Join-Path $EvalDir "$ExperimentPrefix.summary.json"
$Summary | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $SummaryPath -Encoding UTF8

$Summary | Format-Table `
    name, experiment, cost_per_slot, acceptance_pct, w1_m, hap_freeze_pct `
    -AutoSize

$Manifest.completed_at = (Get-Date -Format o)
$Manifest.status = "complete"
$Manifest.summary = $SummaryPath
$Manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $ManifestPath -Encoding UTF8

Write-Host "Summary written to $SummaryPath"
