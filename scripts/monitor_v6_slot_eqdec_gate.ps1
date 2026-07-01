param(
    [int]$PollSeconds = 300,
    [int]$MaxParallel = 2,
    [int]$Steps = 1500000,
    [int]$EpisodeLength = 350,
    [int]$Seed = 1,
    [int]$RolloutThreads = 16,
    [int]$ValidationEpisodes = 24,
    [int]$ValidationSeed = 1000,
    [int]$TestEpisodes = 24,
    [int]$TestSeed = 100000,
    [int]$TestSeedStride = 13,
    [string]$ExperimentPrefix = "diag1500_slot_eqdec"
)

$ErrorActionPreference = "Stop"

if ($MaxParallel -lt 1 -or $MaxParallel -gt 3) {
    throw "MaxParallel must be between 1 and 3"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$EvalDir = Join-Path $RepoRoot "eval_outputs\slot_eqdec_1500k"
$ResultsRoot = Join-Path $RepoRoot (
    "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"
)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
Set-Location -LiteralPath $RepoRoot

$BaseTrainArgs = @(
    "-u",
    "-m", "onpolicy.scripts.train.train_mec",
    "--env_name", "MEC",
    "--algorithm_name", "mappo",
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
)

$Jobs = @(
    [pscustomobject]@{
        Name = "Full Set actor + slot-attention readout, Flat critic"
        Experiment = "${ExperimentPrefix}_set_slot_flatcrit_seed${Seed}"
        Args = @(
            "--mec_policy_arch", "set",
            "--mec_set_encoder_type", "mean_pool",
            "--mec_set_actor_context", "slot_attention",
            "--mec_critic_arch", "flat"
        )
    },
    [pscustomobject]@{
        Name = "Full Set actor + slot-attention readout, Set critic"
        Experiment = "${ExperimentPrefix}_set_slot_setcrit_seed${Seed}"
        Args = @(
            "--mec_policy_arch", "set",
            "--mec_set_encoder_type", "mean_pool",
            "--mec_set_actor_context", "slot_attention",
            "--mec_critic_arch", "set"
        )
    },
    [pscustomobject]@{
        Name = "Full Set actor + token cross-attention readout, Flat critic"
        Experiment = "${ExperimentPrefix}_set_cross_flatcrit_seed${Seed}"
        Args = @(
            "--mec_policy_arch", "set",
            "--mec_set_encoder_type", "mean_pool",
            "--mec_set_actor_context", "cross_attention",
            "--mec_critic_arch", "flat"
        )
    }
)

function Get-GpuSnapshot {
    try {
        return (& nvidia-smi `
            --query-gpu=index,name,utilization.gpu,memory.used,memory.total `
            --format=csv,noheader,nounits) -join "; "
    } catch {
        return "nvidia-smi unavailable"
    }
}

function Get-LatestRunDir {
    param([string]$Experiment)
    $ExperimentDir = Join-Path $ResultsRoot $Experiment
    if (-not (Test-Path -LiteralPath $ExperimentDir)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $ExperimentDir -Directory |
        Where-Object { $_.Name -match "^run\d+$" } |
        Sort-Object { [int]$_.Name.Substring(3) } |
        Select-Object -Last 1
}

function Get-BestDir {
    param([string]$Experiment)
    $RunDir = Get-LatestRunDir -Experiment $Experiment
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

function Get-JobProcess {
    param([string]$Experiment)
    $PidPath = Join-Path $LogDir "$Experiment.pid"
    if (-not (Test-Path -LiteralPath $PidPath)) {
        return $null
    }
    $ProcessId = [int](Get-Content -LiteralPath $PidPath -Raw)
    return Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
}

function Start-TrainingRun {
    param([pscustomobject]$Job)

    $StdoutPath = Join-Path $LogDir "$($Job.Experiment).log"
    $StderrPath = Join-Path $LogDir "$($Job.Experiment).err.log"
    Remove-Item -LiteralPath $StdoutPath, $StderrPath -Force `
        -ErrorAction SilentlyContinue
    $TrainArgs = $BaseTrainArgs + @(
        "--experiment_name", $Job.Experiment
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
    Write-Output (
        "[$(Get-Date -Format o)] started $($Job.Experiment) " +
        "pid=$($Process.Id)"
    )
}

while ($true) {
    $Active = @()
    $Completed = @()
    $Pending = @()
    foreach ($Job in $Jobs) {
        $Process = Get-JobProcess -Experiment $Job.Experiment
        if ($null -ne $Process) {
            $Active += [pscustomobject]@{
                job = $Job
                process = $Process
            }
            continue
        }
        $StderrPath = Join-Path $LogDir "$($Job.Experiment).err.log"
        if ((Test-Path -LiteralPath $StderrPath) -and
            ((Get-Item -LiteralPath $StderrPath).Length -gt 0)) {
            Write-Output (
                "[$(Get-Date -Format o)] will restart " +
                "$($Job.Experiment): non-empty stderr after process exit"
            )
            $Pending += $Job
            continue
        }
        $BestDir = Get-BestDir -Experiment $Job.Experiment
        if ($null -ne $BestDir) {
            $Completed += $Job
            continue
        }
        $Pending += $Job
    }

    while ($Pending.Count -gt 0 -and $Active.Count -lt $MaxParallel) {
        $Job = $Pending[0]
        Start-TrainingRun -Job $Job
        Start-Sleep -Seconds 3
        $Process = Get-JobProcess -Experiment $Job.Experiment
        if ($null -ne $Process) {
            $Active += [pscustomobject]@{
                job = $Job
                process = $Process
            }
        }
        $Pending = @($Pending | Select-Object -Skip 1)
    }

    $Cpu = (Get-CimInstance Win32_Processor |
        Select-Object -ExpandProperty LoadPercentage)
    $Mem = Get-CimInstance Win32_OperatingSystem |
        Select-Object FreePhysicalMemory, TotalVisibleMemorySize
    Write-Output (
        "[$(Get-Date -Format o)] active=$($Active.Count) " +
        "completed=$($Completed.Count)/$($Jobs.Count) cpu=$Cpu " +
        "free_mem_mb=$([Math]::Round($Mem.FreePhysicalMemory / 1024, 1)) " +
        "gpu=$(Get-GpuSnapshot)"
    )
    foreach ($Item in $Active) {
        Write-Output (
            "  running $($Item.job.Experiment) pid=$($Item.process.Id) " +
            "cpu=$($Item.process.CPU) rss_mb=$([Math]::Round($Item.process.WorkingSet64 / 1MB, 1))"
        )
    }

    if ($Completed.Count -eq $Jobs.Count) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Output "[$(Get-Date -Format o)] all training runs complete"

$Evaluations = @()
foreach ($Job in $Jobs) {
    $StderrPath = Join-Path $LogDir "$($Job.Experiment).err.log"
    if ((Test-Path -LiteralPath $StderrPath) -and
        ((Get-Item -LiteralPath $StderrPath).Length -gt 0)) {
        throw "Training stderr is non-empty for $($Job.Experiment)"
    }

    $BestDir = Get-BestDir -Experiment $Job.Experiment
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
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $EvalArgs `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $EvalLog `
        -RedirectStandardError $EvalErr `
        -PassThru
    Write-Output (
        "[$(Get-Date -Format o)] started held-out eval " +
        "$($Job.Experiment) pid=$($Process.Id)"
    )
    $Evaluations += [pscustomobject]@{
        Job = $Job
        Process = $Process
        Output = $EvalJson
        Stderr = $EvalErr
        BestDir = $BestDir
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
    $BestJson = Join-Path $Evaluation.BestDir "best_checkpoint.json"
    $Best = Get-Content -LiteralPath $BestJson -Raw | ConvertFrom-Json
    $Summary += [pscustomobject][ordered]@{
        name = $Evaluation.Job.Name
        experiment = $Evaluation.Job.Experiment
        best_validation = $Best.eval_reward
        selected_step = $Best.total_num_steps
        cost_per_slot = $Cost
        acceptance_pct = $AcceptPct
        w1_m = $Data.metrics.w1
        overflow_per_episode = $Data.metrics.ovf
        source_cost = $Data.metrics.src
        hap_freeze_pct = $FreezePct
        output = $Evaluation.Output
    }
}

$SummaryPath = Join-Path $EvalDir "$ExperimentPrefix.summary.json"
$Summary | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $SummaryPath -Encoding UTF8
$Summary | Format-Table `
    name, experiment, best_validation, selected_step, cost_per_slot, `
    acceptance_pct, w1_m, hap_freeze_pct `
    -AutoSize

Write-Output "[$(Get-Date -Format o)] summary written to $SummaryPath"
