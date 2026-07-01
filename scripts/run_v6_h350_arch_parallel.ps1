param(
    [int]$Steps = 896000,
    [int]$EpisodeLength = 350,
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$Architectures = @("mean", "flat", "set"),
    [int]$MaxParallel = 3,
    [int]$RolloutThreads = 16,
    [int]$ValidationEpisodes = 24,
    [int]$ValidationSeed = 1000,
    [int]$TestEpisodes = 24,
    [int]$TestSeed = 100000,
    [int]$TestSeedStride = 13,
    [string]$ExperimentPrefix = "v6_h350"
)

$ErrorActionPreference = "Stop"

if ($MaxParallel -lt 1 -or $MaxParallel -gt 3) {
    throw "MaxParallel must be between 1 and 3 on this workstation"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE ".conda\envs\marl\python.exe"
$LogDir = Join-Path $RepoRoot "training_logs"
$ResultsRoot = Join-Path $RepoRoot (
    "onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo"
)
$ManifestPath = Join-Path $LogDir "$ExperimentPrefix.manifest.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "marl environment Python was not found at $Python"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $RepoRoot

$Jobs = @()
foreach ($Seed in $Seeds) {
    foreach ($Architecture in $Architectures) {
        $Jobs += [pscustomobject]@{
            Seed = $Seed
            Architecture = $Architecture
            Experiment = "${ExperimentPrefix}_${Architecture}_seed${Seed}"
        }
    }
}

$Manifest = [ordered]@{
    scenario = "v6_hap_loadbearing"
    episode_length = $EpisodeLength
    steps = $Steps
    rollout_threads = $RolloutThreads
    validation_seed = $ValidationSeed
    validation_episodes = $ValidationEpisodes
    test_seed = $TestSeed
    test_seed_stride = $TestSeedStride
    test_episodes = $TestEpisodes
    max_parallel = $MaxParallel
    experiments = @($Jobs | ForEach-Object { $_.Experiment })
    started_at = (Get-Date -Format o)
}
$Manifest | ConvertTo-Json -Depth 4 |
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
        "--mec_policy_arch", $Job.Architecture,
        "--mec_episode_horizon", "$EpisodeLength",
        "--seed", "$($Job.Seed)",
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
        "--eval_episodes", "$ValidationEpisodes",
        "--eval_seed", "$ValidationSeed",
        "--test_episodes", "$TestEpisodes",
        "--test_seed", "$TestSeed",
        "--test_seed_stride", "$TestSeedStride",
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

$Failed = @()
for ($Start = 0; $Start -lt $Jobs.Count; $Start += $MaxParallel) {
    $End = [Math]::Min($Start + $MaxParallel - 1, $Jobs.Count - 1)
    $Wave = @()
    foreach ($Job in $Jobs[$Start..$End]) {
        $Wave += Start-TrainingRun -Job $Job
    }

    foreach ($Run in $Wave) {
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
        $Complete = (
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
            $Failed += $Run.Job.Experiment
        }
    }

    if ($Failed.Count -gt 0) {
        throw "Training failed: $($Failed -join ', ')"
    }
}

for ($Start = 0; $Start -lt $Jobs.Count; $Start += $MaxParallel) {
    $End = [Math]::Min($Start + $MaxParallel - 1, $Jobs.Count - 1)
    $EvalWave = @()
    foreach ($Job in $Jobs[$Start..$End]) {
        $ExperimentDir = Join-Path $ResultsRoot $Job.Experiment
        $RunDir = Get-ChildItem -LiteralPath $ExperimentDir -Directory |
            Where-Object { $_.Name -match "^run\d+$" } |
            Sort-Object { [int]$_.Name.Substring(3) } |
            Select-Object -Last 1
        $BestDir = Join-Path $RunDir.FullName "models\best"
        $EvalLog = Join-Path $LogDir "$($Job.Experiment).heldout.log"
        $EvalErr = Join-Path $LogDir "$($Job.Experiment).heldout.err.log"
        $EvalJson = Join-Path $LogDir "$($Job.Experiment).heldout.json"
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
        $EvalWave += [pscustomobject]@{
            Job = $Job
            Process = $Process
            Output = $EvalJson
            Stderr = $EvalErr
        }
    }
    foreach ($Evaluation in $EvalWave) {
        $Evaluation.Process.WaitForExit()
        $ErrorBytes = (Get-Item -LiteralPath $Evaluation.Stderr).Length
        if (
            -not (Test-Path -LiteralPath $Evaluation.Output) -or
            $ErrorBytes -gt 0
        ) {
            throw "Held-out evaluation failed for $($Evaluation.Job.Experiment)"
        }
    }
}

$HeuristicLog = Join-Path $LogDir "$ExperimentPrefix.heuristic.heldout.log"
$HeuristicJson = Join-Path $LogDir "$ExperimentPrefix.heuristic.heldout.json"
& $Python -u -m onpolicy.scripts.eval.eval_mec `
    --env_name MEC `
    --mec_scenario v6_hap_loadbearing `
    --mec_episode_horizon $EpisodeLength `
    --mec_eval_controller heuristic `
    --mec_eval_seed $TestSeed `
    --mec_eval_seed_stride $TestSeedStride `
    --mec_eval_episodes $TestEpisodes `
    --mec_eval_output $HeuristicJson 2>&1 |
    Tee-Object -FilePath $HeuristicLog
if ($LASTEXITCODE -ne 0) {
    throw "Held-out heuristic evaluation failed"
}

& $Python -u -m onpolicy.scripts.analysis.summarize_h350_arch `
    --log_dir $LogDir `
    --prefix $ExperimentPrefix
if ($LASTEXITCODE -ne 0) {
    throw "Architecture summary generation failed"
}

$Manifest.completed_at = (Get-Date -Format o)
$Manifest.status = "complete"
$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath $ManifestPath -Encoding UTF8
