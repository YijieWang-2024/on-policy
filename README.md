# MAPPO/MAT for MPE

This is a slimmed-down copy of `marlbenchmark/on-policy` focused on:

- Multiagent Particle Environments (MPE)
- R-MAPPO / MAPPO / IPPO (`rmappo`, `mappo`, `ippo`)
- MAT (`mat`, `mat_dec`)

StarCraft/SMAC, Hanabi, Google Research Football, HAPPO, and HATRPO code has been removed.

The MPE stack uses Gymnasium and NumPy 2. MPE time limits are represented as
`truncated=True`; true task-ending conditions use `terminated=True`.

## Installation

```bash
conda env create -f environment.yaml
conda activate marl
```

If the `marl` environment already exists:

```bash
conda activate marl
pip install -r requirements.txt
```

## Train

Run commands from the project root. The project is not installed as a Python library.

Run a small MPE smoke test without Weights & Biases:

```bash
cd /Users/qiaonan/Projects/on-policy
python -m onpolicy.scripts.train.train_mpe \
  --env_name MPE \
  --algorithm_name mappo \
  --experiment_name smoke \
  --scenario_name simple_spread \
  --num_agents 3 \
  --num_landmarks 3 \
  --seed 1 \
  --n_training_threads 1 \
  --n_rollout_threads 1 \
  --num_mini_batch 1 \
  --episode_length 5 \
  --num_env_steps 10 \
  --ppo_epoch 1 \
  --use_ReLU \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 7e-4 \
  --user_name local \
  --use_wandb
```

In this codebase, `--use_wandb` is a legacy inverse switch: adding it disables wandb and writes TensorBoard logs locally.

Example shell scripts for MPE live in `onpolicy/scripts/train_mpe_scripts`.

## Termination Semantics

Vector environments automatically reset completed MPE episodes. The reset
observation is returned for the next rollout step, while the actual last
observation is retained in `info["final_observation"]`.

The return calculation deliberately uses two masks:

- `masks`: zero after either termination or truncation, resetting recurrent
  state and preventing GAE from crossing into the next episode.
- `bootstrap_masks`: zero only after termination, so truncations bootstrap from
  the critic value of `final_observation`.

This follows Gymnasium's time-limit handling and avoids treating a fixed
rollout horizon as a terminal state.

PPO training logs `approx_kl`, `clip_fraction`, and `explained_variance`.
Set `--target_kl <value>` to optionally stop PPO epochs early when an update
moves too far from the rollout policy; it is disabled by default.

For a new MPE-like environment, return `truncated=True` when an external step
limit causes a reset and `terminated=True` only when the task reaches a true
terminal state. If a finite horizon is itself part of the task, expose the
remaining time in the observation and treat the horizon as termination instead.

Run the focused migration tests with:

```bash
conda run -n marl python -m unittest discover -s tests -v
```

## Citation

If you use this codebase, please cite the original MAPPO paper:

```bibtex
@inproceedings{
yu2022the,
title={The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
author={Chao Yu and Akash Velu and Eugene Vinitsky and Jiaxuan Gao and Yu Wang and Alexandre Bayen and Yi Wu},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022}
}
```
