# MAPPO/MAT for MPE

This is a slimmed-down copy of `marlbenchmark/on-policy` focused on:

- Multiagent Particle Environments (MPE)
- R-MAPPO / MAPPO / IPPO (`rmappo`, `mappo`, `ippo`)
- MAT (`mat`, `mat_dec`)

StarCraft/SMAC, Hanabi, Google Research Football, HAPPO, and HATRPO code has been removed.

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
