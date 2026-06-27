#!/usr/bin/env python
"""Offline reconstruction pretraining for MEC Set population encoders."""

from __future__ import annotations

import json
import os
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import ACT_DIM, MECEnv
from onpolicy.algorithms.mec.mec_policy import MECPolicy


def parse_args(args, parser):
    parser.add_argument(
        "--mec_scenario",
        type=str,
        default="v2_iort_6km_mmwave",
        help="scenario yaml name under onpolicy/envs/mec/scenarios",
    )
    parser.add_argument("--mec_fleet_size", type=int, default=None)
    parser.add_argument(
        "--mec_episode_horizon",
        type=int,
        default=None,
        help="override the MEC scenario horizon",
    )
    parser.add_argument(
        "--mec_pretrain_episodes",
        type=int,
        default=64,
        help="number of environment episodes used to collect replay atoms",
    )
    parser.add_argument(
        "--mec_pretrain_seed_stride",
        type=int,
        default=17,
        help="seed stride between replay collection episodes",
    )
    parser.add_argument(
        "--mec_pretrain_controller",
        type=str,
        default="mixed",
        choices=["random", "hover", "mixed"],
        help="controller used while collecting reconstruction replay atoms",
    )
    parser.add_argument(
        "--mec_pretrain_epochs",
        type=int,
        default=40,
        help="number of offline reconstruction epochs",
    )
    parser.add_argument(
        "--mec_pretrain_batch_size",
        type=int,
        default=512,
        help="offline reconstruction minibatch size",
    )
    parser.add_argument(
        "--mec_pretrain_lr",
        type=float,
        default=1e-3,
        help="learning rate for encoder-decoder reconstruction pretraining",
    )
    parser.add_argument(
        "--mec_pretrain_output_dir",
        type=str,
        default=None,
        help="directory for actor.pt and pretrain metadata",
    )
    return parser.parse_known_args(args)[0]


def _action(env: MECEnv, controller: str, rng: np.random.Generator):
    if controller == "hover":
        action = np.zeros((env.num_agents, ACT_DIM), dtype=np.float32)
        action[1:, 2] = 0.5
        return action
    if controller == "random":
        return rng.uniform(-1.0, 1.0, (env.num_agents, ACT_DIM)).astype(
            np.float32
        )
    if controller == "mixed":
        if rng.random() < 0.5:
            return _action(env, "hover", rng)
        return _action(env, "random", rng)
    raise ValueError(f"unsupported pretrain controller: {controller}")


def _collect_atoms(args, env: MECEnv) -> np.ndarray:
    rng = np.random.default_rng(int(args.seed))
    atoms = []
    horizon = int(env.cfg["base"]["episode_horizon_slots"])
    for episode in range(int(args.mec_pretrain_episodes)):
        episode_seed = int(args.seed) + int(args.mec_pretrain_seed_stride) * episode
        obs, _ = env.reset(seed=episode_seed)
        for _ in range(horizon):
            atoms.append(obs[1:, 1:4].copy())
            obs, _, _, _, _ = env.step(
                _action(env, args.mec_pretrain_controller, rng)
            )
    return np.asarray(atoms, dtype=np.float32)


def _output_dir(args) -> Path:
    if args.mec_pretrain_output_dir:
        return Path(args.mec_pretrain_output_dir)
    root = (
        Path(os.path.dirname(os.path.abspath(__file__))).parents[0]
        / "results"
        / args.env_name
        / args.mec_scenario
        / args.algorithm_name
        / args.experiment_name
        / "pretrain"
    )
    return root


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    if all_args.env_name != "MEC":
        raise ValueError("MEC reconstruction pretraining requires --env_name MEC")
    all_args.algorithm_name = "mappo"
    all_args.use_recurrent_policy = False
    all_args.use_naive_recurrent_policy = False
    all_args.use_centralized_V = True
    all_args.mec_policy_arch = "set"
    if float(all_args.mec_set_reconstruction_coef) <= 0.0:
        all_args.mec_set_reconstruction_coef = 1.0

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.set_num_threads(all_args.n_training_threads)
    device = (
        torch.device("cuda:0")
        if all_args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )

    env = MECEnv(all_args)
    all_args.num_agents = env.num_agents
    dataset = _collect_atoms(all_args, env)
    if dataset.ndim != 3 or dataset.shape[0] == 0:
        raise RuntimeError("empty reconstruction replay dataset")

    policy = MECPolicy(
        all_args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
        num_agents=env.num_agents,
    )
    policy.actor.train()
    params = list(policy.actor.population_encoder.parameters()) + list(
        policy.actor.reconstruction_decoder.parameters()
    )
    optimizer = torch.optim.Adam(params, lr=float(all_args.mec_pretrain_lr))

    atoms = torch.as_tensor(dataset, dtype=torch.float32, device=device)
    n = atoms.shape[0]
    losses = []
    batch_size = int(all_args.mec_pretrain_batch_size)
    for epoch in range(int(all_args.mec_pretrain_epochs)):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            batch = atoms[perm[start:start + batch_size]]
            optimizer.zero_grad()
            loss = policy.actor.reconstruction_chamfer_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, all_args.max_grad_norm)
            optimizer.step()
            count = int(batch.shape[0])
            epoch_loss += loss.item() * count
            seen += count
        mean_loss = epoch_loss / max(seen, 1)
        losses.append(mean_loss)
        print(
            f"pretrain epoch {epoch + 1:03d}/"
            f"{int(all_args.mec_pretrain_epochs):03d} "
            f"chamfer={mean_loss:.6f}"
        )

    output_dir = _output_dir(all_args)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.actor.state_dict(), str(output_dir / "actor.pt"))
    metadata = {
        "host": socket.gethostname(),
        "scenario": all_args.mec_scenario,
        "fleet_size_k": int(env.k),
        "episode_horizon": int(env.cfg["base"]["episode_horizon_slots"]),
        "episodes": int(all_args.mec_pretrain_episodes),
        "samples": int(n),
        "controller": all_args.mec_pretrain_controller,
        "encoder_type": all_args.mec_set_encoder_type,
        "seed": int(all_args.seed),
        "epochs": int(all_args.mec_pretrain_epochs),
        "batch_size": batch_size,
        "lr": float(all_args.mec_pretrain_lr),
        "loss_first": float(losses[0]),
        "loss_final": float(losses[-1]),
    }
    with (output_dir / "pretrain_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"saved pretrained actor to {output_dir / 'actor.pt'}")


if __name__ == "__main__":
    main(sys.argv[1:])
