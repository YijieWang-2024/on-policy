#!/usr/bin/env python
"""Render top deterministic-policy MEC episodes from multiple eval seeds.

For each trained run, evaluate the saved actor on N seeded episodes, select the
episode(s) with the lowest cumulative training cost, then replay those seeds and
save animated GIFs in the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import ACT_DIM, MECEnv
from onpolicy.envs.mec.config_loader import (
    access_coverage_radius_m,
    backhaul_service_radius_m,
)
from onpolicy.envs.mec.metrics import demand_matching_w1
from onpolicy.scripts.render.render_mec import _draw, _policy_action
from onpolicy.utils.run_config import apply_saved_args, load_model_config


DEFAULT_RUNS = (
    "v3_explorefix_k12/run1",
    "v3_explorefix_k12/run2",
    "v3_logpfix_k12/run1",
    "v3_logpfix_ent003/run1",
    "v3_logpfix_ent003/run2",
    "v3_logpfix_ent003/run3",
)


def _load_args(model_dir: Path) -> argparse.Namespace:
    base = get_config().parse_known_args([])[0]
    saved = load_model_config(model_dir)
    apply_saved_args(base, saved, explicit_names=set(), skip={"model_dir"})
    base.model_dir = str(model_dir)
    base.env_name = "MEC"
    base.use_recurrent_policy = base.algorithm_name == "rmappo"
    base.use_naive_recurrent_policy = False
    return base


def _load_policy(args: argparse.Namespace, env: MECEnv) -> MECPolicy:
    device = torch.device("cpu")
    args.num_agents = env.num_agents
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device,
    )
    state_dict = torch.load(Path(args.model_dir) / "actor.pt", map_location=device)
    policy.actor.load_state_dict(state_dict)
    policy.actor.eval()
    return policy


def _rollout(env: MECEnv, policy: MECPolicy, args: argparse.Namespace, seed: int,
             *, render: bool = False, fps: int = 12, out: Path | None = None) -> dict:
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    obs_n = env._agent_obs(obs_dict)
    rnn = np.zeros((env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((env.num_agents, 1), dtype=np.float32)

    cov_r = access_coverage_radius_m(env.cfg)
    svc_r = backhaul_service_radius_m(env.cfg)
    horizon = int(env.cfg["base"]["episode_horizon_slots"])
    total_cost = 0.0
    total_reward = 0.0
    total_w1 = 0.0
    frames = []

    for step in range(horizon):
        ux = np.asarray(obs_dict["uavs"]["xy_m"], dtype=float)
        hot = np.asarray(obs_dict["demand"]["hotspot_center_m"], dtype=float)
        total_w1 += demand_matching_w1(ux, raw.grid_xy, raw._demand_density(hot)[0])

        with torch.no_grad():
            hv, uv, beta, rnn = _policy_action(policy, obs_n, rnn, masks, env)
        obs_dict, reward, _, truncated, info = raw.step(
            {"hap_velocity_mps": hv, "uav_velocity_mps": uv, "beta": beta}
        )
        obs_n = env._agent_obs(obs_dict)
        total_reward += float(reward)
        total_cost += float(info["training_cost"])
        if render:
            frames.append(_draw(raw, info, cov_r, svc_r, step, total_cost))
        if truncated:
            break

    if render and out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=fps, loop=0)

    steps = len(frames) if render else horizon
    return {
        "seed": int(seed),
        "total_cost": total_cost,
        "mean_cost_per_slot": total_cost / horizon,
        "total_reward": total_reward,
        "mean_w1_m": total_w1 / horizon,
        "steps": steps,
    }


def _ranked_gif_path(run_dir: Path, gif_name: str, rank: int, top_k: int) -> Path:
    path = Path(gif_name)
    if top_k <= 1:
        return run_dir / path
    return run_dir / f"top{rank}_{path.name}"


def _run_one(run_dir: Path, episodes: int, seed_stride: int, fps: int,
             gif_name: str, top_k: int) -> dict:
    model_dir = run_dir / "models"
    args = _load_args(model_dir)
    env = MECEnv(args)
    policy = _load_policy(args, env)
    base_seed = int(args.seed)
    seeds = [base_seed + seed_stride * i for i in range(episodes)]

    scores = sorted(
        (_rollout(env, policy, args, seed, render=False) for seed in seeds),
        key=lambda row: row["total_cost"],
    )
    winners = []
    for rank, score in enumerate(scores[:top_k], start=1):
        winner = dict(score)
        gif_path = _ranked_gif_path(run_dir, gif_name, rank, top_k)
        rendered = _rollout(env, policy, args, winner["seed"], render=True, fps=fps, out=gif_path)
        winner.update({"rank": rank, "gif": str(gif_path), "rendered_steps": rendered["steps"]})
        winners.append(winner)

    meta = {
        "selection": "lowest cumulative training_cost over deterministic policy eval episodes",
        "episodes": episodes,
        "top_k": top_k,
        "seed_stride": seed_stride,
        "base_seed": base_seed,
        "best": winners[0],
        "top": winners,
        "all_scores": scores,
    }
    meta_path = run_dir / (Path(gif_name).stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return {"run_dir": str(run_dir), "best": winners[0], "top": winners, "meta": str(meta_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("onpolicy/scripts/results/MEC/v3_iort_learnable/mappo"),
        help="mappo result root containing experiment/run directories",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed_stride", type=int, default=13)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif_name", default="best_eval_episode.gif")
    parser.add_argument("--top_k", type=int, default=1,
                        help="number of lowest-cost episodes to render per run")
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    results = []
    for rel in args.runs:
        run_dir = args.root / rel
        print(f"\n=== {run_dir} ===")
        result = _run_one(run_dir, args.episodes, args.seed_stride, args.fps,
                          args.gif_name, args.top_k)
        for winner in result["top"]:
            print(
                f"top{winner['rank']} seed={winner['seed']}  "
                f"mean_cost={winner['mean_cost_per_slot']:.4f}  "
                f"mean_w1={winner['mean_w1_m']:.1f}m  gif={winner['gif']}"
            )
        results.append(result)

    summary = args.root / "best_eval_episode_summary.json"
    summary.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote summary -> {summary}")


if __name__ == "__main__":
    main()
