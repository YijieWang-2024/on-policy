"""Evaluate whether a learned MEC policy is under-using flight speed.

The diagnostic keeps the learned direction and beta decisions fixed, then
multiplies deterministic HAP/UAV velocity commands by user-provided scale
factors before passing them to the environment.  The finite-K environment still
projects velocities to the physical speed limits.  If a larger scale improves
cost/W1 without causing safety or energy blow-ups, the policy is likely
under-exploring or has learned an overly conservative speed magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import ACT_DIM, MECEnv
from onpolicy.envs.mec.metrics import demand_matching_w1
from onpolicy.utils.run_config import (
    apply_legacy_mec_arch_default,
    apply_saved_args,
    load_model_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=100000)
    parser.add_argument("--seed_stride", type=int, default=13)
    parser.add_argument(
        "--uav_scales",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--hap_scales",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_policy(model_dir: str):
    parser = get_config()
    args = parser.parse_known_args(
        ["--env_name", "MEC", "--algorithm_name", "mappo"]
    )[0]
    saved = load_model_config(model_dir)
    apply_saved_args(args, saved, set(), skip={"model_dir"})
    apply_legacy_mec_arch_default(
        args, saved, set(), model_dir=model_dir
    )
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    device = torch.device("cpu")
    env = MECEnv(args)
    args.num_agents = env.num_agents
    policy = MECPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device,
    )
    policy.actor.load_state_dict(
        torch.load(Path(model_dir) / "actor.pt", map_location=device)
    )
    policy.actor.eval()
    return env, policy, args


def rollout(env: MECEnv, policy: MECPolicy, args, seed: int,
            *, uav_scale: float, hap_scale: float):
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    obs_n = env._agent_obs(obs_dict)
    n_agents = env.num_agents
    rnn = np.zeros(
        (n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    masks = np.ones((n_agents, 1), dtype=np.float32)
    horizon = int(env.cfg["base"]["episode_horizon_slots"])

    sums = {
        "cost": 0.0,
        "src": 0.0,
        "ovf": 0.0,
        "queue": 0.0,
        "energy": 0.0,
        "safety": 0.0,
        "accepted": 0.0,
        "source": 0.0,
        "w1": 0.0,
        "min_distance": 0.0,
        "safety_violations": 0.0,
        "uav_speed": 0.0,
        "hap_speed": 0.0,
        "first_quarter_w1": 0.0,
        "last_quarter_w1": 0.0,
    }
    quarter = max(horizon // 4, 1)

    for step in range(horizon):
        uav_xy = np.asarray(obs_dict["uavs"]["xy_m"], float)
        hotspot = np.asarray(obs_dict["demand"]["hotspot_center_m"], float)
        w1 = demand_matching_w1(
            uav_xy, raw.grid_xy, raw._demand_density(hotspot)[0]
        )
        sums["w1"] += w1
        if step < quarter:
            sums["first_quarter_w1"] += w1
        if step >= horizon - quarter:
            sums["last_quarter_w1"] += w1

        with torch.no_grad():
            action, rnn = policy.act(
                obs_n, rnn, masks, deterministic=True
            )
        action = action.detach().cpu().numpy().reshape(n_agents, ACT_DIM)
        hap_v = action[0, :2] * env.v_h_max * hap_scale
        uav_v = action[1:, :2] * env.v_u_max * uav_scale
        beta = np.clip(action[1:, 2], 0.0, 1.0)

        obs_dict, _, _, _, info = raw.step(
            {
                "hap_velocity_mps": hap_v,
                "uav_velocity_mps": uav_v,
                "beta": beta,
            }
        )
        obs_n = env._agent_obs(obs_dict)
        projected = info["projected_action"]
        projected_uav_v = np.asarray(projected["uav_velocity_mps"], float)
        projected_hap_v = np.asarray(projected["hap_velocity_mps"], float)

        sums["cost"] += float(info["training_cost"])
        sums["src"] += float(info["src_cost_component"])
        sums["ovf"] += float(info["ovf_cost_component"])
        sums["queue"] += float(info["queue_cost_component"])
        sums["energy"] += float(info["energy_cost_component"])
        sums["safety"] += float(info["safety_cost_component"])
        sums["accepted"] += float(np.sum(info["A_i"]))
        sums["source"] += float(info["U_src"])
        sums["min_distance"] += float(info["min_uav_distance_m"])
        sums["safety_violations"] += float(info["safety_violation_count"])
        sums["uav_speed"] += float(
            np.mean(np.linalg.norm(projected_uav_v, axis=1))
        )
        sums["hap_speed"] += float(np.linalg.norm(projected_hap_v))

    averaged = {key: value / horizon for key, value in sums.items()}
    averaged["first_quarter_w1"] = sums["first_quarter_w1"] / quarter
    averaged["last_quarter_w1"] = sums["last_quarter_w1"] / quarter
    offered = averaged["accepted"] + averaged["source"]
    averaged["accept_rate"] = (
        averaged["accepted"] / offered if offered > 0.0 else float("nan")
    )
    return averaged


def main():
    cli = parse_args()
    env, policy, args = load_policy(cli.model_dir)
    rows = []
    for hap_scale in cli.hap_scales:
        for uav_scale in cli.uav_scales:
            episodes = [
                rollout(
                    env,
                    policy,
                    args,
                    cli.eval_seed + cli.seed_stride * episode,
                    uav_scale=uav_scale,
                    hap_scale=hap_scale,
                )
                for episode in range(cli.episodes)
            ]
            row = {
                "hap_scale": float(hap_scale),
                "uav_scale": float(uav_scale),
            }
            for key in episodes[0]:
                row[key] = float(np.mean([ep[key] for ep in episodes]))
            rows.append(row)

    baseline = next(
        (
            row
            for row in rows
            if row["hap_scale"] == 1.0 and row["uav_scale"] == 1.0
        ),
        rows[0],
    )
    for row in rows:
        row["cost_delta_percent"] = (
            row["cost"] / baseline["cost"] - 1.0
        ) * 100.0
        row["w1_delta_percent"] = (
            row["w1"] / baseline["w1"] - 1.0
        ) * 100.0

    payload = {
        "model_dir": cli.model_dir,
        "episodes": cli.episodes,
        "eval_seed": cli.eval_seed,
        "seed_stride": cli.seed_stride,
        "rows": rows,
    }
    Path(cli.output).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
