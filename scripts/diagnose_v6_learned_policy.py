"""Matched-seed diagnostics for a trained v6 HAP/UAV MEC policy."""

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
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of using the deterministic policy mean",
    )
    parser.add_argument(
        "--projection_only",
        action="store_true",
        help="only run the policy rollout needed for action projection stats",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(model_dir: str):
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


def safe_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quartile_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = np.quantile(x, [0.25, 0.75])
    return float(np.mean(y[x >= hi]) - np.mean(y[x <= lo]))


def _projection_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {key: float(fn(arr)) for key, fn in {
        "mean": np.mean,
        "p50": np.median,
        "p95": lambda x: np.quantile(x, 0.95),
        "max": np.max,
    }.items()}


def rollout(env, policy, args, seed: int, mode: str, stochastic: bool):
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    obs_n = env._agent_obs(obs_dict)
    n_agents = env.num_agents
    rnn = np.zeros(
        (n_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    masks = np.ones((n_agents, 1), dtype=np.float32)
    horizon = int(env.cfg["base"]["episode_horizon_slots"])
    totals = {
        "cost": 0.0,
        "accepted": 0.0,
        "source": 0.0,
        "overflow": 0.0,
        "w1": 0.0,
        "backhaul_utilization": 0.0,
        "uav_compute_utilization": 0.0,
        "hub_compute_utilization": 0.0,
        "n_hotspot_uav": 0.0,
        "n_background_uav": 0.0,
        "hub_to_hotspot": 0.0,
    }
    traces = {
        "beta": [],
        "uav_queue": [],
        "hap_queue": [],
        "uav_speed": [],
        "uav_toward_hotspot_speed": [],
        "hap_speed": [],
        "hap_toward_centroid_speed": [],
        "hub_to_centroid": [],
        "hap_raw_norm": [],
        "uav_raw_norm": [],
        "hap_component_abs_max": [],
        "uav_component_abs_max": [],
        "hap_projection_delta": [],
        "uav_projection_delta": [],
        "beta_clip_delta": [],
    }

    for _ in range(horizon):
        uav_xy = np.asarray(obs_dict["uavs"]["xy_m"], float)
        hap_xy = np.asarray(obs_dict["hap"]["xy_m"], float)
        hotspot = np.asarray(obs_dict["demand"]["hotspot_center_m"], float)
        uav_queue = np.asarray(obs_dict["uavs"]["queue_bits"], float)
        hap_queue = float(obs_dict["hap"]["queue_bits"])
        totals["w1"] += demand_matching_w1(
            uav_xy, raw.grid_xy, raw._demand_density(hotspot)[0]
        )

        with torch.no_grad():
            action, rnn = policy.act(
                obs_n, rnn, masks, deterministic=not stochastic
            )
        action = action.detach().cpu().numpy().reshape(n_agents, ACT_DIM)
        raw_hap_unit = action[0, :2].astype(float)
        raw_uav_unit = action[1:, :2].astype(float)
        raw_beta = action[1:, 2].astype(float)
        hap_v = action[0, :2] * env.v_h_max
        uav_v = action[1:, :2] * env.v_u_max
        beta = np.clip(action[1:, 2], 0.0, 1.0)

        if mode == "freeze_hap":
            hap_v = np.zeros(2)
        elif mode == "hover_uav":
            uav_v = np.zeros_like(uav_v)
        elif mode == "beta_mean":
            beta = np.full_like(beta, np.mean(beta))

        to_hot = hotspot[None, :] - uav_xy
        to_hot_unit = to_hot / np.maximum(
            np.linalg.norm(to_hot, axis=1, keepdims=True), 1e-9
        )
        centroid = np.mean(uav_xy, axis=0)
        to_centroid = centroid - hap_xy
        to_centroid_unit = to_centroid / max(
            np.linalg.norm(to_centroid), 1e-9
        )
        traces["beta"].extend(beta.tolist())
        traces["uav_queue"].extend(uav_queue.tolist())
        traces["hap_queue"].extend([hap_queue] * env.k)
        traces["uav_speed"].extend(np.linalg.norm(uav_v, axis=1).tolist())
        traces["uav_toward_hotspot_speed"].extend(
            np.sum(uav_v * to_hot_unit, axis=1).tolist()
        )
        traces["hap_speed"].append(float(np.linalg.norm(hap_v)))
        traces["hap_toward_centroid_speed"].append(
            float(np.dot(hap_v, to_centroid_unit))
        )
        traces["hub_to_centroid"].append(
            float(np.linalg.norm(hap_xy - centroid))
        )

        obs_dict, reward, _, _, info = raw.step(
            {
                "hap_velocity_mps": hap_v,
                "uav_velocity_mps": uav_v,
                "beta": beta,
            }
        )
        projected = info["projected_action"]
        executed_hap_unit = (
            np.asarray(projected["hap_velocity_mps"], float)
            / max(env.v_h_max, 1e-9)
        )
        executed_uav_unit = (
            np.asarray(projected["uav_velocity_mps"], float)
            / max(env.v_u_max, 1e-9)
        )
        traces["hap_raw_norm"].append(float(np.linalg.norm(raw_hap_unit)))
        traces["uav_raw_norm"].extend(
            np.linalg.norm(raw_uav_unit, axis=1).tolist()
        )
        traces["hap_component_abs_max"].append(
            float(np.max(np.abs(raw_hap_unit)))
        )
        traces["uav_component_abs_max"].extend(
            np.max(np.abs(raw_uav_unit), axis=1).tolist()
        )
        traces["hap_projection_delta"].append(
            float(np.linalg.norm(raw_hap_unit - executed_hap_unit))
        )
        traces["uav_projection_delta"].extend(
            np.linalg.norm(raw_uav_unit - executed_uav_unit, axis=1).tolist()
        )
        traces["beta_clip_delta"].extend(np.abs(raw_beta - beta).tolist())
        obs_n = env._agent_obs(obs_dict)
        team = env._agent_infos(info, reward)[0]
        totals["cost"] += float(team["training_cost"])
        totals["accepted"] += float(team["accepted"])
        totals["source"] += float(team["U_src"])
        totals["overflow"] += float(team["overflow"])
        for key in (
            "backhaul_utilization",
            "uav_compute_utilization",
            "hub_compute_utilization",
            "n_hotspot_uav",
            "n_background_uav",
            "hub_to_hotspot",
        ):
            totals[key] += float(team[key])

    return (
        {key: value / horizon for key, value in totals.items()},
        traces,
    )


def main():
    cli = parse_args()
    env, policy, args = load(cli.model_dir)
    modes = (
        ("policy",)
        if cli.projection_only
        else ("policy", "freeze_hap", "hover_uav", "beta_mean")
    )
    results = {}
    baseline_traces = {key: [] for key in (
        "beta",
        "uav_queue",
        "hap_queue",
        "uav_speed",
        "uav_toward_hotspot_speed",
        "hap_speed",
        "hap_toward_centroid_speed",
        "hub_to_centroid",
        "hap_raw_norm",
        "uav_raw_norm",
        "hap_component_abs_max",
        "uav_component_abs_max",
        "hap_projection_delta",
        "uav_projection_delta",
        "beta_clip_delta",
    )}

    for mode in modes:
        episodes = []
        for i in range(cli.episodes):
            metrics, traces = rollout(
                env,
                policy,
                args,
                cli.eval_seed + 13 * i,
                mode,
                cli.stochastic,
            )
            episodes.append(metrics)
            if mode == "policy":
                for key, values in traces.items():
                    baseline_traces[key].extend(values)
        results[mode] = {
            key: float(np.mean([episode[key] for episode in episodes]))
            for key in episodes[0]
        }

    baseline = results["policy"]["cost"]
    for mode in modes[1:]:
        results[mode]["cost_delta_percent"] = (
            results[mode]["cost"] / baseline - 1.0
        ) * 100.0

    beta = np.asarray(baseline_traces["beta"])
    uav_queue = np.asarray(baseline_traces["uav_queue"])
    hap_queue = np.asarray(baseline_traces["hap_queue"])
    action_diagnostics = {
        "mean_beta": float(np.mean(beta)),
        "corr_beta_uav_queue": safe_corr(beta, uav_queue),
        "corr_beta_hap_queue": safe_corr(beta, hap_queue),
        "beta_high_minus_low_uav_queue": quartile_delta(uav_queue, beta),
        "beta_high_minus_low_hap_queue": quartile_delta(hap_queue, beta),
        "mean_uav_speed_mps": float(
            np.mean(baseline_traces["uav_speed"])
        ),
        "mean_uav_toward_hotspot_speed_mps": float(
            np.mean(baseline_traces["uav_toward_hotspot_speed"])
        ),
        "mean_hap_speed_mps": float(
            np.mean(baseline_traces["hap_speed"])
        ),
        "mean_hap_toward_centroid_speed_mps": float(
            np.mean(baseline_traces["hap_toward_centroid_speed"])
        ),
        "mean_hub_to_centroid_m": float(
            np.mean(baseline_traces["hub_to_centroid"])
        ),
        "major_sigma": np.exp(
            policy.actor.major_logstd.detach().cpu().numpy()
        ).tolist(),
        "minor_sigma": np.exp(
            policy.actor.minor_logstd.detach().cpu().numpy()
        ).tolist(),
    }
    projection_diagnostics = {
        "hap_norm_gt_1_rate": float(
            np.mean(np.asarray(baseline_traces["hap_raw_norm"]) > 1.0)
        ),
        "uav_norm_gt_1_rate": float(
            np.mean(np.asarray(baseline_traces["uav_raw_norm"]) > 1.0)
        ),
        "hap_component_abs_gt_1_rate": float(
            np.mean(
                np.asarray(baseline_traces["hap_component_abs_max"]) > 1.0
            )
        ),
        "uav_component_abs_gt_1_rate": float(
            np.mean(
                np.asarray(baseline_traces["uav_component_abs_max"]) > 1.0
            )
        ),
        "hap_projection_nonzero_rate": float(
            np.mean(
                np.asarray(baseline_traces["hap_projection_delta"]) > 1e-6
            )
        ),
        "uav_projection_nonzero_rate": float(
            np.mean(
                np.asarray(baseline_traces["uav_projection_delta"]) > 1e-6
            )
        ),
        "beta_clip_nonzero_rate": float(
            np.mean(np.asarray(baseline_traces["beta_clip_delta"]) > 1e-6)
        ),
        "hap_raw_norm": _projection_stats(baseline_traces["hap_raw_norm"]),
        "uav_raw_norm": _projection_stats(baseline_traces["uav_raw_norm"]),
        "hap_projection_delta": _projection_stats(
            baseline_traces["hap_projection_delta"]
        ),
        "uav_projection_delta": _projection_stats(
            baseline_traces["uav_projection_delta"]
        ),
        "beta_clip_delta": _projection_stats(
            baseline_traces["beta_clip_delta"]
        ),
    }
    output = {
        "model_dir": cli.model_dir,
        "episodes": cli.episodes,
        "eval_seed": cli.eval_seed,
        "action_mode": "stochastic" if cli.stochastic else "deterministic",
        "modes": results,
        "action_diagnostics": action_diagnostics,
        "projection_diagnostics": projection_diagnostics,
    }
    Path(cli.output).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
