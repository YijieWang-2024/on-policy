"""Diagnose whether per-UAV offloading decisions are identifiable and useful.

The diagnostic keeps the learned HAP/UAV motion fixed and changes only beta.
This separates offloading quality from trajectory quality and answers three
questions:

1. Does an adaptive beta controller beat the best fleet-wide constant beta?
2. Is there material finite-horizon value available to a clairvoyant beta oracle?
3. Does the learned beta head respond correctly to UAV queue, HAP queue, and
   backhaul geometry counterfactuals?

No training parameters or environment dynamics are modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import kl_divergence

from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import ACT_DIM, MECEnv
from onpolicy.envs.mec.metrics import demand_matching_w1
from onpolicy.utils.run_config import (
    apply_legacy_mec_arch_default,
    apply_saved_args,
    load_model_config,
)


@dataclass(frozen=True)
class MotionStep:
    hap_velocity_mps: np.ndarray
    uav_velocity_mps: np.ndarray


@dataclass(frozen=True)
class InitialDeployment:
    hap_xy_m: np.ndarray
    uav_xy_m: np.ndarray


@dataclass(frozen=True)
class BetaStepContext:
    accepted_bits: np.ndarray
    backhaul_rate_bps: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--seed_stride", type=int, default=13)
    parser.add_argument(
        "--constant_betas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--oracle_iterations", type=int, default=250)
    parser.add_argument("--oracle_lr", type=float, default=0.08)
    parser.add_argument("--counterfactual_stride", type=int, default=10)
    parser.add_argument("--counterfactual_max_rows", type=int, default=4096)
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


def parse_beta_grid(text: str) -> np.ndarray:
    values = np.asarray([float(item) for item in text.split(",")], dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("constant beta grid must contain finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("constant beta grid values must lie in [0, 1]")
    return np.unique(values)


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def queue_heuristic_beta(raw_env) -> np.ndarray:
    q = np.asarray(raw_env.state.uav_queue_bits, dtype=float)
    q_max = float(raw_env.cfg["env"]["uav"]["queue_max_bits"])
    return np.clip(q / max(0.2 * q_max, raw_env.eps), 0.0, 0.95)


def _policy_action(policy, obs_n, args, rnn):
    masks = np.ones((obs_n.shape[0], 1), dtype=np.float32)
    with torch.no_grad():
        action, next_rnn = policy.act(
            obs_n, rnn, masks, deterministic=True
        )
    return (
        action.detach().cpu().numpy().reshape(obs_n.shape[0], ACT_DIM),
        next_rnn,
    )


def record_learned_motion(
    env,
    policy,
    args,
    seed: int,
    *,
    counterfactual_stride: int,
    horizon: int | None = None,
):
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    obs_n = env._agent_obs(obs_dict)
    rnn = np.zeros(
        (env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    steps = []
    sampled_minor_obs = []
    limit = horizon or int(env.cfg["base"]["episode_horizon_slots"])

    for step_index in range(limit):
        action, rnn = _policy_action(policy, obs_n, args, rnn)
        motion = MotionStep(
            hap_velocity_mps=action[0, :2] * env.v_h_max,
            uav_velocity_mps=action[1:, :2] * env.v_u_max,
        )
        steps.append(motion)
        if counterfactual_stride > 0 and step_index % counterfactual_stride == 0:
            sampled_minor_obs.append(obs_n[1:].copy())
        obs_dict, _, _, _, _ = raw.step(
            {
                "hap_velocity_mps": motion.hap_velocity_mps,
                "uav_velocity_mps": motion.uav_velocity_mps,
                "beta": np.clip(action[1:, 2], 0.0, 1.0),
            }
        )
        obs_n = env._agent_obs(obs_dict)

    return steps, sampled_minor_obs


def prepare_beta_step(raw, motion: MotionStep) -> BetaStepContext:
    st = raw.state
    hap_v, uav_v, _ = raw._project_action(
        {
            "hap_velocity_mps": motion.hap_velocity_mps,
            "uav_velocity_mps": motion.uav_velocity_mps,
            "beta": np.zeros(raw.k),
        }
    )
    next_hap_xy = raw._clip_xy(st.hap_xy_m + raw.delta * hap_v)
    next_uav_xy = raw._clip_xy(st.uav_xy_m + raw.delta * uav_v)
    service_hap_xy, service_uav_xy = raw._service_positions(
        st.hap_xy_m,
        st.uav_xy_m,
        next_hap_xy,
        next_uav_xy,
    )
    active_density, fresh_density = raw._demand_density(st.demand_center_m)
    access_gain = raw._access_gain(service_uav_xy)
    _, phi = raw._service_share(access_gain)

    if raw._uses_continuous_workload():
        demand_i = (
            np.sum(phi * fresh_density[None, :], axis=1) * raw.cell_area
        )
        access_rate = raw._continuous_access_rate(
            access_gain, phi, fresh_density, demand_i
        )
        accepted = np.minimum(demand_i, raw.delta * access_rate)
    else:
        n_srv = (
            np.sum(phi * active_density[None, :], axis=1) * raw.cell_area
        )
        access_rate = raw._access_rate(access_gain, n_srv)
        packet_bits = float(raw.cfg["demand"]["packet_size_bits"])
        accepted = np.sum(
            phi
            * active_density[None, :]
            * np.minimum(packet_bits, raw.delta * access_rate),
            axis=1,
        ) * raw.cell_area

    return BetaStepContext(
        accepted_bits=accepted,
        backhaul_rate_bps=raw._backhaul_rate(
            service_hap_xy, service_uav_xy
        ),
    )


def precompute_beta_exogenous(
    env,
    seed: int,
    motion_steps: list[MotionStep],
    advance_beta: float,
    initial_deployment: InitialDeployment | None = None,
):
    """Collect beta-independent arrivals/rates along one fixed motion path."""
    raw = env.env
    raw.reset(seed=seed)
    apply_initial_deployment(raw, initial_deployment)
    initial_uav_queue = raw.state.uav_queue_bits.copy()
    initial_hap_queue = float(raw.state.hap_queue_bits)
    contexts = []
    for motion in motion_steps:
        context = prepare_beta_step(raw, motion)
        contexts.append(context)
        raw.step(
            {
                "hap_velocity_mps": motion.hap_velocity_mps,
                "uav_velocity_mps": motion.uav_velocity_mps,
                "beta": np.full(raw.k, advance_beta, dtype=float),
            }
        )
    return initial_uav_queue, initial_hap_queue, contexts


def apply_initial_deployment(raw, deployment):
    if deployment is None:
        return
    raw.state.hap_xy_m = np.asarray(
        deployment.hap_xy_m, dtype=float
    ).copy()
    raw.state.uav_xy_m = np.asarray(
        deployment.uav_xy_m, dtype=float
    ).copy()


def beta_schedule_objective(
    raw,
    initial_uav_queue,
    initial_hap_queue,
    contexts,
    beta_schedule,
):
    """Differentiable exact beta-dependent finite-horizon team cost."""
    dtype = beta_schedule.dtype
    device = beta_schedule.device
    q_u = torch.as_tensor(
        initial_uav_queue, dtype=dtype, device=device
    )
    q_h = torch.as_tensor(initial_hap_queue, dtype=dtype, device=device)
    accepted = torch.as_tensor(
        np.stack([context.accepted_bits for context in contexts]),
        dtype=dtype,
        device=device,
    )
    backhaul_rate = torch.as_tensor(
        np.stack([context.backhaul_rate_bps for context in contexts]),
        dtype=dtype,
        device=device,
    )
    cfg = raw.cfg
    delta = float(raw.delta)
    c_u = torch.as_tensor(
        float(cfg["derived"]["uav_compute_capacity_bits"]),
        dtype=dtype,
        device=device,
    )
    c_h = torch.as_tensor(
        float(cfg["derived"]["hap_compute_capacity_bits"]),
        dtype=dtype,
        device=device,
    )
    q_u_max = torch.as_tensor(
        float(cfg["env"]["uav"]["queue_max_bits"]),
        dtype=dtype,
        device=device,
    )
    q_h_max = torch.as_tensor(
        float(cfg["env"]["hap"]["queue_max_bits"]),
        dtype=dtype,
        device=device,
    )
    weights = cfg["cost"]["weights"]
    omega_queue = float(weights["omega_queue_per_bit"])
    omega_ovf = float(weights["omega_ovf_per_bit"])
    omega_energy = float(weights["omega_energy_per_j"])
    cycles = float(cfg["compute"]["cycles_per_bit"])
    uav_compute_coeff = (
        float(cfg["compute"]["kappa_uav"]) * cycles**3 / delta**2
    )
    hap_compute_coeff = (
        float(cfg["compute"]["kappa_hap"]) * cycles**3 / delta**2
    )
    tx_power_w = 10.0 ** (
        (
            float(
                cfg["communication"]["backhaul"]["mmwave"][
                    "tx_power_dbm"
                ]
            )
            - 30.0
        )
        / 10.0
    )
    total = torch.zeros((), dtype=dtype, device=device)

    for step in range(beta_schedule.shape[0]):
        beta = beta_schedule[step]
        total = total + omega_queue * (q_u.sum() + q_h)
        s_u = torch.minimum((1.0 - beta) * q_u, c_u)
        b_i = torch.minimum(
            beta * q_u, delta * backhaul_rate[step]
        )
        residual_uav = torch.clamp(q_u - s_u - b_i, min=0.0)
        uav_unclipped = residual_uav + accepted[step]
        d_u = torch.clamp(uav_unclipped - q_u_max, min=0.0)
        q_u = torch.minimum(uav_unclipped, q_u_max)

        s_h = torch.minimum(q_h, c_h)
        h_unclipped = torch.clamp(q_h - s_h, min=0.0) + b_i.sum()
        d_h = torch.clamp(h_unclipped - q_h_max, min=0.0)
        q_h = torch.minimum(h_unclipped, q_h_max)

        compute_energy = (
            uav_compute_coeff * torch.sum(s_u**3)
            + hap_compute_coeff * s_h**3
        )
        tx_energy = torch.sum(
            torch.where(
                backhaul_rate[step] > raw.eps,
                tx_power_w
                * b_i
                / torch.clamp(backhaul_rate[step], min=raw.eps),
                torch.zeros_like(b_i),
            )
        )
        # Flight energy, source loss, and safety are constant under fixed motion.
        # Omitting them preserves the exact ordering of beta schedules.
        total = (
            total
            + omega_ovf * (d_u.sum() + d_h)
            + omega_energy * (compute_energy + tx_energy)
        )

    return total / beta_schedule.shape[0]


def optimize_clairvoyant_beta_schedule(
    raw,
    initial_uav_queue,
    initial_hap_queue,
    contexts,
    initial_beta: float,
    iterations: int,
    learning_rate: float,
):
    """Optimize the full beta schedule while keeping motion/exogenous data fixed."""
    if iterations < 1:
        raise ValueError("oracle_iterations must be positive")
    if learning_rate <= 0.0:
        raise ValueError("oracle_lr must be positive")
    clipped = float(np.clip(initial_beta, 1e-4, 1.0 - 1e-4))
    initial_logit = np.log(clipped / (1.0 - clipped))
    logits = torch.nn.Parameter(
        torch.full(
            (len(contexts), raw.k),
            initial_logit,
            dtype=torch.float64,
        )
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    best_loss = float("inf")
    best_schedule = None

    for _ in range(iterations):
        optimizer.zero_grad()
        schedule = torch.sigmoid(logits)
        loss = beta_schedule_objective(
            raw,
            initial_uav_queue,
            initial_hap_queue,
            contexts,
            schedule,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits.clamp_(-9.0, 9.0)
            value = float(loss.item())
            if value < best_loss:
                best_loss = value
                best_schedule = schedule.detach().cpu().numpy().copy()

    constant_schedule = np.full(
        (len(contexts), raw.k), initial_beta, dtype=float
    )
    with torch.no_grad():
        constant_loss = float(
            beta_schedule_objective(
                raw,
                initial_uav_queue,
                initial_hap_queue,
                contexts,
                torch.as_tensor(constant_schedule, dtype=torch.float64),
            ).item()
        )
    if constant_loss <= best_loss:
        return constant_schedule, constant_loss
    return best_schedule, best_loss


def _empty_trace():
    return {
        "beta": [],
        "uav_queue": [],
        "hap_queue": [],
        "backhaul_rate": [],
        "fleet_beta_std": [],
    }


def _summarize_trace(trace):
    beta = np.asarray(trace["beta"], dtype=float)
    return {
        "mean_beta": float(np.mean(beta)),
        "beta_std_all": float(np.std(beta)),
        "mean_fleet_beta_std": float(np.mean(trace["fleet_beta_std"])),
        "corr_beta_uav_queue": safe_corr(beta, trace["uav_queue"]),
        "corr_beta_hap_queue": safe_corr(beta, trace["hap_queue"]),
        "corr_beta_backhaul_rate": safe_corr(
            beta, trace["backhaul_rate"]
        ),
    }


def run_fixed_motion_episode(
    env,
    policy,
    args,
    seed: int,
    motion_steps: list[MotionStep],
    controller: str,
    *,
    constant_beta: float | None = None,
    beta_schedule: np.ndarray | None = None,
    initial_deployment: InitialDeployment | None = None,
):
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    apply_initial_deployment(raw, initial_deployment)
    obs_dict = raw._observation()
    obs_n = env._agent_obs(obs_dict)
    rnn = np.zeros(
        (env.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    totals = {
        "cost": 0.0,
        "source_cost": 0.0,
        "overflow_cost": 0.0,
        "queue_cost": 0.0,
        "energy_cost": 0.0,
        "beta_sensitive_cost": 0.0,
        "accepted": 0.0,
        "offloaded": 0.0,
        "source": 0.0,
        "overflow": 0.0,
        "w1": 0.0,
    }
    trace = _empty_trace()

    for step_index, motion in enumerate(motion_steps):
        uav_xy = np.asarray(obs_dict["uavs"]["xy_m"], dtype=float)
        hotspot = np.asarray(
            obs_dict["demand"]["hotspot_center_m"], dtype=float
        )
        totals["w1"] += demand_matching_w1(
            uav_xy, raw.grid_xy, raw._demand_density(hotspot)[0]
        )

        policy_action = None
        if controller == "policy":
            policy_action, rnn = _policy_action(policy, obs_n, args, rnn)
        if controller == "policy":
            beta = np.clip(policy_action[1:, 2], 0.0, 1.0)
        elif controller == "constant":
            beta = np.full(env.k, float(constant_beta), dtype=float)
        elif controller == "queue_heuristic":
            beta = queue_heuristic_beta(raw)
        elif controller == "schedule":
            beta = np.asarray(beta_schedule[step_index], dtype=float)
        else:
            raise ValueError(f"unknown beta controller: {controller}")

        context = prepare_beta_step(raw, motion)
        pre_uav_queue = np.asarray(raw.state.uav_queue_bits, dtype=float)
        pre_hap_queue = float(raw.state.hap_queue_bits)
        trace["beta"].extend(beta.tolist())
        trace["uav_queue"].extend(pre_uav_queue.tolist())
        trace["hap_queue"].extend([pre_hap_queue] * env.k)
        trace["backhaul_rate"].extend(
            context.backhaul_rate_bps.tolist()
        )
        trace["fleet_beta_std"].append(float(np.std(beta)))

        obs_dict, reward, _, _, info = raw.step(
            {
                "hap_velocity_mps": motion.hap_velocity_mps,
                "uav_velocity_mps": motion.uav_velocity_mps,
                "beta": beta,
            }
        )
        obs_n = env._agent_obs(obs_dict)
        totals["cost"] += float(info["training_cost"])
        totals["source_cost"] += float(info["src_cost_component"])
        totals["overflow_cost"] += float(info["ovf_cost_component"])
        totals["queue_cost"] += float(info["queue_cost_component"])
        totals["energy_cost"] += float(info["energy_cost_component"])
        totals["beta_sensitive_cost"] += (
            float(info["queue_cost_component"])
            + float(info["ovf_cost_component"])
            + float(info["energy_cost_component"])
        )
        totals["accepted"] += float(np.sum(info["A_i"]))
        totals["offloaded"] += float(np.sum(info["B_i"]))
        totals["source"] += float(info["U_src"])
        totals["overflow"] += (
            float(np.sum(info["D_i_U"])) + float(info["D_H"])
        )

    horizon = len(motion_steps)
    return (
        {key: value / horizon for key, value in totals.items()},
        trace,
    )


def evaluate_controller(
    env,
    policy,
    args,
    seeds,
    motion_by_seed,
    controller,
    deployment_by_seed=None,
    **kwargs,
):
    episodes = []
    combined_trace = _empty_trace()
    for seed in seeds:
        metrics, trace = run_fixed_motion_episode(
            env,
            policy,
            args,
            seed,
            motion_by_seed[seed],
            controller,
            initial_deployment=(
                None
                if deployment_by_seed is None
                else deployment_by_seed[seed]
            ),
            **kwargs,
        )
        episodes.append(metrics)
        for key, values in trace.items():
            combined_trace[key].extend(values)
    summary = {
        key: float(np.mean([episode[key] for episode in episodes]))
        for key in episodes[0]
    }
    summary.update(_summarize_trace(combined_trace))
    return summary


def evaluate_clairvoyant_schedules(
    env,
    policy,
    args,
    seeds,
    motion_by_seed,
    schedules,
    best_constant_beta,
    deployment_by_seed=None,
):
    episodes = []
    combined_trace = _empty_trace()
    fallback_count = 0
    for seed in seeds:
        optimized_metrics, optimized_trace = run_fixed_motion_episode(
            env,
            policy,
            args,
            seed,
            motion_by_seed[seed],
            "schedule",
            beta_schedule=schedules[seed],
            initial_deployment=(
                None
                if deployment_by_seed is None
                else deployment_by_seed[seed]
            ),
        )
        constant_metrics, constant_trace = run_fixed_motion_episode(
            env,
            policy,
            args,
            seed,
            motion_by_seed[seed],
            "constant",
            constant_beta=best_constant_beta,
            initial_deployment=(
                None
                if deployment_by_seed is None
                else deployment_by_seed[seed]
            ),
        )
        if constant_metrics["cost"] < optimized_metrics["cost"]:
            metrics, trace = constant_metrics, constant_trace
            fallback_count += 1
        else:
            metrics, trace = optimized_metrics, optimized_trace
        episodes.append(metrics)
        for key, values in trace.items():
            combined_trace[key].extend(values)
    summary = {
        key: float(np.mean([episode[key] for episode in episodes]))
        for key in episodes[0]
    }
    summary.update(_summarize_trace(combined_trace))
    summary["constant_fallback_episodes"] = fallback_count
    return summary


def _beta_distribution(policy, args, obs_rows):
    obs = torch.as_tensor(obs_rows, dtype=torch.float32)
    rnn = torch.zeros(
        (obs.shape[0], args.recurrent_N, args.hidden_size),
        dtype=torch.float32,
    )
    masks = torch.ones((obs.shape[0], 1), dtype=torch.float32)
    with torch.no_grad():
        features, _ = policy.actor._features(obs, rnn, masks)
        _, _, distribution = policy.actor._dists(features)
    return distribution


def _paired_response(low_dist, high_dist, *, expected_sign: int):
    delta = (high_dist.mean - low_dist.mean).cpu().numpy().reshape(-1)
    signed = expected_sign * delta
    return {
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "expected_sign_fraction": float(np.mean(signed > 0.0)),
        "mean_kl_low_to_high": float(
            kl_divergence(low_dist, high_dist).mean().item()
        ),
    }


def counterfactual_beta_diagnostics(
    policy,
    args,
    minor_obs_rows,
    fleet_size: int,
    max_rows: int,
):
    rows = np.concatenate(minor_obs_rows, axis=0)
    if rows.shape[0] > max_rows:
        indices = np.linspace(
            0, rows.shape[0] - 1, max_rows, dtype=int
        )
        rows = rows[indices]

    baseline = _beta_distribution(policy, args, rows)
    own_low = rows.copy()
    own_high = rows.copy()
    original_own_queue = rows[:, 3].copy()
    own_low[:, 3] = 0.2
    own_high[:, 3] = 0.8
    own_low[:, 13] = np.clip(
        own_low[:, 13] + (0.2 - original_own_queue) / fleet_size,
        0.0,
        1.0,
    )
    own_high[:, 13] = np.clip(
        own_high[:, 13] + (0.8 - original_own_queue) / fleet_size,
        0.0,
        1.0,
    )

    hap_low = rows.copy()
    hap_high = rows.copy()
    hap_low[:, 6] = 0.2
    hap_high[:, 6] = 0.8

    close_hap = rows.copy()
    far_hap = rows.copy()
    own_xy = rows[:, 1:3]
    close_hap[:, 4:6] = own_xy
    corners = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    corner_distances = np.linalg.norm(
        own_xy[:, None, :] - corners[None, :, :], axis=2
    )
    far_hap[:, 4:6] = corners[np.argmax(corner_distances, axis=1)]

    baseline_mean = baseline.mean.cpu().numpy().reshape(-1)
    baseline_entropy = baseline.entropy().cpu().numpy().reshape(-1)
    concentration = (
        baseline.concentration1 + baseline.concentration0
    ).cpu().numpy().reshape(-1)
    return {
        "sample_rows": int(rows.shape[0]),
        "baseline_mean_beta": float(np.mean(baseline_mean)),
        "baseline_mean_entropy": float(np.mean(baseline_entropy)),
        "baseline_mean_concentration": float(np.mean(concentration)),
        "baseline_saturation_fraction": float(
            np.mean((baseline_mean < 0.05) | (baseline_mean > 0.95))
        ),
        "own_queue_high_minus_low": _paired_response(
            _beta_distribution(policy, args, own_low),
            _beta_distribution(policy, args, own_high),
            expected_sign=1,
        ),
        "hap_queue_high_minus_low": _paired_response(
            _beta_distribution(policy, args, hap_low),
            _beta_distribution(policy, args, hap_high),
            expected_sign=-1,
        ),
        "close_minus_far_hap": _paired_response(
            _beta_distribution(policy, args, far_hap),
            _beta_distribution(policy, args, close_hap),
            expected_sign=1,
        ),
    }


def relative_gain(reference_cost: float, candidate_cost: float) -> float:
    return 100.0 * (reference_cost - candidate_cost) / reference_cost


def main():
    cli = parse_args()
    env, policy, args = load_policy(cli.model_dir)
    seeds = [
        cli.eval_seed + cli.seed_stride * index
        for index in range(cli.episodes)
    ]
    constant_grid = parse_beta_grid(cli.constant_betas)

    motion_by_seed = {}
    counterfactual_rows = []
    for seed in seeds:
        motion, sampled_rows = record_learned_motion(
            env,
            policy,
            args,
            seed,
            counterfactual_stride=cli.counterfactual_stride,
        )
        motion_by_seed[seed] = motion
        counterfactual_rows.extend(sampled_rows)

    constant_scan = {}
    for beta in constant_grid:
        constant_scan[f"{beta:.6g}"] = evaluate_controller(
            env,
            policy,
            args,
            seeds,
            motion_by_seed,
            "constant",
            constant_beta=float(beta),
        )
    best_constant_key = min(
        constant_scan, key=lambda key: constant_scan[key]["cost"]
    )
    best_constant_beta = float(best_constant_key)
    best_constant = constant_scan[best_constant_key]
    oracle_schedules = {}
    oracle_surrogate_costs = {}
    for seed in seeds:
        initial_uav_queue, initial_hap_queue, contexts = (
            precompute_beta_exogenous(
                env,
                seed,
                motion_by_seed[seed],
                best_constant_beta,
            )
        )
        schedule, surrogate_cost = optimize_clairvoyant_beta_schedule(
            env.env,
            initial_uav_queue,
            initial_hap_queue,
            contexts,
            best_constant_beta,
            cli.oracle_iterations,
            cli.oracle_lr,
        )
        oracle_schedules[seed] = schedule
        oracle_surrogate_costs[str(seed)] = surrogate_cost

    controllers = {
        "learned": evaluate_controller(
            env,
            policy,
            args,
            seeds,
            motion_by_seed,
            "policy",
        ),
        "queue_heuristic": evaluate_controller(
            env,
            policy,
            args,
            seeds,
            motion_by_seed,
            "queue_heuristic",
        ),
        "clairvoyant_oracle": evaluate_clairvoyant_schedules(
            env,
            policy,
            args,
            seeds,
            motion_by_seed,
            oracle_schedules,
            best_constant_beta,
        ),
        "best_constant": best_constant,
    }
    for name, result in controllers.items():
        if name != "best_constant":
            result["gain_vs_best_constant_percent"] = relative_gain(
                best_constant["cost"], result["cost"]
            )
            result[
                "gain_vs_best_constant_beta_sensitive_percent"
            ] = relative_gain(
                best_constant["beta_sensitive_cost"],
                result["beta_sensitive_cost"],
            )

    counterfactual = counterfactual_beta_diagnostics(
        policy,
        args,
        counterfactual_rows,
        env.k,
        cli.counterfactual_max_rows,
    )
    output = {
        "model_dir": cli.model_dir,
        "scenario": args.mec_scenario,
        "episodes": cli.episodes,
        "eval_seed": cli.eval_seed,
        "seed_stride": cli.seed_stride,
        "protocol": (
            "fixed learned HAP/UAV motion; beta controller remains closed-loop"
        ),
        "best_constant_beta": best_constant_beta,
        "constant_scan": constant_scan,
        "oracle_surrogate_cost_by_seed": oracle_surrogate_costs,
        "controllers": controllers,
        "counterfactual_policy_response": counterfactual,
        "interpretation_thresholds": {
            "adaptive_gain_material_percent": 2.0,
            "oracle_gain_material_percent": 5.0,
            "counterfactual_expected_sign_fraction": 0.7,
        },
    }
    output_path = Path(cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
