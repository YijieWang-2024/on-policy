"""Sweep physical regimes for per-UAV beta identifiability.

The sweep never edits scenario YAML files. It records one trained policy's
HAP/UAV motion in the baseline scenario, applies that exact motion to perturbed
in-memory configs, and compares:

- the best fleet-wide constant beta;
- a finite-horizon clairvoyant per-UAV beta schedule.

This locates regimes where adaptive offloading has enough physical value to
produce a learnable signal before changing the policy architecture.
"""

from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from onpolicy.envs.mec.config_loader import (
    derive_constants,
    derive_cost_weights,
    validate_scenario,
)
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv
from onpolicy.scripts.analysis.diagnose_beta_identifiability import (
    InitialDeployment,
    MotionStep,
    evaluate_clairvoyant_schedules,
    evaluate_controller,
    load_policy,
    optimize_clairvoyant_beta_schedule,
    parse_beta_grid,
    precompute_beta_exogenous,
    record_learned_motion,
    relative_gain,
)
from onpolicy.scripts.analysis.design_v6_sanity import (
    _background_points,
    _sunflower,
)


@dataclass(frozen=True)
class StressCase:
    name: str
    family: str
    workload_scale: float = 1.0
    uav_cpu_scale: float = 1.0
    hap_cpu_scale: float = 1.0
    uav_queue_scale: float = 1.0
    hap_queue_scale: float = 1.0
    backhaul_margin_delta_db: float = 0.0
    backhaul_pathloss_delta: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--seed_stride", type=int, default=13)
    parser.add_argument(
        "--constant_betas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--oracle_iterations", type=int, default=60)
    parser.add_argument("--oracle_lr", type=float, default=0.08)
    parser.add_argument(
        "--preset", choices=("coarse", "extended"), default="coarse"
    )
    parser.add_argument(
        "--motion_profiles",
        default="learned,heuristic,matched_hold,hover",
        help=(
            "comma-separated subset of "
            "learned,heuristic,matched_hold,hover"
        ),
    )
    parser.add_argument(
        "--matched_hot_counts",
        default="4,5,6,7,8",
        help="candidate hotspot UAV counts for matched_hold",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="optional comma-separated stress case names",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def stress_cases(preset: str) -> list[StressCase]:
    cases = [
        StressCase("baseline", "baseline"),
        StressCase(
            "uav_cpu_x0.5", "uav_cpu", uav_cpu_scale=0.5
        ),
        StressCase(
            "hap_cpu_x0.5", "hap_cpu", hap_cpu_scale=0.5
        ),
        StressCase(
            "workload_x1.25", "workload", workload_scale=1.25
        ),
        StressCase(
            "workload_x1.5", "workload", workload_scale=1.5
        ),
        StressCase(
            "bh_margin_plus3db",
            "backhaul",
            backhaul_margin_delta_db=3.0,
        ),
        StressCase(
            "bh_margin_plus6db",
            "backhaul",
            backhaul_margin_delta_db=6.0,
        ),
        StressCase(
            "bh_pathloss_plus0.2",
            "backhaul",
            backhaul_pathloss_delta=0.2,
        ),
        StressCase(
            "uav_queue_x0.5", "queue", uav_queue_scale=0.5
        ),
        StressCase(
            "hap_queue_x0.5", "queue", hap_queue_scale=0.5
        ),
        StressCase(
            "workload_x1.25_uav_cpu_x0.75",
            "interaction",
            workload_scale=1.25,
            uav_cpu_scale=0.75,
        ),
        StressCase(
            "workload_x1.25_bh_margin_plus3db",
            "interaction",
            workload_scale=1.25,
            backhaul_margin_delta_db=3.0,
        ),
    ]
    if preset == "extended":
        cases.extend(
            [
                StressCase(
                    "uav_cpu_x0.75", "uav_cpu", uav_cpu_scale=0.75
                ),
                StressCase(
                    "hap_cpu_x0.75", "hap_cpu", hap_cpu_scale=0.75
                ),
                StressCase(
                    "bh_pathloss_plus0.4",
                    "backhaul",
                    backhaul_pathloss_delta=0.4,
                ),
                StressCase(
                    "uav_queue_x0.75",
                    "queue",
                    uav_queue_scale=0.75,
                ),
                StressCase(
                    "hap_queue_x0.75",
                    "queue",
                    hap_queue_scale=0.75,
                ),
                StressCase(
                    "workload_x1.5_uav_cpu_x0.75",
                    "interaction",
                    workload_scale=1.5,
                    uav_cpu_scale=0.75,
                ),
                StressCase(
                    "workload_x1.25_hap_cpu_x0.75",
                    "interaction",
                    workload_scale=1.25,
                    hap_cpu_scale=0.75,
                ),
                StressCase(
                    "uav_cpu_x0.75_bh_margin_plus3db",
                    "interaction",
                    uav_cpu_scale=0.75,
                    backhaul_margin_delta_db=3.0,
                ),
                StressCase(
                    "workload_x1.10_uav_cpu_x0.90",
                    "matched_hold_refinement",
                    workload_scale=1.10,
                    uav_cpu_scale=0.90,
                ),
                StressCase(
                    "workload_x1.15_uav_cpu_x0.90",
                    "matched_hold_refinement",
                    workload_scale=1.15,
                    uav_cpu_scale=0.90,
                ),
                StressCase(
                    "workload_x1.15_uav_cpu_x0.85",
                    "matched_hold_refinement",
                    workload_scale=1.15,
                    uav_cpu_scale=0.85,
                ),
                StressCase(
                    "workload_x1.20_uav_cpu_x0.90",
                    "matched_hold_refinement",
                    workload_scale=1.20,
                    uav_cpu_scale=0.90,
                ),
                StressCase(
                    "workload_x1.20_uav_cpu_x0.85",
                    "matched_hold_refinement",
                    workload_scale=1.20,
                    uav_cpu_scale=0.85,
                ),
                StressCase(
                    "workload_x1.25_uav_cpu_x0.85",
                    "matched_hold_refinement",
                    workload_scale=1.25,
                    uav_cpu_scale=0.85,
                ),
            ]
        )
    return cases


def parse_motion_profiles(text: str) -> tuple[str, ...]:
    profiles = tuple(
        item.strip() for item in text.split(",") if item.strip()
    )
    allowed = {"learned", "heuristic", "matched_hold", "hover"}
    if not profiles or any(profile not in allowed for profile in profiles):
        raise ValueError(
            "motion profiles must be a non-empty subset of "
            "learned,heuristic,matched_hold,hover"
        )
    return tuple(dict.fromkeys(profiles))


def select_stress_cases(
    available: list[StressCase], selection: str | None
) -> list[StressCase]:
    if selection is None:
        return available
    requested = [
        item.strip() for item in selection.split(",") if item.strip()
    ]
    by_name = {case.name: case for case in available}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(
            f"unknown stress cases: {', '.join(missing)}"
        )
    return [by_name[name] for name in dict.fromkeys(requested)]


def parse_hot_counts(text: str, fleet_size: int) -> tuple[int, ...]:
    counts = tuple(
        int(item.strip()) for item in text.split(",") if item.strip()
    )
    if not counts or any(count < 0 or count > fleet_size for count in counts):
        raise ValueError(
            "matched hotspot counts must lie between 0 and fleet size"
        )
    return tuple(dict.fromkeys(counts))


def _hotspot_sigma_m(cfg):
    field = cfg["demand"].get("workload_field")
    if field is not None and "hotspot_sigma_m" in field:
        return float(field["hotspot_sigma_m"])
    return float(cfg["demand"]["activity_probability"]["hotspot_sigma_m"])


def record_reference_motion(env, seed: int, profile: str):
    """Record queue-independent heuristic or hover motion."""
    if profile not in {"heuristic", "hover"}:
        raise ValueError(f"unsupported reference motion profile: {profile}")
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    horizon = int(env.cfg["base"]["episode_horizon_slots"])
    motions = []
    k = env.k
    sigma = _hotspot_sigma_m(env.cfg)
    limits = np.asarray(
        [
            float(env.cfg["env"]["region"]["lx_m"]),
            float(env.cfg["env"]["region"]["ly_m"]),
        ]
    )
    radii = 1.5 * sigma * np.sqrt((np.arange(k) + 0.5) / k)
    angles = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(k)
    offsets = np.stack(
        [radii * np.cos(angles), radii * np.sin(angles)], axis=1
    )

    for _ in range(horizon):
        if profile == "hover":
            hap_velocity = np.zeros(2)
            uav_velocity = np.zeros((k, 2))
        else:
            uav_xy = np.asarray(obs_dict["uavs"]["xy_m"], dtype=float)
            hap_xy = np.asarray(obs_dict["hap"]["xy_m"], dtype=float)
            hotspot = np.asarray(
                obs_dict["demand"]["hotspot_center_m"], dtype=float
            )
            targets = np.clip(hotspot + offsets, 0.0, limits)
            uav_velocity = np.clip(
                targets - uav_xy, -env.v_u_max, env.v_u_max
            )
            hap_velocity = np.clip(
                np.mean(uav_xy, axis=0) - hap_xy,
                -env.v_h_max,
                env.v_h_max,
            )
        motion = MotionStep(
            hap_velocity_mps=hap_velocity,
            uav_velocity_mps=uav_velocity,
        )
        motions.append(motion)
        obs_dict, _, _, _, _ = raw.step(
            {
                "hap_velocity_mps": hap_velocity,
                "uav_velocity_mps": uav_velocity,
                "beta": np.full(k, 0.5, dtype=float),
            }
        )
    return motions


def matched_deployment(
    cfg: dict, center: np.ndarray, n_hot: int
) -> InitialDeployment:
    k = int(cfg["env"]["fleet_size_k"])
    region = float(cfg["env"]["region"]["lx_m"])
    sigma = float(cfg["demand"]["workload_field"]["hotspot_sigma_m"])
    hot = _sunflower(center, n_hot, 0.9 * sigma)
    background = _background_points(
        center, k - n_hot, sigma, region
    )
    uav_xy = np.clip(
        np.vstack([hot, background]), 0.0, region
    )
    # The HAP sits at the UAV centroid, the natural backhaul geometry center.
    hap_xy = np.mean(uav_xy, axis=0)
    return InitialDeployment(hap_xy_m=hap_xy, uav_xy_m=uav_xy)


def matched_deployment_score(cfg, center, deployment) -> float:
    """One-step beta-independent source loss for deployment selection."""
    raw = FiniteKHAPUAVMECEnv(cfg)
    raw.reset(seed=0)
    raw.state.demand_center_m = np.asarray(center, dtype=float).copy()
    raw.state.demand_velocity_mps[:] = 0.0
    raw.state.hap_xy_m = deployment.hap_xy_m.copy()
    raw.state.uav_xy_m = deployment.uav_xy_m.copy()
    _, _, _, _, info = raw.step(
        {
            "hap_velocity_mps": np.zeros(2),
            "uav_velocity_mps": np.zeros((raw.k, 2)),
            "beta": np.zeros(raw.k),
        }
    )
    return float(info["U_src"])


def select_matched_deployment(cfg, center, hot_counts):
    scored = []
    for n_hot in hot_counts:
        deployment = matched_deployment(cfg, center, n_hot)
        scored.append(
            (
                matched_deployment_score(cfg, center, deployment),
                n_hot,
                deployment,
            )
        )
    _, best_count, best_deployment = min(scored, key=lambda row: row[0])
    return best_deployment, best_count


def apply_stress_case(base_cfg: dict, case: StressCase) -> dict:
    cfg = deepcopy(base_cfg)
    field = cfg["demand"].get("workload_field")
    if field is None or "total_workload_bits_per_slot" not in field:
        raise ValueError("beta stress sweep requires continuous workload")
    field["total_workload_bits_per_slot"] *= case.workload_scale
    cfg["env"]["uav"]["cpu_frequency_hz"] *= case.uav_cpu_scale
    cfg["env"]["hap"]["cpu_frequency_hz"] *= case.hap_cpu_scale
    cfg["env"]["uav"]["queue_max_bits"] *= case.uav_queue_scale
    cfg["env"]["hap"]["queue_max_bits"] *= case.hap_queue_scale

    mmwave = cfg["communication"]["backhaul"]["mmwave"]
    mmwave["link_margin_db"] = (
        float(mmwave.get("link_margin_db", 0.0))
        + case.backhaul_margin_delta_db
    )
    mmwave["pathloss_exponent"] = (
        float(mmwave.get("pathloss_exponent", 2.0))
        + case.backhaul_pathloss_delta
    )

    cfg["normalization"]["uav_queue"]["divide_by_bits"] = float(
        cfg["env"]["uav"]["queue_max_bits"]
    )
    cfg["normalization"]["hap_queue"]["divide_by_bits"] = float(
        cfg["env"]["hap"]["queue_max_bits"]
    )
    cfg["cost"]["weights"] = derive_cost_weights(cfg)
    cfg["derived"] = derive_constants(cfg)
    validate_scenario(cfg)
    return cfg


def build_env_from_config(template_env, cfg):
    env = copy(template_env)
    env.cfg = cfg
    env.env = FiniteKHAPUAVMECEnv(cfg)
    env.k = int(cfg["env"]["fleet_size_k"])
    env.num_agents = env.k + 1
    env.v_h_max = float(cfg["env"]["hap"]["velocity_max_mps"])
    env.v_u_max = float(cfg["env"]["uav"]["velocity_max_mps"])
    return env


def scan_case(
    env,
    policy,
    args,
    seeds,
    motion_by_seed,
    constant_grid,
    oracle_iterations,
    oracle_lr,
    deployment_by_seed=None,
):
    constant_scan = {}
    for beta in constant_grid:
        constant_scan[f"{beta:.6g}"] = evaluate_controller(
            env,
            policy,
            args,
            seeds,
            motion_by_seed,
            "constant",
            deployment_by_seed=deployment_by_seed,
            constant_beta=float(beta),
        )
    best_key = min(
        constant_scan, key=lambda key: constant_scan[key]["cost"]
    )
    best_beta = float(best_key)
    best_constant = constant_scan[best_key]

    schedules = {}
    surrogate_costs = {}
    for seed in seeds:
        initial_uav, initial_hap, contexts = precompute_beta_exogenous(
            env,
            seed,
            motion_by_seed[seed],
            best_beta,
            initial_deployment=(
                None
                if deployment_by_seed is None
                else deployment_by_seed[seed]
            ),
        )
        schedule, surrogate_cost = optimize_clairvoyant_beta_schedule(
            env.env,
            initial_uav,
            initial_hap,
            contexts,
            best_beta,
            oracle_iterations,
            oracle_lr,
        )
        schedules[seed] = schedule
        surrogate_costs[str(seed)] = surrogate_cost

    oracle = evaluate_clairvoyant_schedules(
        env,
        policy,
        args,
        seeds,
        motion_by_seed,
        schedules,
        best_beta,
        deployment_by_seed=deployment_by_seed,
    )
    oracle["gain_vs_best_constant_percent"] = relative_gain(
        best_constant["cost"], oracle["cost"]
    )
    oracle["gain_vs_best_constant_beta_sensitive_percent"] = relative_gain(
        best_constant["beta_sensitive_cost"],
        oracle["beta_sensitive_cost"],
    )
    near_optimal_betas = [
        float(key)
        for key, value in constant_scan.items()
        if value["cost"] <= 1.01 * best_constant["cost"]
    ]
    return {
        "best_constant_beta": best_beta,
        "near_optimal_constant_beta_min": min(near_optimal_betas),
        "near_optimal_constant_beta_max": max(near_optimal_betas),
        "best_constant": best_constant,
        "clairvoyant_oracle": oracle,
        "oracle_surrogate_cost_by_seed": surrogate_costs,
        "constant_scan": constant_scan,
    }


def main():
    cli = parse_args()
    base_env, policy, args = load_policy(cli.model_dir)
    seeds = [
        cli.eval_seed + cli.seed_stride * index
        for index in range(cli.episodes)
    ]
    constant_grid = parse_beta_grid(cli.constant_betas)
    profiles = parse_motion_profiles(cli.motion_profiles)
    matched_hot_counts = parse_hot_counts(
        cli.matched_hot_counts, base_env.k
    )
    motion_by_profile = {}
    center_by_seed = {}
    for profile in profiles:
        if profile == "matched_hold":
            continue
        motion_by_seed = {}
        for seed in seeds:
            if profile == "learned":
                motion, _ = record_learned_motion(
                    base_env,
                    policy,
                    args,
                    seed,
                    counterfactual_stride=0,
                )
            else:
                motion = record_reference_motion(
                    base_env, seed, profile
                )
            motion_by_seed[seed] = motion
        motion_by_profile[profile] = motion_by_seed
    for seed in seeds:
        base_env.env.reset(seed=seed)
        center_by_seed[seed] = (
            base_env.env.state.demand_center_m.copy()
        )

    results = {}
    selected_cases = select_stress_cases(
        stress_cases(cli.preset), cli.cases
    )
    for case in selected_cases:
        cfg = apply_stress_case(base_env.cfg, case)
        by_motion = {}
        matched_count_by_seed = {}
        for profile in profiles:
            stressed_env = build_env_from_config(base_env, cfg)
            deployment_by_seed = None
            if profile == "matched_hold":
                deployment_by_seed = {}
                motion_by_seed = {}
                horizon = int(cfg["base"]["episode_horizon_slots"])
                for seed in seeds:
                    deployment, n_hot = select_matched_deployment(
                        cfg, center_by_seed[seed], matched_hot_counts
                    )
                    deployment_by_seed[seed] = deployment
                    matched_count_by_seed[str(seed)] = n_hot
                    motion_by_seed[seed] = [
                        MotionStep(
                            hap_velocity_mps=np.zeros(2),
                            uav_velocity_mps=np.zeros(
                                (stressed_env.k, 2)
                            ),
                        )
                        for _ in range(horizon)
                    ]
            else:
                motion_by_seed = motion_by_profile[profile]
            result = scan_case(
                stressed_env,
                policy,
                args,
                seeds,
                motion_by_seed,
                constant_grid,
                cli.oracle_iterations,
                cli.oracle_lr,
                deployment_by_seed=deployment_by_seed,
            )
            if profile == "matched_hold":
                result["selected_hot_count_by_seed"] = (
                    matched_count_by_seed.copy()
                )
            by_motion[profile] = result
            oracle = result["clairvoyant_oracle"]
            print(
                f"{case.name:38s} {profile:9s} "
                f"beta={result['best_constant_beta']:.2f} "
                f"total_gain="
                f"{oracle['gain_vs_best_constant_percent']:6.2f}% "
                "beta_sensitive_gain="
                f"{oracle['gain_vs_best_constant_beta_sensitive_percent']:6.2f}%"
            )
        robust_profiles = [
            profile
            for profile in ("matched_hold", "heuristic", "learned")
            if profile in by_motion
        ]
        if not robust_profiles:
            robust_profiles = list(profiles)
        robust_total_gain = min(
            by_motion[profile]["clairvoyant_oracle"][
                "gain_vs_best_constant_percent"
            ]
            for profile in robust_profiles
        )
        robust_beta_sensitive_gain = min(
            by_motion[profile]["clairvoyant_oracle"][
                "gain_vs_best_constant_beta_sensitive_percent"
            ]
            for profile in robust_profiles
        )
        results[case.name] = {
            "case": asdict(case),
            "by_motion": by_motion,
            "robust_profiles": robust_profiles,
            "robust_total_gain_percent": robust_total_gain,
            "robust_beta_sensitive_gain_percent": (
                robust_beta_sensitive_gain
            ),
        }

    ranking = sorted(
        (
            {
                "case": name,
                "robust_total_gain_percent": result[
                    "robust_total_gain_percent"
                ],
                "robust_beta_sensitive_gain_percent": result[
                    "robust_beta_sensitive_gain_percent"
                ],
                "motion_total_gain_percent": {
                    profile: motion_result["clairvoyant_oracle"][
                        "gain_vs_best_constant_percent"
                    ]
                    for profile, motion_result in result[
                        "by_motion"
                    ].items()
                },
            }
            for name, result in results.items()
        ),
        key=lambda row: row["robust_total_gain_percent"],
        reverse=True,
    )
    output = {
        "model_dir": cli.model_dir,
        "scenario": args.mec_scenario,
        "preset": cli.preset,
        "episodes": cli.episodes,
        "eval_seed": cli.eval_seed,
        "motion_profiles": profiles,
        "protocol": (
            "same baseline motion within each learned/heuristic/hover profile "
            "for every in-memory physical stress case"
        ),
        "selection_thresholds": {
            "total_gain_candidate_percent": 2.0,
            "total_gain_strong_percent": 5.0,
        },
        "ranking": ranking,
        "cases": results,
    }
    path = Path(cli.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
