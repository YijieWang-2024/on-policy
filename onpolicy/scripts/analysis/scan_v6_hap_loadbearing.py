#!/usr/bin/env python
"""Scan the v6 operating region where the mobile hub trajectory is load-bearing.

The scan never edits the scenario YAML. Each setting is evaluated with paired
episode seeds under the same complete controller:

* UAVs move to a two-layer demand-matching layout.
* Each UAV uses queue-aware offloading beta.
* The hub follows the queue-weighted UAV centroid.

The main comparison freezes only the hub trajectory. Optional validation also
sets beta=0 or freezes all UAV trajectories. These are evaluation ablations,
not changes to the optimization variables used for training.

Example:

    python -m onpolicy.scripts.analysis.scan_v6_hap_loadbearing \
      --bandwidth-mhz 40,60,80,120,160,240,400 \
      --gain-db 15 --tx-power-dbm 20,23,26 \
      --link-margin-db 0,2,4,6 --episodes 3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from onpolicy.envs.mec.config_loader import derive_constants, load_scenario
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv


DEFAULT_SEED = 20260624


def _float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _sunflower(center: np.ndarray, n: int, radius_m: float) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    radii = radius_m * np.sqrt((np.arange(n) + 0.5) / n)
    angles = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(n)
    return center + np.stack(
        [radii * np.cos(angles), radii * np.sin(angles)], axis=1
    )


def _background_points(
    center: np.ndarray, n: int, sigma_m: float, lx: float, ly: float
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    candidates: list[np.ndarray] = []
    for radius in (2.2 * sigma_m, 3.0 * sigma_m, 3.8 * sigma_m, 4.6 * sigma_m):
        for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            point = center + radius * np.array([math.cos(angle), math.sin(angle)])
            if 0.0 <= point[0] <= lx and 0.0 <= point[1] <= ly:
                candidates.append(point)
    for x in np.linspace(400.0, lx - 400.0, 8):
        for y in np.linspace(400.0, ly - 400.0, 8):
            candidates.append(np.array([x, y], dtype=float))

    points = np.asarray(candidates, dtype=float)
    points = points[np.linalg.norm(points - center, axis=1) > 1.8 * sigma_m]
    chosen = [points[int(np.argmax(np.linalg.norm(points - center, axis=1)))]]
    while len(chosen) < n:
        nearest = np.min(
            [np.linalg.norm(points - point, axis=1) for point in chosen], axis=0
        )
        chosen.append(points[int(np.argmax(nearest))])
    return np.asarray(chosen[:n], dtype=float)


def _assigned_targets(env: FiniteKHAPUAVMECEnv, n_hot: int) -> np.ndarray:
    state = env.state
    assert state is not None
    cfg = env.cfg
    center = state.demand_center_m
    sigma = float(cfg["demand"]["workload_field"]["hotspot_sigma_m"])
    targets = np.vstack(
        [
            _sunflower(center, n_hot, 0.9 * sigma),
            _background_points(
                center, env.k - n_hot, sigma, float(env.lx), float(env.ly)
            ),
        ]
    )
    targets = np.clip(targets, [0.0, 0.0], [env.lx, env.ly])
    costs = np.linalg.norm(
        state.uav_xy_m[:, None, :] - targets[None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(costs)
    assigned = np.empty_like(targets)
    assigned[rows] = targets[cols]
    return assigned


def _velocity_toward(
    current: np.ndarray, target: np.ndarray, velocity_max: float
) -> np.ndarray:
    delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    if delta.ndim == 1:
        norm = float(np.linalg.norm(delta))
        return delta if norm <= velocity_max else delta * velocity_max / norm
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, velocity_max / np.maximum(norm, 1e-12))
    return delta * scale


def _action(
    env: FiniteKHAPUAVMECEnv,
    targets: np.ndarray,
    mode: str,
) -> dict[str, np.ndarray]:
    state = env.state
    assert state is not None
    cfg = env.cfg
    qmax = float(cfg["env"]["uav"]["queue_max_bits"])
    v_u = float(cfg["env"]["uav"]["velocity_max_mps"])
    v_h = float(cfg["env"]["hap"]["velocity_max_mps"])

    uav_velocity = _velocity_toward(state.uav_xy_m, targets, v_u)
    if mode == "uav_hover":
        uav_velocity[:] = 0.0

    # A positive floor keeps the hub centered on the fleet before queues build.
    weights = 0.05 + state.uav_queue_bits / max(qmax, 1e-12)
    hub_target = np.average(state.uav_xy_m, axis=0, weights=weights)
    hub_velocity = _velocity_toward(state.hap_xy_m, hub_target, v_h)
    if mode == "hub_frozen":
        hub_velocity[:] = 0.0

    beta = np.clip(state.uav_queue_bits / (0.2 * qmax), 0.0, 0.95)
    if mode == "beta_zero":
        beta[:] = 0.0

    return {
        "hap_velocity_mps": hub_velocity,
        "uav_velocity_mps": uav_velocity,
        "beta": beta,
    }


def _configured(
    base: dict,
    bandwidth_mhz: float,
    gain_db: float,
    tx_power_dbm: float,
    noise_figure_db: float,
    pathloss_exponent: float,
    link_margin_db: float,
    hap_cpu_ghz: float,
    workload_mbit: float,
) -> dict:
    cfg = deepcopy(base)
    k = int(cfg["env"]["fleet_size_k"])
    mmwave = cfg["communication"]["backhaul"]["mmwave"]
    mmwave["total_bandwidth_hz"] = bandwidth_mhz * 1e6
    mmwave["beam_bandwidth_hz"] = bandwidth_mhz * 1e6 / k
    mmwave["antenna_gain_total_db"] = gain_db
    mmwave["tx_power_dbm"] = tx_power_dbm
    mmwave["noise_figure_db"] = noise_figure_db
    mmwave["pathloss_exponent"] = pathloss_exponent
    mmwave["link_margin_db"] = link_margin_db
    cfg["env"]["hap"]["cpu_frequency_hz"] = hap_cpu_ghz * 1e9
    cfg["demand"]["workload_field"]["total_workload_bits_per_slot"] = (
        workload_mbit * 1e6
    )
    cfg["derived"] = derive_constants(cfg)
    return cfg


def _episode(
    cfg: dict,
    seed: int,
    mode: str,
    horizon: int,
    n_hot: int,
) -> dict[str, float]:
    env = FiniteKHAPUAVMECEnv(cfg, enforce_horizon=False)
    env.reset(seed=seed)
    targets = _assigned_targets(env, n_hot)
    c_h = float(cfg["derived"]["hap_compute_capacity_bits"])
    metrics = {
        key: 0.0
        for key in (
            "cost",
            "src",
            "queue",
            "overflow",
            "energy",
            "accepted_mbit",
            "offloaded_mbit",
            "uav_queue_mbit",
            "hub_queue_mbit",
            "backhaul_util",
            "hub_compute_util",
            "mean_bh_rate_mbps",
            "p10_bh_rate_mbps",
            "mean_uav_hub_m",
            "max_uav_hub_m",
            "hub_speed_mps",
            "uav_speed_mps",
        )
    }
    for _ in range(horizon):
        action = _action(env, targets, mode)
        _, _, _, _, info = env.step(action)
        rates = np.asarray(info["backhaul_rate_bps"], dtype=float)
        offloaded = np.asarray(info["B_i"], dtype=float)
        pre_queue = info["queue_bits_pre"]
        metrics["cost"] += float(info["training_cost"])
        metrics["src"] += float(info["src_cost_component"])
        metrics["queue"] += float(info["queue_cost_component"])
        metrics["overflow"] += float(info["ovf_cost_component"])
        metrics["energy"] += float(info["energy_cost_component"])
        metrics["accepted_mbit"] += float(np.sum(info["A_i"])) / 1e6
        metrics["offloaded_mbit"] += float(np.sum(offloaded)) / 1e6
        metrics["uav_queue_mbit"] += float(np.mean(pre_queue["uav"])) / 1e6
        metrics["hub_queue_mbit"] += float(pre_queue["hap"]) / 1e6
        metrics["backhaul_util"] += float(
            np.sum(offloaded) / max(np.sum(rates), 1e-12)
        )
        metrics["hub_compute_util"] += float(info["S_H"]) / max(c_h, 1e-12)
        metrics["mean_bh_rate_mbps"] += float(np.mean(rates)) / 1e6
        metrics["p10_bh_rate_mbps"] += float(np.quantile(rates, 0.1)) / 1e6
        distances = np.linalg.norm(
            np.asarray(info["service_uav_xy_m"])
            - np.asarray(info["service_hap_xy_m"])[None, :],
            axis=1,
        )
        metrics["mean_uav_hub_m"] += float(np.mean(distances))
        metrics["max_uav_hub_m"] += float(np.max(distances))
        metrics["hub_speed_mps"] += float(
            np.linalg.norm(action["hap_velocity_mps"])
        )
        metrics["uav_speed_mps"] += float(
            np.mean(np.linalg.norm(action["uav_velocity_mps"], axis=1))
        )
    return {key: value / horizon for key, value in metrics.items()}


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    out = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    for key in ("cost", "accepted_mbit", "offloaded_mbit", "backhaul_util"):
        values = [row[key] for row in rows]
        out[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def _paired_delta(
    baseline: list[dict[str, float]], ablated: list[dict[str, float]]
) -> float:
    return 100.0 * float(
        np.mean(
            [
                ablated_row["cost"] / max(base_row["cost"], 1e-12) - 1.0
                for base_row, ablated_row in zip(baseline, ablated)
            ]
        )
    )


def _link_rates(cfg: dict) -> dict[str, float]:
    env = FiniteKHAPUAVMECEnv(cfg)
    env.reset(seed=1)
    hub = np.array([0.0, 0.0])
    distances = (500.0, 1000.0, 2000.0, 3000.0)
    uavs = np.asarray([[distance, 0.0] for distance in distances])
    rates = env._backhaul_rate(hub, uavs) / 1e6
    return {
        f"rate_{int(distance)}m_mbps": float(rate)
        for distance, rate in zip(distances, rates)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="v6_continuous_workload")
    parser.add_argument("--bandwidth-mhz", default="50,100,200,400")
    parser.add_argument("--gain-db", default="15")
    parser.add_argument("--tx-power-dbm", default="27")
    parser.add_argument("--noise-figure-db", default="8")
    parser.add_argument("--pathloss-exponent", default="2.2")
    parser.add_argument("--link-margin-db", default="0")
    parser.add_argument("--hap-cpu-ghz", default="45")
    parser.add_argument("--workload-mbit", default="150")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n-hot", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validate-ablations",
        action="store_true",
        help="also evaluate beta=0 and UAV hover at every setting",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_scratch/analysis/v6_hap_loadbearing_scan.csv"),
    )
    args = parser.parse_args()

    base = load_scenario(args.scenario)
    settings = itertools.product(
        _float_grid(args.bandwidth_mhz),
        _float_grid(args.gain_db),
        _float_grid(args.tx_power_dbm),
        _float_grid(args.noise_figure_db),
        _float_grid(args.pathloss_exponent),
        _float_grid(args.link_margin_db),
        _float_grid(args.hap_cpu_ghz),
        _float_grid(args.workload_mbit),
    )
    modes = ["full", "hub_frozen"]
    if args.validate_ablations:
        modes.extend(["beta_zero", "uav_hover"])

    output_rows: list[dict[str, float | str]] = []
    for index, (
        bandwidth,
        gain,
        tx_power,
        noise_figure,
        pathloss_exponent,
        link_margin,
        hap_cpu,
        workload,
    ) in enumerate(settings, start=1):
        cfg = _configured(
            base,
            bandwidth,
            gain,
            tx_power,
            noise_figure,
            pathloss_exponent,
            link_margin,
            hap_cpu,
            workload,
        )
        seeds = [args.seed + 13 * episode for episode in range(args.episodes)]
        episodes = {
            mode: [
                _episode(cfg, seed, mode, args.horizon, args.n_hot)
                for seed in seeds
            ]
            for mode in modes
        }
        summary = {mode: _aggregate(rows) for mode, rows in episodes.items()}
        row: dict[str, float | str] = {
            "scenario": args.scenario,
            "bandwidth_mhz": bandwidth,
            "bandwidth_per_uav_mhz": bandwidth / int(cfg["env"]["fleet_size_k"]),
            "gain_db": gain,
            "tx_power_dbm": tx_power,
            "noise_figure_db": noise_figure,
            "pathloss_exponent": pathloss_exponent,
            "link_margin_db": link_margin,
            "hap_cpu_ghz": hap_cpu,
            "workload_mbit": workload,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "n_hot": args.n_hot,
            "hub_ablation_pct": _paired_delta(
                episodes["full"], episodes["hub_frozen"]
            ),
            **_link_rates(cfg),
        }
        for mode, values in summary.items():
            for key, value in values.items():
                row[f"{mode}_{key}"] = value
        if "beta_zero" in episodes:
            row["beta_zero_ablation_pct"] = _paired_delta(
                episodes["full"], episodes["beta_zero"]
            )
            row["uav_hover_ablation_pct"] = _paired_delta(
                episodes["full"], episodes["uav_hover"]
            )
        output_rows.append(row)
        print(
            f"[{index:03d}] W={bandwidth:6.1f}MHz gain={gain:4.1f}dB "
            f"P={tx_power:4.1f}dBm margin={link_margin:3.1f}dB "
            f"fH={hap_cpu:4.1f}GHz A={workload:5.1f}M "
            f"hub={row['hub_ablation_pct']:+6.2f}% "
            f"cost={summary['full']['cost']:.3f} "
            f"bh_util={summary['full']['backhaul_util']:.2f} "
            f"ovf={summary['full']['overflow']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "arguments": vars(args) | {"output": str(args.output)},
        "rows": len(output_rows),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {len(output_rows)} settings to {args.output}")


if __name__ == "__main__":
    main()
