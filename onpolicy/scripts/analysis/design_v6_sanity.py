"""Design sanity checks for the v6 continuous-workload MEC scenario.

This script is intentionally lightweight: it does not train a policy. It checks
the scale calibration that matters before training:

1. offered and accepted compute load ratios,
2. actual access spectral efficiency/capacity from channel integration,
3. continuous backhaul rates at representative hub-UAV distances,
4. deploy-and-hold probes over several hotspot centers.

Run from the repo root:

    python -m onpolicy.scripts.analysis.design_v6_sanity
"""

from __future__ import annotations

import math

import numpy as np

from onpolicy.envs.mec.config_loader import load_scenario
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv

def _hotspot_centers(region_m: float, n: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(20260623)
    fracs = rng.uniform(0.3, 0.7, size=(n, 2))
    anchors = np.array([
        [0.35, 0.35], [0.35, 0.65], [0.50, 0.50], [0.65, 0.35],
        [0.65, 0.65], [0.42, 0.58], [0.58, 0.42],
    ])
    fracs[: len(anchors)] = anchors
    return [np.array([fx * region_m, fy * region_m], dtype=float) for fx, fy in fracs]


def _sunflower(center: np.ndarray, n: int, radius_m: float) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    rr = radius_m * np.sqrt((np.arange(n) + 0.5) / n)
    ga = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(n)
    return center + np.stack([rr * np.cos(ga), rr * np.sin(ga)], axis=1)


def _background_points(center: np.ndarray, n: int, sigma_m: float, region_m: float) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 2), dtype=float)
    candidates: list[np.ndarray] = []
    for radius in (2.2 * sigma_m, 3.0 * sigma_m, 3.8 * sigma_m, 4.6 * sigma_m):
        for angle in np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False):
            point = center + np.array([radius * math.cos(angle), radius * math.sin(angle)])
            if 0.0 <= point[0] <= region_m and 0.0 <= point[1] <= region_m:
                candidates.append(point)
    for x in np.linspace(400.0, region_m - 400.0, 7):
        for y in np.linspace(400.0, region_m - 400.0, 7):
            candidates.append(np.array([x, y], dtype=float))

    pts = np.asarray(candidates, dtype=float)
    pts = pts[np.linalg.norm(pts - center, axis=1) > 1.8 * sigma_m]
    chosen = [pts[int(np.argmax(np.linalg.norm(pts - center, axis=1)))]]
    while len(chosen) < n:
        nearest = np.min([np.linalg.norm(pts - point, axis=1) for point in chosen], axis=0)
        chosen.append(pts[int(np.argmax(nearest))])
    return np.asarray(chosen[:n], dtype=float)


def _one_step_probe(cfg: dict, center: np.ndarray, n_hot: int) -> dict[str, float]:
    k = int(cfg["env"]["fleet_size_k"])
    region = float(cfg["env"]["region"]["lx_m"])
    sigma = float(cfg["demand"]["workload_field"]["hotspot_sigma_m"])
    env = FiniteKHAPUAVMECEnv(cfg)
    env.reset(seed=123)
    env.state.hap_xy_m = center.copy()
    env.state.demand_center_m = center.copy()
    env.state.demand_velocity_mps[:] = 0.0
    hot = _sunflower(center, n_hot, 0.9 * sigma)
    bg = _background_points(center, k - n_hot, sigma, region)
    env.state.uav_xy_m = np.vstack([hot, bg])
    action = {
        "hap_velocity_mps": np.zeros(2),
        "uav_velocity_mps": np.zeros((k, 2)),
        "beta": np.zeros(k),
    }
    _, _, _, _, info = env.step(action)
    diag = info["access_diagnostics"]
    regions = diag["regions"]
    hot = regions["hotspot"]
    bg = regions["background"]
    cap = np.asarray(info["access_rate_bps"], dtype=float)
    dem = np.asarray(info["A_dem_i"], dtype=float)
    eta = cap / float(cfg["communication"]["access"]["bandwidth_per_uav_hz"])
    active = dem > 1e-9
    return {
        "n_hot": float(n_hot),
        "accepted_mbit": float(np.sum(info["A_i"]) / 1e6),
        "source_mbit": float(info["U_src"] / 1e6),
        "outside_mbit": float(info["source_loss_outside_bits"] / 1e6),
        "capacity_source_mbit": float(info["source_loss_capacity_bits"] / 1e6),
        "hotspot_accepted_mbit": float(hot["accepted_bits"] / 1e6),
        "hotspot_source_mbit": float(hot["source_bits"] / 1e6),
        "background_accepted_mbit": float(bg["accepted_bits"] / 1e6),
        "background_source_mbit": float(bg["source_bits"] / 1e6),
        "access_cap_mbit": float(np.sum(cap) / 1e6),
        "access_util": float(np.sum(info["A_i"]) / max(np.sum(cap), 1e-9)),
        "eta_mean": float(diag["eta_served_workload_weighted"]),
        "eta_all": float(diag["eta_all_workload_weighted"]),
        "eta_p05": float(diag["eta_p05"]),
        "eta_p50": float(diag["eta_p50"]),
        "eta_p95": float(diag["eta_p95"]),
        "eta_min": float(np.min(eta[active])),
        "eta_max": float(np.max(eta[active])),
        "cost": float(info["training_cost"]),
    }


def _probe_hotspot_count(cfg: dict) -> tuple[list[int], dict[int, list[dict[str, float]]]]:
    region = float(cfg["env"]["region"]["lx_m"])
    centers = _hotspot_centers(region)
    by_n = {n: [] for n in range(int(cfg["env"]["fleet_size_k"]) + 1)}
    best_counts = []
    for center in centers:
        rows = [_one_step_probe(cfg, center, n) for n in by_n]
        best_counts.append(int(min(rows, key=lambda row: row["source_mbit"])["n_hot"]))
        for row in rows:
            by_n[int(row["n_hot"])].append(row)
    return best_counts, by_n


def _rollout_probe(cfg: dict, n_hot: int, hub_offset_m: float = 0.0, horizon: int = 120) -> dict[str, float]:
    k = int(cfg["env"]["fleet_size_k"])
    region = float(cfg["env"]["region"]["lx_m"])
    center = np.array([0.5 * region, 0.5 * region], dtype=float)
    sigma = float(cfg["demand"]["workload_field"]["hotspot_sigma_m"])
    env = FiniteKHAPUAVMECEnv(cfg)
    env.reset(seed=1)
    env.state.hap_xy_m = np.clip(center + np.array([hub_offset_m, 0.0]), 0.0, region)
    env.state.demand_center_m = center.copy()
    env.state.demand_velocity_mps[:] = 0.0
    env.state.uav_xy_m = np.vstack([
        _sunflower(center, n_hot, 0.9 * sigma),
        _background_points(center, k - n_hot, sigma, region),
    ])
    q_max = float(cfg["env"]["uav"]["queue_max_bits"])
    c_u = float(cfg["derived"]["uav_compute_capacity_bits"])
    c_h = float(cfg["derived"]["hap_compute_capacity_bits"])
    totals = {key: 0.0 for key in (
        "accepted_mbit", "source_mbit", "source_outside_mbit",
        "source_capacity_mbit", "overflow_mbit", "access_util",
        "backhaul_util", "uav_compute_util", "hub_compute_util",
        "uav_queue_mbit", "hub_queue_mbit")}
    for _ in range(horizon):
        beta = np.clip(env.state.uav_queue_bits / (0.25 * q_max), 0.0, 0.9)
        action = {
            "hap_velocity_mps": np.zeros(2),
            "uav_velocity_mps": np.zeros((k, 2)),
            "beta": beta,
        }
        _, _, _, _, info = env.step(action)
        access_cap = float(np.sum(info["access_rate_bps"]))
        backhaul_cap = float(np.sum(info["backhaul_rate_bps"]))
        accepted = float(np.sum(info["A_i"]))
        offloaded = float(np.sum(info["B_i"]))
        local = float(np.sum(info["S_i_U"]))
        hub_compute = float(info["S_H"])
        totals["accepted_mbit"] += accepted / 1e6
        totals["source_mbit"] += float(info["U_src"]) / 1e6
        totals["source_outside_mbit"] += float(info["source_loss_outside_bits"]) / 1e6
        totals["source_capacity_mbit"] += float(info["source_loss_capacity_bits"]) / 1e6
        totals["overflow_mbit"] += (float(np.sum(info["D_i_U"])) + float(info["D_H"])) / 1e6
        totals["access_util"] += accepted / max(access_cap, 1e-9)
        totals["backhaul_util"] += offloaded / max(backhaul_cap, 1e-9)
        totals["uav_compute_util"] += local / max(k * c_u, 1e-9)
        totals["hub_compute_util"] += hub_compute / max(c_h, 1e-9)
        totals["uav_queue_mbit"] += float(np.sum(env.state.uav_queue_bits)) / 1e6
        totals["hub_queue_mbit"] += float(env.state.hap_queue_bits) / 1e6
    return {key: val / horizon for key, val in totals.items()}


def main() -> None:
    cfg = load_scenario("v6_continuous_workload")
    k = int(cfg["env"]["fleet_size_k"])
    field = cfg["demand"]["workload_field"]
    a_tot = float(field["total_workload_bits_per_slot"])
    f_total = k * float(cfg["env"]["uav"]["cpu_frequency_hz"]) + float(
        cfg["env"]["hap"]["cpu_frequency_hz"])
    offered_compute_ratio = float(cfg["compute"]["cycles_per_bit"]) * a_tot / f_total

    print("=== v6 continuous-workload sanity ===")
    print(f"K={k}  A_tot={a_tot / 1e6:.1f} Mbit/slot  zeta={field['hotspot_fraction']:.2f}")
    print(f"W_ac_i={cfg['communication']['access']['bandwidth_per_uav_hz'] / 1e6:.3f} MHz")
    print(f"W_bh_i={cfg['communication']['backhaul']['mmwave']['beam_bandwidth_hz'] / 1e6:.3f} MHz")
    print(f"offered_compute_ratio={offered_compute_ratio:.2f}")

    env = FiniteKHAPUAVMECEnv(cfg)
    env.reset(seed=1)
    hub = np.array([3000.0, 3000.0])
    horiz = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0])
    uav = np.stack([hub[0] + horiz, np.full_like(horiz, hub[1])], axis=1)
    rates = env._backhaul_rate(hub, uav)
    print("backhaul Mbps by horizontal distance:")
    for d_m, rate in zip(horiz, rates):
        print(f"  {d_m:4.0f} m: {rate / 1e6:6.1f}")

    best_counts, by_n = _probe_hotspot_count(cfg)
    print(f"best n_hot counts over {len(best_counts)} centers: "
          f"{ {n: best_counts.count(n) for n in sorted(set(best_counts))} }")
    print("n_hot  acc  src(out/cap)  hot_acc/src  bg_acc/src  cap  util  eta(p05/p50/p95)")
    mean_source = {n: np.mean([row["source_mbit"] for row in rows])
                   for n, rows in by_n.items()}
    best_n = min(mean_source, key=mean_source.get)
    for n_hot in list(range(3, 11)) + [16]:
        rows = by_n[n_hot]
        marker = "*" if n_hot == best_n else " "
        print(
            f"{marker}{n_hot:2d}    "
            f"{np.mean([r['accepted_mbit'] for r in rows]):6.1f}  "
            f"{np.mean([r['source_mbit'] for r in rows]):5.1f}"
            f"({np.mean([r['outside_mbit'] for r in rows]):4.1f}/"
            f"{np.mean([r['capacity_source_mbit'] for r in rows]):4.1f})  "
            f"{np.mean([r['hotspot_accepted_mbit'] for r in rows]):5.1f}/"
            f"{np.mean([r['hotspot_source_mbit'] for r in rows]):4.1f}  "
            f"{np.mean([r['background_accepted_mbit'] for r in rows]):5.1f}/"
            f"{np.mean([r['background_source_mbit'] for r in rows]):4.1f}  "
            f"{np.mean([r['access_cap_mbit'] for r in rows]):6.1f}  "
            f"{np.mean([r['access_util'] for r in rows]):4.2f}  "
            f"{np.mean([r['eta_p05'] for r in rows]):4.1f}/"
            f"{np.mean([r['eta_p50'] for r in rows]):4.1f}/"
            f"{np.mean([r['eta_p95'] for r in rows]):4.1f}"
        )

    print("rollout probes (static layout, q-aware beta):")
    for offset in (0.0, 3000.0):
        row = _rollout_probe(cfg, best_n, hub_offset_m=offset)
        print(
            f"  hub_offset={offset / 1000:.1f} km  "
            f"acc={row['accepted_mbit']:.1f} "
            f"src={row['source_mbit']:.1f}"
            f"(out={row['source_outside_mbit']:.1f}, cap={row['source_capacity_mbit']:.1f}) "
            f"ovf={row['overflow_mbit']:.1f}  "
            f"util(ac/bh/uav/hub)="
            f"{row['access_util']:.2f}/{row['backhaul_util']:.2f}/"
            f"{row['uav_compute_util']:.2f}/{row['hub_compute_util']:.2f}  "
            f"Q(uav/hub)={row['uav_queue_mbit']:.1f}/{row['hub_queue_mbit']:.1f} Mbit"
        )


if __name__ == "__main__":
    main()
