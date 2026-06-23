"""Tests for the v2 finite-K MEC env (spec section 6).

Two layers:
1. PARITY (safety rope): unchanged physics (access channel, service split,
   queues, DVFS, rotary/hub energy, safety, cost plumbing) must match the
   ORIGIN env step-by-step to 1e-8 when both are configured identically and
   the deliberately-changed pieces (backhaul model, src/ovf split, demand
   noise) are neutralized: port backhaul patched to the origin 2.4 GHz
   formula, omega_src == omega_ovf == origin omega_loss, demand noise = 0
   with matched start/velocity.
2. v2 DELTA unit tests: 60 GHz service circle, cost split, random-walk
   demand, permutation invariance, mid-move service positions, smoke.

Runnable standalone: ``python3 test_finite_k_env.py``.
"""

from __future__ import annotations

import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ONPOLICY_ROOT = Path(__file__).resolve().parents[4]
ORIGIN_SRC_ENV = "MFMEC_ORIGIN_SRC"
_origin_src = os.environ.get(ORIGIN_SRC_ENV)
ORIGIN_SRC = Path(_origin_src).expanduser() if _origin_src else None
sys.path.insert(0, str(ONPOLICY_ROOT))
if ORIGIN_SRC is not None and ORIGIN_SRC.exists():
    sys.path.insert(0, str(ORIGIN_SRC))

from onpolicy.envs.mec.config_loader import load_scenario  # noqa: E402
from onpolicy.envs.mec.finite_k_env import EnvState, FiniteKHAPUAVMECEnv  # noqa: E402

SEED = 123
K = 24


# --------------------------------------------------------------- helpers

def _port_env(noise_std: float | None = None, fleet_size_k: int | None = None,
              seed: int = SEED):
    cfg = load_scenario(fleet_size_k=fleet_size_k)
    if noise_std is not None:
        cfg["demand"]["process"]["noise_std_m"] = noise_std
    env = FiniteKHAPUAVMECEnv(cfg)
    obs, _ = env.reset(seed=seed)
    return env, cfg, obs


def _random_actions(rng: np.random.Generator, k: int):
    return {
        "hap_velocity_mps": rng.uniform(-30, 30, 2),
        "uav_velocity_mps": rng.uniform(-40, 40, (k, 2)),
        "beta": rng.uniform(0, 1, k),
    }


# ------------------------------------------------------------ parity test

def test_full_step_parity_with_origin():
    if ORIGIN_SRC is None or not ORIGIN_SRC.exists():
        pytest.skip(
            f"set {ORIGIN_SRC_ENV} to the origin Mean Field Mec/src path "
            "to run the parity test"
        )

    from mfmec.config import (apply_scenario_derivations, derive_constants,
                              load_scenario as load_origin)
    from mfmec.env.finite_k_env import FiniteKHAPUAVMECEnv as OriginEnv

    # ---- port env: neutralize v2-only deltas
    port_env, cfg_p, _ = _port_env(noise_std=0.0, seed=SEED)
    omega_common = 5.0e-8
    cfg_p["cost"]["weights"]["omega_src_per_bit"] = omega_common
    cfg_p["cost"]["weights"]["omega_ovf_per_bit"] = omega_common
    cfg_p["communication"]["backhaul"]["mmwave"]["tx_power_dbm"] = 40.0  # 10 W like origin
    start = port_env.state.demand_center_m.copy()
    vel = port_env.state.demand_velocity_mps.copy()

    # ---- origin env: mutate the paper scenario onto the v2 locked geometry
    cfg_o = load_origin("configs/scenario/paper_iort_10km_hap8km.yaml")
    e, p = cfg_o["env"], cfg_p["env"]
    e["fleet_size_k"] = K
    e["region"] = {"lx_m": 6000.0, "ly_m": 6000.0}
    e["slot_timing"] = {"service_position": "mid_move"}
    for grp in ("hap", "uav"):
        for key in ("altitude_m", "velocity_max_mps", "queue_max_bits",
                    "cpu_frequency_hz", "initial_xy_m", "initial_queue_bits"):
            e[grp][key] = deepcopy(p[grp][key])
    e["hap"]["static_power_w"] = p["hap"]["static_power_w"]
    e["hap"]["speed_smoothing_eps_mps"] = p["hap"]["speed_smoothing_eps_mps"]
    e["hap"]["wind_speed_mps"] = p["hap"]["wind_speed_mps"]
    e["hap"]["air_density_kg_per_m3"] = p["hap"]["air_density_kg_per_m3"]
    e["hap"]["drag_coefficient"] = p["hap"]["drag_coefficient"]
    cfg_o["demand"]["device_density"] = deepcopy(cfg_p["demand"]["device_density"])
    cfg_o["demand"]["packet_size_bits"] = cfg_p["demand"]["packet_size_bits"]
    ap_o, ap_p = cfg_o["demand"]["activity_probability"], cfg_p["demand"]["activity_probability"]
    for key in ("base_probability", "hotspot_peak_increment", "hotspot_sigma_m",
                "clip_probability"):
        ap_o[key] = deepcopy(ap_p[key])
    cfg_o["demand"]["process"] = {
        "model": "drifting_hotspot", "label": "drifting_hotspot",
        "initial_center_m": [float(start[0]), float(start[1])],
        "velocity_mps": [float(vel[0]), float(vel[1])],
    }
    cfg_o["compute"]["cycles_per_bit"] = 4000.0
    cfg_o["energy"]["uav_rotary_wing"] = deepcopy(cfg_p["energy"]["uav_rotary_wing"])
    w = cfg_p["cost"]["weights"]
    cfg_o["cost"]["weights"] = {
        "omega_queue_per_bit": w["omega_queue_per_bit"],
        "omega_loss_per_bit": omega_common,
        "omega_energy_per_j": w["omega_energy_per_j"],
    }
    cfg_o["cost"]["safety"] = deepcopy(cfg_p["cost"]["safety"])
    cfg_o["cost"]["safety"]["weight"] = cfg_p["cost"]["lambdas"]["safety"]
    apply_scenario_derivations(cfg_o)
    cfg_o["derived"] = derive_constants(cfg_o)

    origin_env = OriginEnv(cfg_o)
    origin_env.reset(seed=SEED)

    # ---- patch port backhaul to the origin 2.4 GHz formula (same numbers)
    bh = cfg_o["communication"]["backhaul"]
    dz = float(e["hap"]["altitude_m"]) - float(e["uav"]["altitude_m"])
    ref_gain = float(bh["reference_gain_linear_at_1m"])
    alpha = float(bh["pathloss_exponent"])
    p_tx = float(bh["uav_transmit_power_w"])
    w_bh = float(bh["bandwidth_per_uav_hz"])
    n_bh = float(cfg_o["derived"]["backhaul_noise_power_w"])

    def origin_backhaul(hap_xy, uav_xy):
        horiz = np.linalg.norm(uav_xy - hap_xy[None, :], axis=1)
        d = np.sqrt(horiz**2 + dz**2)
        snr = p_tx * ref_gain * d ** (-alpha) / n_bh
        return w_bh * np.log2(1.0 + snr)

    port_env._backhaul_rate = origin_backhaul

    rng = np.random.default_rng(7)
    for t in range(60):
        act = _random_actions(rng, K)
        _, r_p, _, _, i_p = port_env.step(deepcopy(act))
        _, r_o, _, _, i_o = origin_env.step(deepcopy(act))
        kw = dict(rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(r_p, r_o, err_msg=f"reward t={t}", **kw)
        np.testing.assert_allclose(i_p["A_i"], i_o["A_i"], err_msg=f"A_i t={t}", **kw)
        np.testing.assert_allclose(i_p["U_src"], i_o["U_src"], **kw)
        np.testing.assert_allclose(i_p["D_i_U"], i_o["D_i_U"], **kw)
        np.testing.assert_allclose(i_p["D_H"], i_o["D_H"], **kw)
        np.testing.assert_allclose(i_p["queue_bits_post"]["uav"], i_o["queue_bits_post"]["uav"], **kw)
        np.testing.assert_allclose(i_p["queue_bits_post"]["hap"], i_o["queue_bits_post"]["hap"], **kw)
        np.testing.assert_allclose(i_p["uav_energy_j"], i_o["uav_energy_j"], **kw)
        np.testing.assert_allclose(i_p["hap_energy_j"], i_o["hap_energy_j"], **kw)
        np.testing.assert_allclose(
            i_p["raw_src_bits"] + i_p["raw_ovf_bits"], i_o["raw_loss_bits"], **kw)
        np.testing.assert_allclose(i_p["next_demand_center_m"], i_o["next_demand_center_m"], **kw)


# --------------------------------------------------------- v2 delta tests

def test_backhaul_60ghz_service_circle():
    env, cfg, _ = _port_env()
    d = cfg["derived"]
    hub = np.array([3000.0, 3000.0])
    horiz = np.array([0.0, 1000.0, 2000.0, 2300.0, 3000.0])
    uav_xy = np.stack([hub[0] + horiz, np.full_like(horiz, hub[1])], axis=1)
    rates = env._backhaul_rate(hub, uav_xy)
    dz = 1200.0
    for idx, h in enumerate(horiz):
        dist = math.hypot(h, dz)
        snr_db = (d["bh_link_budget_const_db"] - 20 * math.log10(dist)
                  - d["bh_kappa_o2_db_per_km"] * dist / 1000.0)
        expected = d["bh_beam_bandwidth_hz"] * math.log2(1 + 10 ** (snr_db / 10))
        if snr_db >= d["bh_demod_snr_min_db"]:
            np.testing.assert_allclose(rates[idx], expected, rtol=1e-12)
        else:
            assert rates[idx] == 0.0, (h, rates[idx])
    assert rates[0] > rates[1] > rates[2] > 0.0
    assert rates[3] == 0.0 and rates[4] == 0.0  # outside ~2.2 km circle


def test_v6_continuous_workload_and_backhaul_smoke():
    cfg = load_scenario("v6_continuous_workload")
    env = FiniteKHAPUAVMECEnv(cfg)
    obs, _ = env.reset(seed=SEED)
    density, fresh = env._demand_density(obs["demand"]["hotspot_center_m"])
    total = float(cfg["demand"]["workload_field"]["total_workload_bits_per_slot"])
    np.testing.assert_allclose(np.sum(density) * env.cell_area, total, rtol=1e-12)
    np.testing.assert_allclose(density, fresh, rtol=0, atol=0)

    hub = np.array([3000.0, 3000.0])
    horiz = np.array([0.0, 1000.0, 3000.0, 6000.0])
    uav_xy = np.stack([hub[0] + horiz, np.full_like(horiz, hub[1])], axis=1)
    rates = env._backhaul_rate(hub, uav_xy)
    assert np.all(rates > 0.0)
    assert rates[0] > rates[1] > rates[2] > rates[3]

    k = cfg["env"]["fleet_size_k"]
    null = {"hap_velocity_mps": np.zeros(2), "uav_velocity_mps": np.zeros((k, 2)),
            "beta": np.zeros(k)}
    _, reward, terminated, truncated, info = env.step(null)
    assert terminated is False and truncated is False
    assert math.isfinite(reward)
    assert info["phi_sum_error"] < 1e-6
    diag = info["access_diagnostics"]
    hot = diag["regions"]["hotspot"]
    bg = diag["regions"]["background"]
    np.testing.assert_allclose(info["A_i"] + info["source_loss_uav_bits"],
                               info["A_dem_i"], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(
        info["source_loss_outside_bits"] + info["source_loss_capacity_bits"],
        info["U_src"], rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(
        hot["offered_bits"] + bg["offered_bits"],
        cfg["demand"]["workload_field"]["total_workload_bits_per_slot"],
        rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(
        hot["accepted_bits"] + bg["accepted_bits"], np.sum(info["A_i"]),
        rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(
        hot["source_bits"] + bg["source_bits"], info["U_src"],
        rtol=1e-12, atol=1e-6)
    assert 0.0 <= diag["eta_p05"] <= diag["eta_p50"] <= diag["eta_p95"]
    assert diag["n_hotspot_uav"] + diag["n_background_uav"] == k
    assert 0.0 <= float(np.sum(info["A_i"])) <= total
    assert 0.0 <= info["U_src"] <= total + 1e-6


def test_cost_components_and_split():
    env, cfg, _ = _port_env()
    w = cfg["cost"]["weights"]
    rng = np.random.default_rng(0)
    for _ in range(5):
        _, reward, _, _, info = env.step(_random_actions(rng, K))
        total = (info["queue_cost_component"] + info["src_cost_component"]
                 + info["ovf_cost_component"] + info["energy_cost_component"]
                 + info["safety_cost_component"])
        np.testing.assert_allclose(info["training_cost"], total, rtol=1e-12)
        np.testing.assert_allclose(reward, -info["training_cost"], rtol=1e-12)
        np.testing.assert_allclose(
            info["src_cost_component"], w["omega_src_per_bit"] * info["raw_src_bits"], rtol=1e-12)
        np.testing.assert_allclose(
            info["ovf_cost_component"], w["omega_ovf_per_bit"] * info["raw_ovf_bits"], rtol=1e-12)
    # equal loss weights (a lost bit is a lost bit); ovf must never exceed src (anti-laziness)
    np.testing.assert_allclose(
        w["omega_ovf_per_bit"] / w["omega_src_per_bit"], 1.0, rtol=1e-12)
    assert w["omega_ovf_per_bit"] <= w["omega_src_per_bit"] + 1e-18


def test_demand_random_walk():
    env1, _, _ = _port_env(noise_std=0.0, seed=1)
    start = env1.state.demand_center_m.copy()
    np.testing.assert_allclose(start, [1200.0, 1200.0])
    v1 = env1.state.demand_velocity_mps.copy()
    assert abs(np.linalg.norm(v1) - 25.0) < 1e-9
    env2, _, _ = _port_env(noise_std=0.0, seed=2)
    assert not np.allclose(env2.state.demand_velocity_mps, v1)  # heading differs by seed

    null = {"hap_velocity_mps": np.zeros(2), "uav_velocity_mps": np.zeros((K, 2)),
            "beta": np.zeros(K)}
    for t in range(1, 11):  # straight line while away from boundaries
        env1.step(null)
        np.testing.assert_allclose(
            env1.state.demand_center_m, start + t * 1.0 * v1, rtol=0, atol=1e-9)

    # reflection: craft a state pushing out of the east edge
    env1.state = EnvState(env1.state.hap_xy_m, env1.state.hap_queue_bits,
                          env1.state.uav_xy_m, env1.state.uav_queue_bits,
                          np.array([5990.0, 3000.0]), np.array([25.0, 0.0]), 0)
    center, vel = env1._advance_demand(env1.state)
    np.testing.assert_allclose(center, [2 * 6000.0 - 6015.0, 3000.0], atol=1e-9)
    np.testing.assert_allclose(vel, [-25.0, 0.0], atol=1e-12)


def test_permutation_invariance():
    k = 8
    perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    env_a, cfg_a, _ = _port_env(fleet_size_k=k, seed=SEED)
    cfg_b = load_scenario(fleet_size_k=k)
    cfg_b["env"]["uav"]["initial_xy_m"] = [
        cfg_a["env"]["uav"]["initial_xy_m"][j] for j in perm]
    env_b = FiniteKHAPUAVMECEnv(cfg_b)
    env_b.reset(seed=SEED)  # same seed -> same demand heading
    rng = np.random.default_rng(11)
    for _ in range(20):
        act = _random_actions(rng, k)
        act_perm = {"hap_velocity_mps": act["hap_velocity_mps"],
                    "uav_velocity_mps": act["uav_velocity_mps"][perm],
                    "beta": act["beta"][perm]}
        _, r_a, _, _, i_a = env_a.step(act)
        _, r_b, _, _, i_b = env_b.step(act_perm)
        np.testing.assert_allclose(r_b, r_a, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(i_b["A_i"], i_a["A_i"][perm], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(i_b["D_i_U"], i_a["D_i_U"][perm], rtol=1e-9, atol=1e-9)


def test_mid_move_service_positions_and_smoke():
    env, _, _ = _port_env(seed=SEED)
    rng = np.random.default_rng(3)
    truncated = False
    for t in range(1, 201):
        act = _random_actions(rng, K)
        pre_uav = env.state.uav_xy_m.copy()
        obs, reward, terminated, truncated, info = env.step(act)
        assert terminated is False
        assert math.isfinite(reward)
        assert info["phi_sum_error"] < 1e-6
        np.testing.assert_allclose(
            info["service_uav_xy_m"], 0.5 * (pre_uav + info["next_uav_xy_m"]), atol=1e-9)
        assert truncated == (t == 200)
    assert truncated
    assert obs["normalized"]["uavs"].shape == (K, 3)
    assert obs["normalized"]["hap"].shape == (2 + 1 + 4,)  # pos2 + queue1 + demand4


if __name__ == "__main__":
    mod = sys.modules["__main__"]
    fns = [getattr(mod, n) for n in sorted(dir(mod)) if n.startswith("test_")]
    passed = 0
    skipped = 0
    for fn in fns:
        try:
            fn()
        except pytest.skip.Exception as exc:
            skipped += 1
            print(f"SKIP {fn.__name__}: {exc}")
        else:
            passed += 1
            print(f"PASS {fn.__name__}")
    print(f"\n{passed} tests passed, {skipped} skipped.")
