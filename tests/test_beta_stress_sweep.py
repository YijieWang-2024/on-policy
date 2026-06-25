from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onpolicy.envs.mec.MEC_env import MECEnv
from onpolicy.scripts.analysis.sweep_beta_identifiability import (
    StressCase,
    apply_stress_case,
    build_env_from_config,
    parse_motion_profiles,
    parse_hot_counts,
    record_reference_motion,
    select_matched_deployment,
    select_stress_cases,
    stress_cases,
)


def _env():
    return MECEnv(
        SimpleNamespace(
            mec_scenario="v6_hap_loadbearing",
            mec_fleet_size=None,
        )
    )


def test_apply_stress_case_rederives_physical_constants():
    env = _env()
    case = StressCase(
        "test",
        "interaction",
        workload_scale=1.25,
        uav_cpu_scale=0.5,
        hap_queue_scale=0.5,
        backhaul_margin_delta_db=3.0,
    )
    cfg = apply_stress_case(env.cfg, case)

    assert (
        cfg["demand"]["workload_field"]["total_workload_bits_per_slot"]
        == 1.25
        * env.cfg["demand"]["workload_field"][
            "total_workload_bits_per_slot"
        ]
    )
    assert cfg["derived"]["uav_compute_capacity_bits"] == (
        0.5 * env.cfg["derived"]["uav_compute_capacity_bits"]
    )
    assert cfg["env"]["hap"]["queue_max_bits"] == (
        0.5 * env.cfg["env"]["hap"]["queue_max_bits"]
    )
    assert cfg["derived"]["bh_link_margin_db"] == (
        env.cfg["derived"]["bh_link_margin_db"] + 3.0
    )
    assert cfg["cost"]["weights"]["omega_queue_per_bit"] != (
        env.cfg["cost"]["weights"]["omega_queue_per_bit"]
    )


def test_stress_cases_are_unique_and_buildable():
    env = _env()
    cases = stress_cases("coarse")
    assert len({case.name for case in cases}) == len(cases)
    stressed = build_env_from_config(
        env, apply_stress_case(env.cfg, cases[-1])
    )
    assert stressed.k == env.k
    assert stressed.env.cfg is stressed.cfg


def test_motion_profiles_validate_and_reference_motion_has_full_horizon():
    assert parse_motion_profiles("learned,heuristic,learned") == (
        "learned",
        "heuristic",
    )
    env = _env()
    hover = record_reference_motion(env, seed=5, profile="hover")
    heuristic = record_reference_motion(
        env, seed=5, profile="heuristic"
    )
    horizon = int(env.cfg["base"]["episode_horizon_slots"])
    assert len(hover) == horizon
    assert len(heuristic) == horizon
    assert all((step.uav_velocity_mps == 0.0).all() for step in hover)
    assert any(
        (step.uav_velocity_mps != 0.0).any() for step in heuristic
    )
    assert parse_motion_profiles("matched_hold") == ("matched_hold",)


def test_select_stress_cases_preserves_requested_order():
    cases = stress_cases("coarse")
    selected = select_stress_cases(
        cases, "workload_x1.5,baseline,workload_x1.5"
    )
    assert [case.name for case in selected] == [
        "workload_x1.5",
        "baseline",
    ]


def test_matched_deployment_selects_valid_hot_count_and_geometry():
    env = _env()
    center = np.array([3000.0, 3000.0])
    counts = parse_hot_counts("5,6,7", env.k)
    deployment, count = select_matched_deployment(
        env.cfg, center, counts
    )
    assert count in counts
    assert deployment.uav_xy_m.shape == (env.k, 2)
    np.testing.assert_allclose(
        deployment.hap_xy_m,
        np.mean(deployment.uav_xy_m, axis=0),
    )
