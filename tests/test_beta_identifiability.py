from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.envs.mec.MEC_env import MECEnv
from onpolicy.scripts.analysis.diagnose_beta_identifiability import (
    InitialDeployment,
    MotionStep,
    beta_schedule_objective,
    optimize_clairvoyant_beta_schedule,
    parse_beta_grid,
    precompute_beta_exogenous,
)


def _env():
    return MECEnv(
        SimpleNamespace(
            mec_scenario="v6_hap_loadbearing",
            mec_fleet_size=None,
        )
    )


def test_beta_grids_validate_and_include_boundaries():
    np.testing.assert_allclose(
        parse_beta_grid("0.5,0,1,0.5"), np.array([0.0, 0.5, 1.0])
    )


def test_clairvoyant_schedule_never_loses_to_its_constant_start():
    env = _env()
    raw = env.env
    motion = [
        MotionStep(
            hap_velocity_mps=np.zeros(2),
            uav_velocity_mps=np.zeros((raw.k, 2)),
        )
        for _ in range(4)
    ]
    initial_uav, initial_hap, contexts = precompute_beta_exogenous(
        env, 11, motion, 0.5
    )
    schedule, optimized = optimize_clairvoyant_beta_schedule(
        raw,
        initial_uav,
        initial_hap,
        contexts,
        initial_beta=0.5,
        iterations=3,
        learning_rate=0.05,
    )
    constant = np.full_like(schedule, 0.5)
    constant_value = float(
        beta_schedule_objective(
            raw,
            initial_uav,
            initial_hap,
            contexts,
            torch.as_tensor(constant, dtype=torch.float64),
        ).item()
    )
    assert schedule.shape == (4, raw.k)
    assert np.all((0.0 <= schedule) & (schedule <= 1.0))
    assert optimized <= constant_value + 1e-12


def test_initial_deployment_is_applied_before_exogenous_probe():
    env = _env()
    raw = env.env
    target = InitialDeployment(
        hap_xy_m=np.array([3000.0, 3000.0]),
        uav_xy_m=np.full((raw.k, 2), 2500.0),
    )
    motion = [
        MotionStep(
            hap_velocity_mps=np.zeros(2),
            uav_velocity_mps=np.zeros((raw.k, 2)),
        )
    ]
    precompute_beta_exogenous(
        env, 3, motion, 0.5, initial_deployment=target
    )
    np.testing.assert_allclose(raw.state.hap_xy_m, target.hap_xy_m)
    np.testing.assert_allclose(raw.state.uav_xy_m, target.uav_xy_m)
