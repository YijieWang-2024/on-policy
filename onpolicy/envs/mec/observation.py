"""Canonical observation layouts for the HAP/UAV MEC task."""

from __future__ import annotations

import numpy as np

ROLE_INDEX = 0
OWN_SLICE = slice(1, 4)
PUBLIC_SLICE = slice(4, 11)

UAV_STATE_DIM = 3
PUBLIC_STATE_DIM = 7
AGENT_OBS_DIM = 11
LEGACY_AGENT_OBS_DIM = 14


def team_state_dim(num_agents: int) -> int:
    """Return the dimension of ``[public, s_1, ..., s_K]``."""
    if num_agents < 2:
        raise ValueError("MEC requires one HAP and at least one UAV")
    return PUBLIC_STATE_DIM + UAV_STATE_DIM * (num_agents - 1)


def build_team_state(agent_obs: np.ndarray) -> np.ndarray:
    """Build one canonical centralized state from grouped local rows.

    ``agent_obs`` may be shaped ``[N, D]`` or ``[..., N, D]``. Agent zero is
    the HAP and agents one through K are UAVs.
    """
    obs = np.asarray(agent_obs)
    if obs.shape[-1] != AGENT_OBS_DIM:
        raise ValueError(
            f"expected MEC local observation dim {AGENT_OBS_DIM}, "
            f"got {obs.shape[-1]}"
        )
    if obs.shape[-2] < 2:
        raise ValueError("MEC grouped observations require K+1 agent rows")

    public = obs[..., 0, PUBLIC_SLICE]
    uavs = obs[..., 1:, OWN_SLICE].reshape(*obs.shape[:-2], -1)
    return np.concatenate([public, uavs], axis=-1)


def repeat_team_state(agent_obs: np.ndarray) -> np.ndarray:
    """Build and repeat the canonical state for every agent buffer row."""
    obs = np.asarray(agent_obs)
    state = build_team_state(obs)
    return np.repeat(state[..., None, :], obs.shape[-2], axis=-2)
