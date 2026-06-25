"""Integration checks for MEC local and centralized observation contracts."""

from types import SimpleNamespace

import numpy as np

from onpolicy.envs.mec.MEC_env import MECEnv
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    build_team_state,
    team_state_dim,
)
from onpolicy.runner.shared.mec_runner import MECRunner


def test_environment_and_runner_use_canonical_team_state():
    env = MECEnv(
        SimpleNamespace(
            mec_scenario="v2_iort_6km_mmwave",
            mec_fleet_size=4,
        )
    )
    obs, _ = env.reset(seed=3)
    assert obs.shape == (env.num_agents, AGENT_OBS_DIM)
    assert env.observation_space[0].shape == (AGENT_OBS_DIM,)
    assert env.share_observation_space[0].shape == (
        team_state_dim(env.num_agents),
    )

    runner = MECRunner.__new__(MECRunner)
    runner.use_centralized_V = True
    runner.num_agents = env.num_agents
    batched_obs = np.stack([obs, obs], axis=0)
    share_obs = runner._share_obs(batched_obs)
    assert share_obs.shape == (
        2,
        env.num_agents,
        team_state_dim(env.num_agents),
    )
    expected = build_team_state(batched_obs)
    np.testing.assert_allclose(
        share_obs,
        np.repeat(expected[:, None], env.num_agents, axis=1),
    )
