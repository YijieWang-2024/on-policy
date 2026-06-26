"""Integration checks for MEC local and centralized observation contracts."""

from types import SimpleNamespace

import numpy as np

from onpolicy.envs.mec.MEC_env import MECEnv
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    RESOURCE_CONTEXT_SLICE,
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
    np.testing.assert_allclose(
        obs[:, RESOURCE_CONTEXT_SLICE],
        np.repeat(
            env.resource_context[None],
            env.num_agents,
            axis=0,
        ),
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


def test_resource_context_exposes_cardinality_and_k_dependent_shares():
    env8 = MECEnv(
        SimpleNamespace(
            mec_scenario="v6_hap_loadbearing",
            mec_fleet_size=8,
            mec_episode_horizon=350,
        )
    )
    env32 = MECEnv(
        SimpleNamespace(
            mec_scenario="v6_hap_loadbearing",
            mec_fleet_size=32,
            mec_episode_horizon=350,
        )
    )

    assert env8.env.horizon == 350
    assert env32.env.horizon == 350
    assert env8.resource_context[0] == 0.5
    assert env32.resource_context[0] == 2.0
    np.testing.assert_allclose(env8.resource_context[1:3], 1.0 / 8.0)
    np.testing.assert_allclose(env32.resource_context[1:3], 1.0 / 32.0)
