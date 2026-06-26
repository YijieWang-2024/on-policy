"""Structural tests for aligned Mean/Flat/Set MEC policies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces

from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    PUBLIC_STATE_DIM,
    UAV_STATE_DIM,
    build_team_state,
    repeat_team_state,
    team_state_dim,
)
from onpolicy.utils.shared_buffer import SharedReplayBuffer

OBS_DIM = AGENT_OBS_DIM
ACT_DIM = 3
K = 5
N = K + 1
N_ENV = 3
CENT_DIM = team_state_dim(N)


def _args(architecture="set"):
    return SimpleNamespace(
        hidden_size=64,
        layer_N=1,
        gain=0.01,
        use_orthogonal=True,
        use_ReLU=True,
        use_feature_normalization=False,
        use_policy_active_masks=True,
        use_recurrent_policy=False,
        use_naive_recurrent_policy=False,
        recurrent_N=1,
        lr=5e-4,
        critic_lr=5e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        use_popart=False,
        use_valuenorm=True,
        algorithm_name="mappo",
        env_name="MEC",
        stacked_frames=1,
        mec_logstd_init=-1.9,
        mec_rolewise_loss=True,
        mec_policy_arch=architecture,
        mec_set_dim=32,
        mec_set_heads=4,
        mec_set_num_seeds=3,
        mec_set_element_blocks=2,
        mec_set_latent_blocks=1,
    )


def _spaces():
    obs = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    cent = spaces.Box(-np.inf, np.inf, (CENT_DIM,), np.float32)
    act = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    return obs, cent, act


def _team_obs(seed=0):
    rng = np.random.default_rng(seed)
    teams = np.zeros((N_ENV, N, OBS_DIM), dtype=np.float32)
    for env_id in range(N_ENV):
        public = rng.uniform(-0.5, 0.5, size=PUBLIC_STATE_DIM)
        uavs = rng.uniform(0.0, 1.0, size=(K, UAV_STATE_DIM))
        teams[env_id, 0, 0] = 1.0
        teams[env_id, 0, 1:4] = public[:3]
        teams[env_id, :, 4:4 + PUBLIC_STATE_DIM] = public
        teams[env_id, 1:, 1:4] = uavs
    return teams


def _central_obs(teams):
    return repeat_team_state(teams)


def _policy(architecture="set"):
    obs, cent, act = _spaces()
    return MECPolicy(
        _args(architecture),
        obs,
        cent,
        act,
        torch.device("cpu"),
        num_agents=N,
    )


def _rnn_masks():
    rnn = np.zeros((N_ENV * N, 1, 64), np.float32)
    masks = np.ones((N_ENV * N, 1), np.float32)
    return rnn, masks


def _permuted(teams):
    permutation = np.array([3, 0, 4, 1, 2])
    permuted = teams.copy()
    permuted[:, 1:] = teams[:, 1:][:, permutation]
    return permutation, permuted


def test_canonical_team_state_layout():
    teams = _team_obs(seed=2)
    state = build_team_state(teams)
    assert state.shape == (N_ENV, CENT_DIM)
    np.testing.assert_allclose(
        state[:, :PUBLIC_STATE_DIM],
        teams[:, 0, 4:4 + PUBLIC_STATE_DIM],
    )
    np.testing.assert_allclose(
        state[:, PUBLIC_STATE_DIM:].reshape(N_ENV, K, 3),
        teams[:, 1:, 1:4],
    )
    repeated = repeat_team_state(teams)
    assert repeated.shape == (N_ENV, N, CENT_DIM)
    np.testing.assert_allclose(
        repeated, np.repeat(repeated[:, :1], N, axis=1)
    )


def test_mean_and_set_are_invariant_and_equivariant():
    teams = _team_obs(seed=8)
    permutation, permuted = _permuted(teams)
    rnn, masks = _rnn_masks()

    for architecture in ("mean", "set"):
        torch.manual_seed(4)
        policy = _policy(architecture)
        with torch.no_grad():
            representation = policy.actor.population_representation(
                teams.reshape(-1, OBS_DIM)
            )
            representation_perm = policy.actor.population_representation(
                permuted.reshape(-1, OBS_DIM)
            )
            actions, _ = policy.act(
                teams.reshape(-1, OBS_DIM),
                rnn,
                masks,
                deterministic=True,
            )
            actions_perm, _ = policy.act(
                permuted.reshape(-1, OBS_DIM),
                rnn,
                masks,
                deterministic=True,
            )
            values = policy.get_values(
                _central_obs(teams).reshape(N_ENV * N, -1),
                rnn,
                masks,
            )
            values_perm = policy.get_values(
                _central_obs(permuted).reshape(N_ENV * N, -1),
                rnn,
                masks,
            )

        torch.testing.assert_close(
            representation, representation_perm
        )
        actions = actions.reshape(N_ENV, N, ACT_DIM)
        actions_perm = actions_perm.reshape(N_ENV, N, ACT_DIM)
        torch.testing.assert_close(actions[:, 0], actions_perm[:, 0])
        torch.testing.assert_close(
            actions[:, 1:][:, permutation], actions_perm[:, 1:]
        )
        torch.testing.assert_close(values, values_perm)


def test_flat_representation_is_order_sensitive():
    policy = _policy("flat")
    teams = _team_obs(seed=11)
    _, permuted = _permuted(teams)
    with torch.no_grad():
        representation = policy.actor.population_representation(
            teams.reshape(-1, OBS_DIM)
        )
        representation_perm = policy.actor.population_representation(
            permuted.reshape(-1, OBS_DIM)
        )
    assert not torch.allclose(representation, representation_perm)


def test_all_aligned_architectures_share_actor_and_critic_contracts():
    expected_rep_dims = {
        "mean": UAV_STATE_DIM,
        "flat": UAV_STATE_DIM * K,
        "set": 32 * 3,
    }
    for architecture, representation_dim in expected_rep_dims.items():
        policy = _policy(architecture)
        assert policy.uses_grouped_batches
        assert policy.actor.major_fusion.input_dim == (
            PUBLIC_STATE_DIM + representation_dim
        )
        assert policy.actor.minor_fusion.input_dim == (
            UAV_STATE_DIM + PUBLIC_STATE_DIM + representation_dim
        )
        assert policy.critic.readout.input_dim == (
            PUBLIC_STATE_DIM + representation_dim
        )


def test_aligned_readouts_keep_legacy_mlp_contract():
    args = _args("mean")
    args.use_feature_normalization = True
    args.layer_N = 2
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )

    for readout in (
        policy.actor.major_fusion,
        policy.actor.minor_fusion,
        policy.critic.readout,
    ):
        assert hasattr(readout, "feature_norm")
        assert readout.feature_norm.normalized_shape == (
            readout.input_dim,
        )
        assert readout.mlp._layer_N == 2
        assert len(readout.mlp.fc2) == 2
        assert isinstance(readout.mlp.fc1[-1], torch.nn.LayerNorm)
        assert all(
            isinstance(block[-1], torch.nn.LayerNorm)
            for block in readout.mlp.fc2
        )


def test_set_forward_and_evaluate_log_probs_match():
    torch.manual_seed(5)
    policy = _policy("set")
    teams = _team_obs(seed=9)
    obs = teams.reshape(-1, OBS_DIM)
    cent = _central_obs(teams).reshape(N_ENV * N, -1)
    rnn, masks = _rnn_masks()
    values, actions, old_log_probs, _, _ = policy.get_actions(
        cent, obs, rnn, rnn, masks
    )
    new_values, new_log_probs, entropy = policy.evaluate_actions(
        cent,
        obs,
        rnn,
        rnn,
        actions.detach(),
        masks,
    )
    torch.testing.assert_close(old_log_probs, new_log_probs)
    torch.testing.assert_close(values, new_values)
    assert torch.isfinite(entropy)


def test_set_critic_does_not_own_or_update_encoder():
    policy = _policy("set")
    encoder_parameters = {
        id(parameter)
        for parameter in policy.actor.population_encoder.parameters()
    }
    critic_parameters = {
        id(parameter) for parameter in policy.critic.parameters()
    }
    assert encoder_parameters.isdisjoint(critic_parameters)

    teams = _team_obs(seed=10)
    cent = _central_obs(teams).reshape(N_ENV * N, -1)
    rnn, masks = _rnn_masks()
    policy.actor_optimizer.zero_grad()
    policy.critic_optimizer.zero_grad()
    values = policy.get_values(cent, rnn, masks)
    values.sum().backward()
    assert all(
        parameter.grad is None
        for parameter in policy.actor.population_encoder.parameters()
    )


def test_grouped_generator_preserves_complete_team_rows():
    buffer = SharedReplayBuffer.__new__(SharedReplayBuffer)
    t_steps, envs = 2, 2
    shape = (t_steps, envs, N)
    teams = np.zeros(
        (t_steps + 1, envs, N, OBS_DIM), dtype=np.float32
    )
    for time_id in range(t_steps + 1):
        for env_id in range(envs):
            teams[time_id, env_id, 0, 0] = 1.0
            teams[time_id, env_id, :, 4] = 10 * time_id + env_id
    buffer.obs = teams
    buffer.share_obs = np.zeros(
        (t_steps + 1, envs, N, CENT_DIM), dtype=np.float32
    )
    buffer.rnn_states = np.zeros(
        (t_steps + 1, envs, N, 1, 4), dtype=np.float32
    )
    buffer.rnn_states_critic = buffer.rnn_states.copy()
    buffer.actions = np.zeros((*shape, ACT_DIM), dtype=np.float32)
    buffer.value_preds = np.zeros(
        (t_steps + 1, envs, N, 1), dtype=np.float32
    )
    buffer.returns = np.zeros_like(buffer.value_preds)
    buffer.masks = np.ones(
        (t_steps + 1, envs, N, 1), dtype=np.float32
    )
    buffer.active_masks = buffer.masks.copy()
    buffer.action_log_probs = np.zeros((*shape, 1), dtype=np.float32)
    buffer.rewards = np.zeros((*shape, 1), dtype=np.float32)
    buffer.available_actions = None
    advantages = np.zeros((*shape, 1), dtype=np.float32)

    sample = next(
        buffer.feed_forward_generator_transformer(
            advantages, num_mini_batch=1
        )
    )
    obs_batch = sample[1].reshape(-1, N, OBS_DIM)
    assert np.all(obs_batch[:, 0, 0] == 1.0)
    assert np.all(obs_batch[:, 1:, 0] == 0.0)
    assert np.all(obs_batch[:, :, 4] == obs_batch[:, :1, 4])
