"""Structural tests for aligned Mean/Flat/Set MEC policies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces

from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.algorithms.mec.set_networks import chamfer_set_loss
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
        mec_set_encoder_type="latent_slots",
        mec_set_dim=32,
        mec_set_heads=4,
        mec_set_num_seeds=3,
        mec_set_element_blocks=2,
        mec_set_latent_blocks=1,
        mec_set_critic_encoder="actor_detached",
        mec_set_actor_encoder="shared",
        mec_set_actor_context="pooled",
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


def test_set_population_encoders_keep_mlp_contract():
    for encoder_type in ("latent_slots", "mean_pool"):
        args = _args("set")
        args.mec_set_encoder_type = encoder_type
        args.use_feature_normalization = True
        args.layer_N = 2
        policy = MECPolicy(
            args,
            *_spaces(),
            torch.device("cpu"),
            num_agents=N,
        )
        encoder = policy.actor.population_encoder
        assert hasattr(encoder, "atom_norm")
        assert encoder.atom_norm.normalized_shape == (UAV_STATE_DIM,)
        assert encoder.atom_encoder._layer_N == 2
        assert len(encoder.atom_encoder.fc2) == 2
        assert isinstance(
            encoder.atom_encoder.fc1[-1], torch.nn.LayerNorm
        )
        assert all(
            isinstance(block[-1], torch.nn.LayerNorm)
            for block in encoder.atom_encoder.fc2
        )


def test_set_flat_mlp_encoder_contract_is_order_sensitive():
    args = _args("set")
    args.mec_set_encoder_type = "flat_mlp"
    args.use_feature_normalization = True
    args.layer_N = 2
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    encoder = policy.actor.population_encoder
    assert hasattr(encoder, "atom_norm")
    assert encoder.atom_norm.normalized_shape == (K * UAV_STATE_DIM,)
    assert encoder.atom_encoder._layer_N == 2
    assert len(encoder.atom_encoder.fc2) == 2
    assert isinstance(encoder.atom_encoder.fc1[-1], torch.nn.LayerNorm)
    assert all(
        isinstance(block[-1], torch.nn.LayerNorm)
        for block in encoder.atom_encoder.fc2
    )

    teams = _team_obs(seed=17)
    _, permuted = _permuted(teams)
    with torch.no_grad():
        representation = policy.actor.population_representation(
            teams.reshape(-1, OBS_DIM)
        )
        representation_perm = policy.actor.population_representation(
            permuted.reshape(-1, OBS_DIM)
        )
    assert representation.shape == (N_ENV, args.mec_set_dim)
    assert not torch.allclose(representation, representation_perm)


def test_chamfer_set_loss_is_target_order_invariant():
    prediction = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, 0.2]]]
    )
    target = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [0.0, 0.0, 0.0]]]
    )
    target_permuted = target[:, [2, 0, 1]]

    torch.testing.assert_close(
        chamfer_set_loss(prediction, target),
        chamfer_set_loss(prediction, target_permuted),
    )


def test_set_mean_pool_reconstruction_aux_is_unordered_and_trains_encoder():
    args = _args("set")
    args.mec_set_encoder_type = "mean_pool"
    args.mec_set_reconstruction_coef = 0.25
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    assert hasattr(policy.actor, "reconstruction_decoder")
    decoder_parameters = {
        id(parameter)
        for parameter in policy.actor.reconstruction_decoder.parameters()
    }
    actor_optimizer_parameters = {
        id(parameter)
        for group in policy.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    assert decoder_parameters <= actor_optimizer_parameters

    teams = _team_obs(seed=18)
    _, permuted = _permuted(teams)
    cent = _central_obs(teams).reshape(N_ENV * N, -1)
    cent_permuted = _central_obs(permuted).reshape(N_ENV * N, -1)

    loss = policy.mec_set_reconstruction_loss(cent)
    loss_permuted = policy.mec_set_reconstruction_loss(cent_permuted)
    torch.testing.assert_close(loss, loss_permuted)

    policy.actor_optimizer.zero_grad()
    loss.backward()

    def has_nonzero_grad(parameters):
        return any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in parameters
        )

    assert has_nonzero_grad(policy.actor.population_encoder.parameters())
    assert has_nonzero_grad(
        policy.actor.reconstruction_decoder.parameters()
    )


def test_set_pretrained_actor_loader_filters_representation_weights():
    source_args = _args("set")
    source_args.mec_set_encoder_type = "mean_pool"
    source_args.mec_set_reconstruction_coef = 0.25
    source = MECPolicy(
        source_args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    with torch.no_grad():
        for parameter in source.actor.population_encoder.parameters():
            parameter.fill_(0.123)
        for parameter in source.actor.reconstruction_decoder.parameters():
            parameter.fill_(0.234)
        for parameter in source.actor.major_fusion.parameters():
            parameter.fill_(0.345)

    target_args = _args("set")
    target_args.mec_set_encoder_type = "mean_pool"
    target_args.mec_set_reconstruction_coef = 0.25
    target = MECPolicy(
        target_args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    readout_before = {
        key: value.clone()
        for key, value in target.actor.state_dict().items()
        if key.startswith("major_fusion.")
    }

    loaded = target.load_set_pretrained_actor(source.actor.state_dict())

    assert loaded
    assert all(
        key.startswith(("population_encoder.", "reconstruction_decoder."))
        for key in loaded
    )
    for parameter in target.actor.population_encoder.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.123),
        )
    for parameter in target.actor.reconstruction_decoder.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.234),
        )
    for key, value in target.actor.state_dict().items():
        if key.startswith("major_fusion."):
            torch.testing.assert_close(value, readout_before[key])


def test_set_encoder_trainable_toggle_only_affects_actor_encoder():
    args = _args("set")
    args.mec_set_encoder_type = "mean_pool"
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    changed = policy.set_set_encoder_trainable(False)
    assert changed > 0
    assert all(
        not parameter.requires_grad
        for parameter in policy.actor.population_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in policy.actor.major_fusion.parameters()
    )
    policy.set_set_encoder_trainable(True)
    assert all(
        parameter.requires_grad
        for parameter in policy.actor.population_encoder.parameters()
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


def test_set_relational_actor_context_is_equivariant():
    args = _args("set")
    args.mec_set_actor_context = "relational"
    torch.manual_seed(6)
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    assert policy.actor.major_fusion.input_dim == (
        PUBLIC_STATE_DIM + args.mec_set_dim * args.mec_set_num_seeds
    )
    assert policy.actor.minor_fusion.input_dim == (
        UAV_STATE_DIM
        + PUBLIC_STATE_DIM
        + args.mec_set_dim * args.mec_set_num_seeds
        + args.mec_set_dim
    )

    teams = _team_obs(seed=13)
    permutation, permuted = _permuted(teams)
    rnn, masks = _rnn_masks()
    with torch.no_grad():
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
    actions = actions.reshape(N_ENV, N, ACT_DIM)
    actions_perm = actions_perm.reshape(N_ENV, N, ACT_DIM)
    torch.testing.assert_close(actions[:, 0], actions_perm[:, 0])
    torch.testing.assert_close(
        actions[:, 1:][:, permutation], actions_perm[:, 1:]
    )


def test_set_mean_pool_encoder_contract_is_invariant():
    args = _args("set")
    args.mec_set_encoder_type = "mean_pool"
    torch.manual_seed(7)
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    representation_dim = args.mec_set_dim
    assert policy.actor.major_fusion.input_dim == (
        PUBLIC_STATE_DIM + representation_dim
    )
    assert policy.actor.minor_fusion.input_dim == (
        UAV_STATE_DIM + PUBLIC_STATE_DIM + representation_dim
    )
    assert policy.critic.readout.input_dim == (
        PUBLIC_STATE_DIM + representation_dim
    )

    teams = _team_obs(seed=14)
    permutation, permuted = _permuted(teams)
    rnn, masks = _rnn_masks()
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

    assert representation.shape == (N_ENV, representation_dim)
    torch.testing.assert_close(representation, representation_perm)
    actions = actions.reshape(N_ENV, N, ACT_DIM)
    actions_perm = actions_perm.reshape(N_ENV, N, ACT_DIM)
    torch.testing.assert_close(actions[:, 0], actions_perm[:, 0])
    torch.testing.assert_close(
        actions[:, 1:][:, permutation], actions_perm[:, 1:]
    )
    torch.testing.assert_close(values, values_perm)


def test_set_critic_does_not_own_or_update_encoder():
    policy = _policy("set")
    assert policy.recompute_values_after_actor_update
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


def test_set_shared_grad_critic_updates_actor_encoder():
    args = _args("set")
    args.mec_set_critic_encoder = "shared_grad"
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    assert policy.recompute_values_after_actor_update
    assert policy.critic.population_encoder is policy.actor.population_encoder

    encoder_parameters = {
        id(parameter)
        for parameter in policy.actor.population_encoder.parameters()
    }
    actor_optimizer_parameters = {
        id(parameter)
        for group in policy.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    critic_optimizer_parameters = {
        id(parameter)
        for group in policy.critic_optimizer.param_groups
        for parameter in group["params"]
    }
    assert encoder_parameters <= actor_optimizer_parameters
    assert encoder_parameters <= critic_optimizer_parameters

    teams = _team_obs(seed=15)
    cent = _central_obs(teams).reshape(N_ENV * N, -1)
    rnn, masks = _rnn_masks()
    policy.actor_optimizer.zero_grad()
    policy.critic_optimizer.zero_grad()
    values = policy.get_values(cent, rnn, masks)
    values.sum().backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in policy.actor.population_encoder.parameters()
    )


def test_set_separate_critic_encoder_gets_value_gradients():
    args = _args("set")
    args.mec_set_critic_encoder = "separate"
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )
    assert not policy.recompute_values_after_actor_update
    assert hasattr(policy.critic, "population_encoder")

    actor_encoder_parameters = {
        id(parameter)
        for parameter in policy.actor.population_encoder.parameters()
    }
    critic_encoder_parameters = {
        id(parameter)
        for parameter in policy.critic.population_encoder.parameters()
    }
    assert actor_encoder_parameters.isdisjoint(critic_encoder_parameters)

    teams = _team_obs(seed=12)
    cent = _central_obs(teams).reshape(N_ENV * N, -1)
    rnn, masks = _rnn_masks()
    policy.actor_optimizer.zero_grad()
    policy.critic_optimizer.zero_grad()
    values = policy.get_values(cent, rnn, masks)
    values.sum().backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in policy.critic.population_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in policy.actor.population_encoder.parameters()
    )


def test_set_separate_actor_encoders_route_role_gradients():
    args = _args("set")
    args.mec_set_encoder_type = "mean_pool"
    args.mec_set_actor_encoder = "separate"
    args.mec_set_critic_encoder = "separate"
    policy = MECPolicy(
        args,
        *_spaces(),
        torch.device("cpu"),
        num_agents=N,
    )

    major_encoder_parameters = {
        id(parameter)
        for parameter in policy.actor.major_population_encoder.parameters()
    }
    minor_encoder_parameters = {
        id(parameter)
        for parameter in policy.actor.minor_population_encoder.parameters()
    }
    critic_encoder_parameters = {
        id(parameter)
        for parameter in policy.critic.population_encoder.parameters()
    }
    actor_optimizer_parameters = {
        id(parameter)
        for group in policy.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    critic_optimizer_parameters = {
        id(parameter)
        for group in policy.critic_optimizer.param_groups
        for parameter in group["params"]
    }
    assert major_encoder_parameters.isdisjoint(minor_encoder_parameters)
    assert major_encoder_parameters.isdisjoint(critic_encoder_parameters)
    assert minor_encoder_parameters.isdisjoint(critic_encoder_parameters)
    assert major_encoder_parameters <= actor_optimizer_parameters
    assert minor_encoder_parameters <= actor_optimizer_parameters
    assert critic_encoder_parameters <= critic_optimizer_parameters
    assert major_encoder_parameters.isdisjoint(critic_optimizer_parameters)
    assert minor_encoder_parameters.isdisjoint(critic_optimizer_parameters)

    teams = _team_obs(seed=16)
    obs = teams.reshape(-1, OBS_DIM)
    cent = _central_obs(teams).reshape(N_ENV * N, -1)
    rnn, masks = _rnn_masks()
    _, actions, _, _, _ = policy.get_actions(
        cent, obs, rnn, rnn, masks
    )

    def has_nonzero_grad(parameters):
        return any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in parameters
        )

    is_major = torch.as_tensor(obs[:, 0:1] > 0.5)
    policy.actor_optimizer.zero_grad()
    _, log_probs, _ = policy.evaluate_actions(
        cent,
        obs,
        rnn,
        rnn,
        actions.detach(),
        masks,
    )
    (-log_probs[is_major[:, 0]].mean()).backward()
    assert has_nonzero_grad(
        policy.actor.major_population_encoder.parameters()
    )
    assert not has_nonzero_grad(
        policy.actor.minor_population_encoder.parameters()
    )

    policy.actor_optimizer.zero_grad()
    _, log_probs, _ = policy.evaluate_actions(
        cent,
        obs,
        rnn,
        rnn,
        actions.detach(),
        masks,
    )
    (-log_probs[~is_major[:, 0]].mean()).backward()
    assert not has_nonzero_grad(
        policy.actor.major_population_encoder.parameters()
    )
    assert has_nonzero_grad(
        policy.actor.minor_population_encoder.parameters()
    )


def test_set_separate_actor_encoder_requires_separate_critic_encoder():
    args = _args("set")
    args.mec_set_actor_encoder = "separate"
    args.mec_set_critic_encoder = "actor_detached"
    with pytest.raises(ValueError, match="requires"):
        MECPolicy(
            args,
            *_spaces(),
            torch.device("cpu"),
            num_agents=N,
        )


def test_set_reconstruction_aux_requires_shared_actor_encoder():
    args = _args("set")
    args.mec_set_encoder_type = "mean_pool"
    args.mec_set_reconstruction_coef = 0.25
    args.mec_set_actor_encoder = "separate"
    args.mec_set_critic_encoder = "separate"
    with pytest.raises(ValueError, match="reconstruction"):
        MECPolicy(
            args,
            *_spaces(),
            torch.device("cpu"),
            num_agents=N,
        )


def test_set_flat_mlp_rejects_relational_actor_context():
    args = _args("set")
    args.mec_set_encoder_type = "flat_mlp"
    args.mec_set_actor_context = "relational"
    with pytest.raises(ValueError, match="flat_mlp"):
        MECPolicy(
            args,
            *_spaces(),
            torch.device("cpu"),
            num_agents=N,
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
