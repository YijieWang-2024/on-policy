"""Correctness tests for the major-minor MECActor / MECPolicy.

The decisive check: forward() (behavior log-prob, logpi_old) and
evaluate_actions() (update-time log-prob, logpi_new) must return identical
log-probs for the SAME (obs, action) before any gradient step. If they disagree,
the PPO importance ratio is wrong (role routing / Beta math bug). Also checks
role routing (major beta slot is a dummy, ignored) and gradient flow.

Run: /opt/anaconda3/envs/marl/bin/python -m onpolicy.algorithms.mec.tests.test_mec_policy
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces

from onpolicy.algorithms.mec.mec_policy import MECActor, MECPolicy   # noqa: E402

OBS_DIM, ACT_DIM, N_ENV, K = 14, 3, 4, 8
N = K + 1


def _args():
    return SimpleNamespace(
        hidden_size=64, layer_N=1, gain=0.01, use_orthogonal=True, use_ReLU=True,
        use_feature_normalization=False, use_policy_active_masks=True,
        use_recurrent_policy=False, use_naive_recurrent_policy=False, recurrent_N=1,
        lr=5e-4, critic_lr=5e-4, opti_eps=1e-5, weight_decay=0.0,
        use_popart=False, use_valuenorm=True, algorithm_name="mappo", stacked_frames=1,
    )


def _obs_batch(seed=0):
    """A flattened [N_ENV*N, OBS_DIM] batch with role flag in column 0
    (agent 0 of each env = major, rest = minors)."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(N_ENV, N, OBS_DIM)).astype(np.float32)
    obs[:, 0, 0] = 1.0       # major role flag
    obs[:, 1:, 0] = 0.0      # minor role flag
    return obs.reshape(N_ENV * N, OBS_DIM)


def _spaces():
    obs_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    return obs_space, act_space


def test_forward_eval_logprob_consistency():
    torch.manual_seed(0)
    obs_space, act_space = _spaces()
    actor = MECActor(_args(), obs_space, act_space, torch.device("cpu"))
    obs = _obs_batch()
    rnn = np.zeros((N_ENV * N, 1, 64), np.float32)
    masks = np.ones((N_ENV * N, 1), np.float32)

    actions, lp_old, _ = actor(obs, rnn, masks, deterministic=False)
    lp_new, entropy = actor.evaluate_actions(obs, rnn, actions.detach(), masks)

    lp_old = lp_old.detach().numpy().ravel()
    lp_new = lp_new.detach().numpy().ravel()
    assert lp_old.shape == (N_ENV * N,) and lp_new.shape == (N_ENV * N,)
    np.testing.assert_allclose(lp_old, lp_new, rtol=1e-5, atol=1e-5)
    ratio = np.exp(lp_new - lp_old)
    assert np.allclose(ratio, 1.0, atol=1e-4), f"PPO ratio not ~1: {ratio.min()}..{ratio.max()}"
    assert np.isfinite(float(entropy.detach())) and float(entropy.detach()) > 0.0
    print(f"PASS consistency: |lp_old-lp_new|max={np.abs(lp_old-lp_new).max():.2e}, "
          f"entropy={float(entropy):.3f}")


def test_role_routing_and_action_shapes():
    torch.manual_seed(1)
    obs_space, act_space = _spaces()
    actor = MECActor(_args(), obs_space, act_space, torch.device("cpu"))
    obs = _obs_batch(seed=2)
    rnn = np.zeros((N_ENV * N, 1, 64), np.float32)
    masks = np.ones((N_ENV * N, 1), np.float32)
    actions, _, _ = actor(obs, rnn, masks, deterministic=True)
    actions = actions.detach().numpy().reshape(N_ENV, N, ACT_DIM)
    assert actions.shape == (N_ENV, N, ACT_DIM)
    # major (agent 0) beta slot is a dummy 0; minors emit beta in [0,1]
    np.testing.assert_allclose(actions[:, 0, 2], 0.0, atol=1e-7)
    assert (actions[:, 1:, 2] >= -1e-6).all() and (actions[:, 1:, 2] <= 1.0 + 1e-6).all()
    print("PASS role routing: major beta=0 (ignored), minor beta in [0,1]")


def test_gradients_flow_to_both_heads():
    torch.manual_seed(2)
    obs_space, act_space = _spaces()
    actor = MECActor(_args(), obs_space, act_space, torch.device("cpu"))
    obs = _obs_batch(seed=3)
    rnn = np.zeros((N_ENV * N, 1, 64), np.float32)
    masks = np.ones((N_ENV * N, 1), np.float32)
    # PPO-style: log-prob of a FIXED (detached) action, like ratio*adv in the real loss.
    # (-log pi(rsample) would have zero grad w.r.t. a Gaussian mean by reparam cancellation,
    #  which is why the loss must score a fixed stored action.)
    actions, _, _ = actor(obs, rnn, masks)
    lp, ent = actor.evaluate_actions(obs, rnn, actions.detach(), masks)
    loss = -(lp.mean()) - 0.01 * ent
    loss.backward()
    assert actor.major_mean.weight.grad is not None and actor.major_mean.weight.grad.abs().sum() > 0
    assert actor.minor_mean.weight.grad is not None and actor.minor_mean.weight.grad.abs().sum() > 0
    assert actor.minor_beta.weight.grad is not None and actor.minor_beta.weight.grad.abs().sum() > 0
    assert actor.major_logstd.grad is not None and actor.major_logstd.grad.abs().sum() > 0
    print("PASS gradient flow: major head, minor velocity head, minor beta head all updated")


def test_policy_builds_and_acts():
    obs_space, act_space = _spaces()
    share_space = spaces.Box(-np.inf, np.inf, (OBS_DIM * N,), np.float32)
    pol = MECPolicy(_args(), obs_space, share_space, act_space, torch.device("cpu"))
    obs = _obs_batch(seed=4)
    cent = np.zeros((N_ENV * N, OBS_DIM * N), np.float32)
    rnn = np.zeros((N_ENV * N, 1, 64), np.float32)
    rnn_c = np.zeros((N_ENV * N, 1, 64), np.float32)
    masks = np.ones((N_ENV * N, 1), np.float32)
    values, actions, lp, _, _ = pol.get_actions(cent, obs, rnn, rnn_c, masks)
    v2, lp2, ent = pol.evaluate_actions(cent, obs, rnn, rnn_c, actions.detach(), masks)
    np.testing.assert_allclose(lp.detach().numpy().ravel(), lp2.detach().numpy().ravel(),
                               rtol=1e-5, atol=1e-5)
    assert values.shape == (N_ENV * N, 1)
    print("PASS MECPolicy: get_actions / evaluate_actions consistent, value shape ok")


if __name__ == "__main__":
    mod = sys.modules["__main__"]
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} MEC policy tests passed.")
