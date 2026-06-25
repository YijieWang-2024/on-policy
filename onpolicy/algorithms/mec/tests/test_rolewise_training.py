"""Unit tests for MEC major/minor role-balanced PPO math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO  # noqa: E402


def _trainer(rolewise=True):
    trainer = R_MAPPO.__new__(R_MAPPO)
    trainer._use_mec_rolewise_loss = rolewise
    trainer._use_policy_active_masks = True
    trainer.tpdv = dict(dtype=torch.float32, device=torch.device("cpu"))
    return trainer


def test_rolewise_policy_loss_gives_major_equal_weight():
    trainer = _trainer(rolewise=True)
    # One major sample and sixteen minor samples. The result must be the
    # equal-role mean, not the sample-count-weighted mean.
    surrogate = torch.tensor([[10.0]] + [[2.0]] * 16)
    active = torch.ones_like(surrogate)
    obs = np.zeros((17, 14), dtype=np.float32)
    obs[0, 0] = 1.0

    loss, role_losses = trainer._policy_action_loss(
        surrogate, active, obs
    )

    assert torch.isclose(loss, torch.tensor(6.0))
    assert role_losses["major_policy_loss"] == 10.0
    assert role_losses["minor_policy_loss"] == 2.0


def test_rolewise_advantages_are_normalized_per_role():
    trainer = _trainer(rolewise=True)
    advantages = np.array(
        [
            [[[1.0], [100.0], [104.0]]],
            [[[3.0], [102.0], [106.0]]],
        ],
        dtype=np.float32,
    )
    active = np.ones_like(advantages)
    obs = np.zeros((2, 1, 3, 14), dtype=np.float32)
    obs[:, :, 0, 0] = 1.0

    normalized = trainer._normalize_advantages(
        advantages, active, obs
    )
    major = normalized[:, :, 0].ravel()
    minor = normalized[:, :, 1:].ravel()

    np.testing.assert_allclose(np.mean(major), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(major), 1.0, atol=2e-5)
    np.testing.assert_allclose(np.mean(minor), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(minor), 1.0, atol=2e-5)


def test_legacy_policy_loss_remains_sample_weighted():
    trainer = _trainer(rolewise=False)
    surrogate = torch.tensor([[10.0]] + [[2.0]] * 16)
    active = torch.ones_like(surrogate)
    obs = np.zeros((17, 14), dtype=np.float32)
    obs[0, 0] = 1.0

    loss, role_losses = trainer._policy_action_loss(
        surrogate, active, obs
    )

    assert torch.isclose(loss, surrogate.mean())
    assert role_losses == {}
