"""Checkpoint retention and fixed-validation best selection tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onpolicy.runner.shared.base_runner import Runner  # noqa: E402


def _runner(tmp_path):
    runner = Runner.__new__(Runner)
    runner.algorithm_name = "mappo"
    runner.device = torch.device("cpu")
    runner.run_dir = tmp_path / "run1"
    runner.save_dir = str(runner.run_dir / "models")
    runner.num_agents = 17
    runner.save_step_checkpoints = True
    runner.best_eval_reward = float("-inf")
    runner.all_args = SimpleNamespace(
        eval_seed=1000,
        eval_episodes=24,
        use_render=False,
        env_name="MEC",
        algorithm_name="mappo",
    )

    actor = torch.nn.Linear(3, 2)
    critic = torch.nn.Linear(4, 1)
    policy = SimpleNamespace(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
    )
    runner.policy = policy
    runner.trainer = SimpleNamespace(
        policy=policy,
        value_normalizer=None,
    )
    return runner


def test_step_and_best_checkpoints_are_retained_and_restorable(tmp_path):
    runner = _runner(tmp_path)
    original = {
        key: value.detach().clone()
        for key, value in runner.policy.actor.state_dict().items()
    }

    runner.save(total_num_steps=3200)
    latest = Path(runner.save_dir)
    step = latest / "checkpoints" / "step_000000003200"

    for directory in (latest, step):
        assert (directory / "actor.pt").exists()
        assert (directory / "critic.pt").exists()
        assert (directory / "trainer_state.pt").exists()
        assert (directory / "config.json").exists()

    assert runner.maybe_save_best(-5.0, 3200)
    best = latest / "best"
    metadata = json.loads(
        (best / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["total_num_steps"] == 3200
    assert metadata["eval_seed"] == 1000
    assert not runner.maybe_save_best(-6.0, 6400)

    with torch.no_grad():
        for parameter in runner.policy.actor.parameters():
            parameter.add_(10.0)
    runner.restore(step)
    for key, value in runner.policy.actor.state_dict().items():
        torch.testing.assert_close(value, original[key])
