"""Regression tests for MEC training versus validation logging."""

from onpolicy.runner.shared.mec_runner import MECRunner


class _Writer:
    def __init__(self):
        self.logged = []

    def add_scalar(self, key, value, step):
        self.logged.append((key, value, step))


def test_validation_logging_does_not_reuse_stale_training_infos():
    runner = MECRunner.__new__(MECRunner)
    runner.use_wandb = False
    runner.writter = _Writer()
    runner._last_infos = [
        {
            "agent_infos": [
                {
                    "training_cost": 9.0,
                    "accepted": 1.0,
                    "U_src": 2.0,
                }
            ]
        }
    ]

    env_infos = {"eval_average_episode_rewards": [-3.0]}
    runner.log_env(env_infos, total_num_steps=100)

    assert list(env_infos) == ["eval_average_episode_rewards"]
    assert runner.writter.logged == [
        (
            "eval_average_episode_rewards",
            -3.0,
            100,
        )
    ]

