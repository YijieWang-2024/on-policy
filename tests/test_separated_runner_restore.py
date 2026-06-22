from types import SimpleNamespace

import gymnasium as gym
import torch

from onpolicy.runner.separated.base_runner import Runner


class _FakeModule:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


class _FakePolicy:
    instances = []

    def __init__(self, args, obs_space, share_obs_space, act_space, device):
        self.actor = _FakeModule()
        self.critic = _FakeModule()
        _FakePolicy.instances.append(self)


class _FakeValueNormalizer:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state_dict):
        self.loaded_state = state_dict


class _FakeTrainer:
    instances = []

    def __init__(self, args, policy, device):
        self.policy = policy
        self._use_valuenorm = True
        self.value_normalizer = _FakeValueNormalizer()
        _FakeTrainer.instances.append(self)


class _FakeEnv:
    def __init__(self, num_agents):
        obs = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        act = gym.spaces.Discrete(2)
        self.observation_space = [obs for _ in range(num_agents)]
        self.share_observation_space = [obs for _ in range(num_agents)]
        self.action_space = [act for _ in range(num_agents)]


def _args(model_dir):
    return SimpleNamespace(
        env_name="MPE",
        algorithm_name="mappo",
        experiment_name="restore_test",
        use_centralized_V=True,
        use_obs_instead_of_state=False,
        num_env_steps=1,
        episode_length=1,
        n_rollout_threads=1,
        n_eval_rollout_threads=1,
        use_linear_lr_decay=False,
        hidden_size=1,
        use_wandb=False,
        use_render=False,
        recurrent_N=1,
        save_interval=1,
        use_eval=False,
        eval_interval=1,
        log_interval=1,
        model_dir=str(model_dir),
        gamma=0.99,
        gae_lambda=0.95,
        use_gae=True,
        use_popart=False,
        use_valuenorm=True,
    )


def test_separated_runner_restores_after_trainers_exist(tmp_path, monkeypatch):
    import onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy as policy_module
    import onpolicy.algorithms.r_mappo.r_mappo as trainer_module

    _FakePolicy.instances = []
    _FakeTrainer.instances = []
    monkeypatch.setattr(policy_module, "R_MAPPOPolicy", _FakePolicy)
    monkeypatch.setattr(trainer_module, "R_MAPPO", _FakeTrainer)

    num_agents = 2
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    for agent_id in range(num_agents):
        torch.save({"agent": torch.tensor(agent_id)}, model_dir / f"actor_agent{agent_id}.pt")
        torch.save({"agent": torch.tensor(agent_id)}, model_dir / f"critic_agent{agent_id}.pt")
        torch.save({"agent": torch.tensor(agent_id)}, model_dir / f"vnrom_agent{agent_id}.pt")

    runner = Runner(
        {
            "all_args": _args(model_dir),
            "envs": _FakeEnv(num_agents),
            "eval_envs": None,
            "num_agents": num_agents,
            "device": torch.device("cpu"),
            "run_dir": tmp_path / "run",
        }
    )
    runner.writter.close()

    assert len(_FakeTrainer.instances) == num_agents
    for agent_id in range(num_agents):
        assert _FakePolicy.instances[agent_id].actor.loaded_state["agent"].item() == agent_id
        assert _FakePolicy.instances[agent_id].critic.loaded_state["agent"].item() == agent_id
        assert (
            _FakeTrainer.instances[agent_id]
            .value_normalizer.loaded_state["agent"]
            .item()
            == agent_id
        )
