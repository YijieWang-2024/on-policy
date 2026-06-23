"""Multi-agent adapter: wrap FiniteKHAPUAVMECEnv into the on-policy list interface.

Homogeneous N = K+1 agent layout for the as-built D1a-folded design (spec
section 8.2): agent 0 = major/hub, agents 1..K = minor UAVs. All agents share one
14-d observation layout and one 3-d Box action; a role flag distinguishes major
from minor. The major uses action dims [0,1] as v^H and ignores dim 2; each minor
uses [0,1] as v_i and dim 2 as beta in [0,1]. The actual role routing (separate
major head + shared minor head) lives in MECActor (algorithms/mec/mec_policy.py);
the buffer stays homogeneous so this runs on the stock shared runner with only a
Box passthrough (D4, see mec_runner). The single-head D1b variant (padding + role
one-hot) was rejected; the later upgrade is the Set Transformer population
descriptor (spec section 8.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from gym import spaces
except Exception:  # pragma: no cover
    from gymnasium import spaces

from onpolicy.envs.mec.config_loader import load_scenario
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv
from onpolicy.envs.mec.metrics import demand_matching_w1

OBS_DIM = 14  # role(1) + own(3) + hub(3) + demand(4) + descriptor(3)
ACT_DIM = 3   # [vx, vy, beta_raw]; major ignores beta_raw


class MECEnv:
    """on-policy compatible HAP/UAV MEC environment (major + K minors)."""

    def __init__(self, all_args):
        scenario = getattr(all_args, "mec_scenario", "v2_iort_6km_mmwave")
        k_override = getattr(all_args, "mec_fleet_size", None)
        self.cfg = load_scenario(scenario, fleet_size_k=k_override)
        self.env = FiniteKHAPUAVMECEnv(self.cfg)

        self.k = int(self.cfg["env"]["fleet_size_k"])
        self.num_agents = self.k + 1
        self.v_h_max = float(self.cfg["env"]["hap"]["velocity_max_mps"])
        self.v_u_max = float(self.cfg["env"]["uav"]["velocity_max_mps"])

        obs_box = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        share_box = spaces.Box(-np.inf, np.inf, (OBS_DIM * self.num_agents,), dtype=np.float32)
        act_box = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        self.observation_space = [obs_box for _ in range(self.num_agents)]
        self.share_observation_space = [share_box for _ in range(self.num_agents)]
        self.action_space = [act_box for _ in range(self.num_agents)]

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options=None):
        obs, _ = self.env.reset(seed=seed)
        return self._agent_obs(obs), {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=float).reshape(self.num_agents, ACT_DIM)
        major, minors = actions[0], actions[1:]
        env_action = {
            "hap_velocity_mps": major[:2] * self.v_h_max,
            "uav_velocity_mps": minors[:, :2] * self.v_u_max,
            "beta": np.clip(minors[:, 2], 0.0, 1.0),  # MECActor emits Beta in [0,1]
        }
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs_n = self._agent_obs(obs)
        reward_n = [[float(reward)] for _ in range(self.num_agents)]
        terminated_n = [bool(terminated)] * self.num_agents          # always False (D3)
        truncated_n = [bool(truncated)] * self.num_agents
        info_n = self._agent_infos(info, float(reward))
        return obs_n, reward_n, terminated_n, truncated_n, info_n

    def seed(self, seed=None):
        if seed is not None:
            self.env.reset(seed=int(seed))

    def close(self):
        pass

    # -------------------------------------------------------------- obs build
    def _agent_obs(self, obs: dict[str, Any]) -> np.ndarray:
        hub = obs["normalized"]["hap"]            # [xy(2), q(1), demand(4)] = 7
        uav = obs["normalized"]["uavs"]           # (K, 3): [xy(2), q(1)]
        hub_pub = hub[:3]                         # hub xy + queue
        demand = hub[3:7]                         # demand center + velocity
        descriptor = uav.mean(axis=0)             # mean UAV [xy(2), q(1)]

        rows = np.zeros((self.num_agents, OBS_DIM), dtype=np.float32)
        # agent 0 = major: own block = hub state
        rows[0] = np.concatenate([[1.0], hub_pub, hub_pub, demand, descriptor])
        # agents 1..K = minors: own block = own UAV state
        for i in range(self.k):
            rows[i + 1] = np.concatenate([[0.0], uav[i], hub_pub, demand, descriptor])
        return rows

    def _agent_infos(self, info: dict[str, Any], reward: float) -> list[dict]:
        # W1 demand-matching (mass-weighted nearest-UAV distance, m) on the
        # pre-step state, matching eval_mec.py's convention so the training-time
        # mec/w1 curve is comparable to the deterministic eval number. Lower =
        # the swarm sits where the demand is.
        uav_xy = np.asarray(info["uav_xy_m"], float)
        center = np.asarray(info["demand_center_m"], float)
        w1 = demand_matching_w1(uav_xy, self.env.grid_xy, self.env._demand_density(center)[0])
        diag = info.get("access_diagnostics", {})
        regions = diag.get("regions", {})
        hot = regions.get("hotspot", {})
        bg = regions.get("background", {})
        cost = {
            "training_cost": info.get("training_cost"),
            "src_cost": info.get("src_cost_component"),
            "ovf_cost": info.get("ovf_cost_component"),
            "queue_cost": info.get("queue_cost_component"),
            "energy_cost": info.get("energy_cost_component"),
            "U_src": info.get("U_src"),
            "accepted": float(np.sum(info.get("A_i", 0.0))),
            "offloaded": float(np.sum(info.get("B_i", 0.0))),
            "overflow": float(np.sum(info.get("D_i_U", 0.0))) + float(info.get("D_H", 0.0)),
            "source_outside": float(info.get("source_loss_outside_bits", 0.0)),
            "source_capacity": float(info.get("source_loss_capacity_bits", 0.0)),
            "hotspot_offered": float(hot.get("offered_bits", 0.0)),
            "hotspot_accepted": float(hot.get("accepted_bits", 0.0)),
            "hotspot_source": float(hot.get("source_bits", 0.0)),
            "background_offered": float(bg.get("offered_bits", 0.0)),
            "background_accepted": float(bg.get("accepted_bits", 0.0)),
            "background_source": float(bg.get("source_bits", 0.0)),
            "eta_p05": float(diag.get("eta_p05", 0.0)),
            "eta_p50": float(diag.get("eta_p50", 0.0)),
            "eta_p95": float(diag.get("eta_p95", 0.0)),
            "eta_served": float(diag.get("eta_served_workload_weighted", 0.0)),
            "eta_all": float(diag.get("eta_all_workload_weighted", 0.0)),
            "n_hotspot_uav": float(diag.get("n_hotspot_uav", 0.0)),
            "n_background_uav": float(diag.get("n_background_uav", 0.0)),
            "hub_to_hotspot": float(diag.get("hub_to_hotspot_m", 0.0)),
            "w1": float(w1),
        }
        infos = [{"individual_reward": reward} for _ in range(self.num_agents)]
        infos[0].update(cost)  # team metrics ride on the major agent's info slot
        return infos


if __name__ == "__main__":
    class _Args:
        mec_scenario = "v2_iort_6km_mmwave"
        mec_fleet_size = 8  # small for a fast smoke
    env = MECEnv(_Args())
    print(f"K={env.k}  N(agents)={env.num_agents}  obs={env.observation_space[0].shape}  "
          f"act={env.action_space[0].shape}  share={env.share_observation_space[0].shape}")
    rng = np.random.default_rng(0)
    obs_n, _ = env.reset(seed=20260604)
    assert obs_n.shape == (env.num_agents, OBS_DIM), obs_n.shape
    assert np.isfinite(obs_n).all()
    tot = 0.0
    for t in range(200):
        a = rng.uniform(-1, 1, (env.num_agents, ACT_DIM))
        obs_n, r_n, term_n, trunc_n, info_n = env.step(a)
        assert obs_n.shape == (env.num_agents, OBS_DIM)
        assert np.isfinite(obs_n).all() and np.isfinite(r_n).all()
        assert len(r_n) == env.num_agents and not any(term_n)
        tot += r_n[0][0]
    assert all(trunc_n), "episode should truncate at horizon"
    m = info_n[0]
    print(f"200-step random rollout OK. sum team reward={tot:.3f}")
    print(f"  last-step metrics: accepted={m['accepted']/1e6:.1f} Mbit  "
          f"overflow={m['overflow']/1e6:.1f} Mbit  U_src={m['U_src']/1e6:.1f} Mbit  "
          f"train_cost={m['training_cost']:.4f}")
    print("MECEnv adapter smoke PASS")
