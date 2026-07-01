"""Diagnose WHY the learned major (hub) head failed (ablation -8.4% though the hub
is +111% load-bearing). Three hypotheses, measured on the trained seed1 policy:

H1 exploration collapse : major_logstd value (did the 2-d hub Gaussian shrink to ~0?)
H2 dead observation     : does major MEAN velocity respond to "swarm centroid is
                          off from where it should be"? Feed controlled obs where the
                          hub is displaced from the UAV centroid and see if it steers back.
H3 critic/credit        : compare major's mean |action| and effective movement vs minor.
Also: replay an episode and log how far the LEARNED hub drifts vs the UAV centroid
(if it lags badly, the major policy is the bottleneck).
"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import MECEnv, ACT_DIM
from onpolicy.utils.run_config import (
    apply_legacy_mec_arch_default, apply_saved_args, load_model_config)
from onpolicy.algorithms.mec.mec_policy import MECPolicy

MD = "F:/置换不变性/YijieWang-2024-on-policy/onpolicy/scripts/results/MEC/v3_iort_learnable/mappo/v3rc_k12/run1/models"


def load():
    p = get_config(); a = p.parse_known_args(["--env_name", "MEC", "--algorithm_name", "mappo"])[0]
    a.mec_scenario = "v3_iort_learnable"; a.mec_fleet_size = None
    s = load_model_config(MD)
    if s: apply_saved_args(a, s, set(), skip={"model_dir"})
    apply_legacy_mec_arch_default(a, s, set(), model_dir=MD)
    a.use_recurrent_policy = False; a.use_naive_recurrent_policy = False
    env = MECEnv(a); a.num_agents = env.num_agents
    pol = MECPolicy(a, env.observation_space[0], env.share_observation_space[0], env.action_space[0], torch.device("cpu"))
    pol.actor.load_state_dict(torch.load(MD + "/actor.pt", map_location="cpu")); pol.actor.eval()
    return env, pol, a


env, pol, a = load()
act = pol.actor

# H1: exploration scale
print("=== H1: exploration (logstd -> sigma) ===")
print(f"  major_logstd = {act.major_logstd.detach().numpy()}  -> sigma {np.exp(act.major_logstd.detach().numpy())}")
print(f"  minor_logstd = {act.minor_logstd.detach().numpy()}  -> sigma {np.exp(act.minor_logstd.detach().numpy())}")

# H2: does major mean velocity steer the hub toward the UAV centroid when displaced?
# Build a major obs row: [role=1, own(hub xy,q), hub_pub(xy,q), demand(4), descriptor(uav mean xy,q)]
# Vary hub position relative to a fixed UAV-centroid (descriptor); see if mean vel points to centroid.
print("\n=== H2: major mean velocity vs hub-offset-from-centroid (should steer toward centroid) ===")
rnn = np.zeros((1, a.recurrent_N, a.hidden_size), np.float32); m = np.ones((1, 1), np.float32)
cen = np.array([0.5, 0.5])  # normalized UAV centroid at region center
for off, name in [((0.2, 0.0), "hub EAST of centroid"), ((-0.2, 0.0), "hub WEST"),
                  ((0.0, 0.2), "hub NORTH"), ((0.0, -0.2), "hub SOUTH")]:
    hub_xy = cen + np.array(off)
    obs = np.zeros((1, 14), np.float32)
    obs[0, 0] = 1.0                       # role=major
    obs[0, 1:3] = hub_xy; obs[0, 3] = 0.0  # own = hub xy,q
    obs[0, 4:6] = hub_xy; obs[0, 6] = 0.0  # hub_pub
    obs[0, 7:11] = [cen[0], cen[1], 0, 0]  # demand center at centroid (hub should go there for backhaul)
    obs[0, 11:13] = cen; obs[0, 13] = 0.0  # descriptor = UAV centroid
    with torch.no_grad():
        action, _, _ = act(obs, rnn, m, deterministic=True)
    v = action[0, :2].numpy()
    toward = -np.array(off) / (np.linalg.norm(off) + 1e-9)  # unit vector hub->centroid
    align = float(np.dot(v / (np.linalg.norm(v) + 1e-9), toward))
    print(f"  {name:22s}: mean_v=[{v[0]:+.3f},{v[1]:+.3f}]  align-with-toward-centroid={align:+.2f}")

# H3: replay an episode, measure hub lag behind UAV centroid under the learned policy
print("\n=== H3: learned hub lag behind UAV centroid (deterministic episode) ===")
raw = env.env
obs_d, _ = raw.reset(seed=1); obs_n = env._agent_obs(obs_d)
N = env.num_agents; rnn = np.zeros((N, a.recurrent_N, a.hidden_size), np.float32); m = np.ones((N, 1), np.float32)
lags = []; hub_speed = []
for t in range(200):
    with torch.no_grad():
        ac, rnn = pol.act(obs_n, rnn, m, deterministic=True)
    ac = ac.detach().numpy().reshape(N, ACT_DIM)
    env_a = {"hap_velocity_mps": ac[0, :2] * env.v_h_max, "uav_velocity_mps": ac[1:, :2] * env.v_u_max,
             "beta": np.clip(ac[1:, 2], 0, 1)}
    obs_d, _, _, _, info = raw.step(env_a); obs_n = env._agent_obs(obs_d)
    hub = np.asarray(info["next_hap_xy_m"], float); cen = np.asarray(info["next_uav_xy_m"], float).mean(0)
    lags.append(float(np.linalg.norm(hub - cen)))
    hub_speed.append(float(np.linalg.norm(ac[0, :2]) * env.v_h_max))
print(f"  mean hub-to-centroid distance = {np.mean(lags):.0f} m  (backhaul radius ~2.2 km; >2200 => UAVs drop out)")
print(f"  mean hub speed = {np.mean(hub_speed):.1f} m/s  (V_H_max=30; ~0 => hub barely moves)")
print(f"  hub-centroid dist: start {lags[0]:.0f} -> end {lags[-1]:.0f} m")
