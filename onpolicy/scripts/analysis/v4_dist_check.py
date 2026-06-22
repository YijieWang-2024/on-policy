"""Check the learned v4 deployment is the TWO-LAYER pattern: ~5 UAVs on the hotspot
(within ~sigma) + ~11 spread on the background, hub near hotspot centre.
Deterministic rollout to the end (static demand), averaged over episodes.
"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import MECEnv, ACT_DIM
from onpolicy.utils.run_config import apply_saved_args, load_model_config
from onpolicy.algorithms.mec.mec_policy import MECPolicy

MD = "F:/置换不变性/YijieWang-2024-on-policy/onpolicy/scripts/results/MEC/v4_static_demand/mappo/v4_static_k16/run1/models"
p = get_config(); a = p.parse_known_args(["--env_name", "MEC", "--algorithm_name", "mappo"])[0]
a.mec_scenario = "v4_static_demand"; a.mec_fleet_size = None
s = load_model_config(MD)
if s: apply_saved_args(a, s, set(), skip={"model_dir"})
a.use_recurrent_policy = False; a.use_naive_recurrent_policy = False
env = MECEnv(a); a.num_agents = env.num_agents
pol = MECPolicy(a, env.observation_space[0], env.share_observation_space[0], env.action_space[0], torch.device("cpu"))
pol.actor.load_state_dict(torch.load(MD + "/actor.pt", map_location="cpu")); pol.actor.eval()
raw = env.env; N = env.num_agents
sigma = float(env.cfg["demand"]["activity_probability"]["hotspot_sigma_m"])

print(f"K={env.k}, hotspot sigma={sigma} m. Counting UAVs within r of hotspot centre at episode end.\n")
print(f"{'ep':>3s} {'hot_centre':>14s} {'#UAV<sig':>8s} {'#UAV<1.5sig':>11s} {'#bg(>2sig)':>10s} {'hub_dist':>8s}")
nin1 = nin15 = nbg = hubd = 0.0
for ep in range(8):
    obs_d, info0 = raw.reset(seed=1000 + 17 * ep); obs_n = env._agent_obs(obs_d)
    rnn = np.zeros((N, a.recurrent_N, a.hidden_size), np.float32); m = np.ones((N, 1), np.float32)
    for _ in range(200):
        with torch.no_grad():
            ac, rnn = pol.act(obs_n, rnn, m, deterministic=True)
        ac = ac.detach().numpy().reshape(N, ACT_DIM)
        env_a = {"hap_velocity_mps": ac[0, :2] * env.v_h_max, "uav_velocity_mps": ac[1:, :2] * env.v_u_max,
                 "beta": np.clip(ac[1:, 2], 0, 1)}
        obs_d, _, _, _, info = raw.step(env_a); obs_n = env._agent_obs(obs_d)
    hot = np.asarray(info["next_demand_center_m"], float)
    uav = np.asarray(info["next_uav_xy_m"], float); hub = np.asarray(info["next_hap_xy_m"], float)
    d = np.linalg.norm(uav - hot, axis=1)
    c1 = int((d < sigma).sum()); c15 = int((d < 1.5 * sigma).sum()); cbg = int((d > 2 * sigma).sum())
    hd = float(np.linalg.norm(hub - hot))
    nin1 += c1; nin15 += c15; nbg += cbg; hubd += hd
    print(f"{ep:>3d} {str((hot/6000).round(2)):>14s} {c1:>8d} {c15:>11d} {cbg:>10d} {hd:>8.0f}")
print(f"\nMEAN over 8 ep: #UAV<sigma={nin1/8:.1f}  #UAV<1.5sigma={nin15/8:.1f}  "
      f"#UAV background(>2sigma)={nbg/8:.1f}  hub-to-hotspot={hubd/8:.0f} m")
print(f"TARGET: hotspot ~5, background ~11, hub near 0.  (phase-1 n*=5)")
