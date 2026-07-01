"""Render the learned v4 end-state deployment for a few episodes to PNG, so we can
SEE the two-layer pattern (or its failure). Overlays: demand density heatmap,
UAV positions (size ~ within hotspot), hub position, hotspot centre.
"""
import sys, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import MECEnv, ACT_DIM
from onpolicy.utils.run_config import (
    apply_legacy_mec_arch_default, apply_saved_args, load_model_config)
from onpolicy.algorithms.mec.mec_policy import MECPolicy

MD = "F:/置换不变性/YijieWang-2024-on-policy/onpolicy/scripts/results/MEC/v5_static_randinit/mappo/v5_randinit_k16/run1/models"
p = get_config(); a = p.parse_known_args(["--env_name", "MEC", "--algorithm_name", "mappo"])[0]
a.mec_scenario = "v5_static_randinit"; a.mec_fleet_size = None
s = load_model_config(MD)
if s: apply_saved_args(a, s, set(), skip={"model_dir"})
apply_legacy_mec_arch_default(a, s, set(), model_dir=MD)
a.use_recurrent_policy = False; a.use_naive_recurrent_policy = False
env = MECEnv(a); a.num_agents = env.num_agents
pol = MECPolicy(a, env.observation_space[0], env.share_observation_space[0], env.action_space[0], torch.device("cpu"))
pol.actor.load_state_dict(torch.load(MD + "/actor.pt", map_location="cpu")); pol.actor.eval()
raw = env.env; N = env.num_agents; R = 6000.0
gx = raw.grid_xy

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, ep in enumerate([0, 1, 2, 3, 5, 6]):
    obs_d, info0 = raw.reset(seed=1000 + 17 * ep); obs_n = env._agent_obs(obs_d)
    start_uav = np.asarray(raw.state.uav_xy_m, float).copy()
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
    dens = raw._demand_density(hot)[0]
    gs = int(round(math.sqrt(dens.size)))
    ax = axes[idx // 3][idx % 3]
    ax.imshow(dens.reshape(gs, gs), origin="lower", extent=[0, R, 0, R], cmap="Oranges", alpha=0.7)
    ax.scatter(start_uav[:, 0], start_uav[:, 1], c="gray", s=15, marker="s", label="UAV start", alpha=0.4)
    ax.scatter(uav[:, 0], uav[:, 1], c="blue", s=40, label="UAV end")
    ax.scatter([hub[0]], [hub[1]], c="green", s=200, marker="*", label="hub")
    ax.scatter([hot[0]], [hot[1]], c="red", s=120, marker="x", label="hotspot")
    cir = plt.Circle((hot[0], hot[1]), 600, color="red", fill=False, ls="--", alpha=0.5)
    ax.add_patch(cir)
    ax.set_xlim(0, R); ax.set_ylim(0, R); ax.set_title(f"ep{ep} hot=({hot[0]/R:.2f},{hot[1]/R:.2f})")
    if idx == 0: ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
out = "F:/置换不变性/v5_deployment.png"
plt.savefig(out, dpi=80); print(f"saved {out}")
