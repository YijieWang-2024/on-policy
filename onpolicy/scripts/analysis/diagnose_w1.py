"""Diagnose WHY low-cost != low-W1 across the 3 final-config seeds.

For each trained seed (deterministic rollout, shared eval seeds) collect, per slot:
  cost components (src/ovf/queue/energy), accepted, W1, mean beta, mean |v|.
Then:
  1) per-seed averages -> how does each seed spend its cost budget / move?
  2) across the pooled (slot-level) data: correlation of W1 with team cost and
     with each component -> is W1 even coupled to the objective?
A near-zero corr(W1, cost) means W1 is a weak/again-misaligned signal in v3
(=> shaping is the right fix). A strong negative corr that the policy fails to
exploit would instead point to a credit-assignment / exploration problem.
"""
import sys, math
import numpy as np
import torch
sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import MECEnv, ACT_DIM
from onpolicy.envs.mec.metrics import demand_matching_w1
from onpolicy.utils.run_config import apply_saved_args, load_model_config
from onpolicy.algorithms.mec.mec_policy import MECPolicy

BASE = "F:/置换不变性/YijieWang-2024-on-policy/onpolicy/scripts/results/MEC/v3_iort_learnable/mappo/v3_logpfix_ent003"
SEEDS = {"seed1": f"{BASE}/run1/models", "seed2": f"{BASE}/run2/models", "seed3": f"{BASE}/run3/models"}
EPISODES = 8


def load_policy(model_dir):
    parser = get_config()
    a = parser.parse_known_args(["--env_name", "MEC", "--algorithm_name", "mappo"])[0]
    a.mec_scenario = "v3_iort_learnable"; a.mec_fleet_size = None
    saved = load_model_config(model_dir)
    if saved:
        apply_saved_args(a, saved, set(), skip={"model_dir"})
    a.use_recurrent_policy = False; a.use_naive_recurrent_policy = False
    dev = torch.device("cpu")
    env = MECEnv(a)
    a.num_agents = env.num_agents
    pol = MECPolicy(a, env.observation_space[0], env.share_observation_space[0], env.action_space[0], dev)
    pol.actor.load_state_dict(torch.load(model_dir + "/actor.pt", map_location=dev))
    pol.actor.eval()
    return env, pol, a


def rollout(env, pol, a, seed):
    raw = env.env
    obs_d, _ = raw.reset(seed=seed)
    obs_n = env._agent_obs(obs_d)
    N = env.num_agents
    rnn = np.zeros((N, a.recurrent_N, a.hidden_size), np.float32)
    masks = np.ones((N, 1), np.float32)
    H = int(env.cfg["base"]["episode_horizon_slots"])
    rows = []
    for _ in range(H):
        ux = np.asarray(obs_d["uavs"]["xy_m"], float)
        hot = np.asarray(obs_d["demand"]["hotspot_center_m"], float)
        w1 = demand_matching_w1(ux, raw.grid_xy, raw._demand_density(hot)[0])
        with torch.no_grad():
            act, rnn = pol.act(obs_n, rnn, masks, deterministic=True)
        act = act.detach().cpu().numpy().reshape(N, ACT_DIM)
        beta = float(np.clip(act[1:, 2], 0, 1).mean())
        vmag = float(np.linalg.norm(act[1:, :2], axis=1).mean())
        env_a = {"hap_velocity_mps": act[0, :2] * env.v_h_max,
                 "uav_velocity_mps": act[1:, :2] * env.v_u_max,
                 "beta": np.clip(act[1:, 2], 0, 1)}
        obs_d, _, _, _, info = raw.step(env_a)
        obs_n = env._agent_obs(obs_d)
        rows.append([info["training_cost"], info["src_cost_component"], info["ovf_cost_component"],
                     info["queue_cost_component"], info["energy_cost_component"],
                     float(np.sum(info["A_i"])), w1, beta, vmag])
    return np.array(rows)


pooled = []
print(f"{'seed':6s} {'cost':>6s} {'src':>5s} {'ovf':>5s} {'queue':>5s} {'enrg':>5s} "
      f"{'acc':>5s} {'W1':>6s} {'beta':>5s} {'|v|':>5s}")
for name, md in SEEDS.items():
    env, pol, a = load_policy(md)
    allr = []
    for i in range(EPISODES):
        allr.append(rollout(env, pol, a, seed=a.seed + 13 * i))
    R = np.concatenate(allr, 0)
    pooled.append(R)
    mu = R.mean(0)
    print(f"{name:6s} {mu[0]:6.3f} {mu[1]:5.3f} {mu[2]:5.3f} {mu[3]:5.3f} {mu[4]:5.3f} "
          f"{mu[5]/1e6:5.1f} {mu[6]:6.0f} {mu[7]:5.2f} {mu[8]:5.2f}")

P = np.concatenate(pooled, 0)
cols = ["cost", "src", "ovf", "queue", "energy", "accepted", "W1", "beta", "|v|"]
w1 = P[:, 6]
print("\n--- slot-level correlation of W1 with each quantity (pooled, n={}) ---".format(len(P)))
for j, c in enumerate(cols):
    if c == "W1":
        continue
    r = np.corrcoef(w1, P[:, j])[0, 1]
    print(f"  corr(W1, {c:8s}) = {r:+.3f}")
print("\nKey: corr(W1,cost)~0 => W1 decoupled from objective in v3 (shaping needed).")
print("     corr(W1,src)>0  => only the coverage term rewards demand-matching.")
