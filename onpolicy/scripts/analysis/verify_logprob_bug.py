"""Empirically verify the action_log_probs shape/broadcast claim end-to-end on the
real MEC policy + SharedReplayBuffer + the exact r_mappo PPO-loss math.
"""
import json, sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
RUN = "F:/置换不变性/YijieWang-2024-on-policy/onpolicy/scripts/results/MEC/v3_iort_learnable/mappo/v3_explorefix_k12/run1"
args = SimpleNamespace(**json.load(open(RUN + "/models/config.json"))["all_args"])
args.mec_policy_arch = getattr(args, "mec_policy_arch", "legacy_mean")

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces
from onpolicy.algorithms.mec.mec_policy import MECPolicy
from onpolicy.utils.shared_buffer import SharedReplayBuffer

N = args.num_agents
obs_sp = spaces.Box(-np.inf, np.inf, (14,), np.float32)
share_sp = spaces.Box(-np.inf, np.inf, (14 * N,), np.float32)
act_sp = spaces.Box(-1.0, 1.0, (3,), np.float32)
torch.manual_seed(0); np.random.seed(0)
policy = MECPolicy(args, obs_sp, share_sp, act_sp, torch.device("cpu"))

# --- 1) what shape does the actor emit? ---
B = 16
o = np.random.randn(B, 14).astype(np.float32)
so = np.random.randn(B, 14 * N).astype(np.float32)
rnn = np.zeros((B, args.recurrent_N, args.hidden_size), np.float32)
m = np.ones((B, 1), np.float32)
val, action, alp, _, _ = policy.get_actions(so, o, rnn, rnn, m)
print(f"[1] actor get_actions -> action_log_probs shape = {tuple(alp.shape)}   (joint, expect [B,1])")

# --- 2) how does SharedReplayBuffer store it? ---
buf = SharedReplayBuffer(args, N, obs_sp, share_sp, act_sp)
print(f"[2] buffer.action_log_probs last dim = {buf.action_log_probs.shape[-1]}   "
      f"(act_shape, expect 3)")
# emulate insert: distinct per-agent joint logp of width 1 -> broadcast into width-3 slot
lp_step = np.arange(N, dtype=np.float32).reshape(N, 1)            # [N,1] distinct per agent
buf.action_log_probs[0, 0] = lp_step                              # [N,1] -> [N,3] slot
print(f"    stored agent3 row = {buf.action_log_probs[0,0,3]}   (expect 3 identical copies of 3.0)")

# --- 3) the exact r_mappo PPO-loss math: [.,3] (current) vs [.,1] (correct) ---
obs_t = torch.tensor(o); rnn_t = torch.tensor(rnn); m_t = torch.tensor(m)
act_t = action.detach().clone()
new_lp, ent = policy.actor.evaluate_actions(obs_t, rnn_t, act_t, m_t)   # [B,1]
old_lp1 = (new_lp.detach() + 0.05 * torch.randn(B, 1))                  # correct width-1 "old"
old_lp3 = old_lp1.repeat(1, 3)                                          # buffer's broadcast copies
adv = torch.randn(B, 1)
clip = args.clip_param

def ppo_policy_loss(old_lp):
    r = torch.exp(new_lp - old_lp)                 # broadcasts to old_lp's width
    s1 = r * adv
    s2 = torch.clamp(r, 1 - clip, 1 + clip) * adv
    return -torch.sum(torch.min(s1, s2), dim=-1, keepdim=True).mean(), r.shape

L3, sh3 = ppo_policy_loss(old_lp3)
L1, sh1 = ppo_policy_loss(old_lp1)
print(f"[3] ratio shape current(old=[.,3]) = {tuple(sh3)} ; correct(old=[.,1]) = {tuple(sh1)}")
print(f"    policy_loss current = {L3.item():.6f} ; correct = {L1.item():.6f} ; "
      f"ratio = {L3.item()/L1.item():.4f}x")

# --- 4) gradient magnitude effect (policy term only) ---
for label, L in [("current(3col)", L3), ("correct(1col)", L1)]:
    policy.actor.zero_grad()
    L.backward(retain_graph=True)
    g = np.sqrt(sum((p.grad.norm().item() ** 2) for p in policy.actor.parameters() if p.grad is not None))
    print(f"    actor grad-norm {label} = {g:.5f}")
print("\nNote: with Adam, a global kx on the policy gradient ~cancels (m/sqrt(v) scale-invariant); "
      "the material residue is that the SEPARATE entropy term is then ~3x under-weighted.")
