#!/usr/bin/env python
"""Evaluate a MEC controller on the locked v2 env (spec section 6 science numbers).

Reports over deterministic episodes:
  - cost-component shares (coverage / overflow / queue / energy) + accept rate
  - W1 demand-matching (mass-weighted nearest-UAV distance, metres) for the
    controller vs hover / random / uniform baselines -> is the goal achieved
  - for a trained policy: hub ablation (let the learned hub move vs freeze it at
    centre) -> is the major's trajectory load-bearing under the learned policy

Examples
  # heuristic sanity (no model)
  python -m onpolicy.scripts.eval.eval_mec --env_name MEC --mec_eval_controller heuristic
  # a trained MECPolicy (pass the SAME network hyperparams used in training)
  python -m onpolicy.scripts.eval.eval_mec --env_name MEC --mec_eval_controller policy \
      --model_dir <run>/models --hidden_size 128 --layer_N 2
"""

import math
import sys

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mec.MEC_env import MECEnv, ACT_DIM
from onpolicy.envs.mec.metrics import demand_matching_w1
from onpolicy.utils.run_config import (
    apply_saved_args,
    explicit_option_names,
    load_model_config,
)


def _hotspot_sigma_m(cfg):
    field = cfg["demand"].get("workload_field")
    if field is not None and "hotspot_sigma_m" in field:
        return float(field["hotspot_sigma_m"])
    return float(cfg["demand"]["activity_probability"]["hotspot_sigma_m"])


def parse_args(args, parser):
    parser.add_argument("--mec_scenario", type=str, default="v2_iort_6km_mmwave")
    parser.add_argument("--mec_fleet_size", type=int, default=None)
    parser.add_argument("--mec_eval_controller", type=str, default="heuristic",
                        choices=["heuristic", "hover", "random", "policy"])
    parser.add_argument("--mec_eval_episodes", type=int, default=10)
    return parser.parse_known_args(args)[0]


def _heuristic_action(env, obs_dict):
    """Sunflower demand-matcher + centroid-tracking hub + queue-aware beta,
    expressed in the [-1,1]^3 action space MECEnv decodes."""
    k, R = env.k, float(env.cfg["env"]["region"]["lx_m"])
    sigma = _hotspot_sigma_m(env.cfg)
    qmax = float(env.cfg["env"]["uav"]["queue_max_bits"])
    ux = np.asarray(obs_dict["uavs"]["xy_m"], float)
    q = np.asarray(obs_dict["uavs"]["queue_bits"], float)
    hx = np.asarray(obs_dict["hap"]["xy_m"], float)
    hot = np.asarray(obs_dict["demand"]["hotspot_center_m"], float)
    rr = 1.5 * sigma * np.sqrt((np.arange(k) + 0.5) / k)
    ga = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(k)
    tgt = np.clip(hot + np.stack([rr * np.cos(ga), rr * np.sin(ga)], 1), [0, 0], [R, R])
    act = np.zeros((env.num_agents, ACT_DIM))
    # hub steers toward the UAV centroid, scaled into [-1,1]
    cen = ux.mean(0)
    act[0, :2] = np.clip((cen - hx) / max(env.v_h_max, 1e-9), -1, 1)
    act[1:, :2] = np.clip((tgt - ux) / max(env.v_u_max, 1e-9), -1, 1)
    act[1:, 2] = np.clip(q / (0.2 * qmax), 0.0, 0.95)
    return act


def _episode(env, policy, mode, args, seed, freeze_hub=False):
    raw = env.env
    obs_dict, _ = raw.reset(seed=seed)
    obs_n = env._agent_obs(obs_dict)
    N = env.num_agents
    rnn = np.zeros((N, args.recurrent_N, args.hidden_size), np.float32)
    masks = np.ones((N, 1), np.float32)
    acc = dict(train=0.0, src=0.0, ovf=0.0, q=0.0, en=0.0, U=0.0, A=0.0, w1=0.0)
    random_rng = np.random.default_rng(seed * 7 + 1)
    H = int(env.cfg["base"]["episode_horizon_slots"])
    for _ in range(H):
        ux = np.asarray(obs_dict["uavs"]["xy_m"], float)
        hot = np.asarray(obs_dict["demand"]["hotspot_center_m"], float)
        acc["w1"] += demand_matching_w1(ux, raw.grid_xy, raw._demand_density(hot)[0])

        if mode == "policy":
            with torch.no_grad():
                act, rnn = policy.act(obs_n, rnn, masks, deterministic=True)
            act = act.detach().cpu().numpy().reshape(N, ACT_DIM)
        elif mode == "heuristic":
            act = _heuristic_action(env, obs_dict)
        elif mode == "random":
            act = random_rng.uniform(-1, 1, (N, ACT_DIM))
        else:  # hover: nobody moves; queue-aware beta
            act = np.zeros((N, ACT_DIM))
            act[1:, 2] = np.clip(np.asarray(obs_dict["uavs"]["queue_bits"], float)
                                 / (0.2 * float(env.cfg["env"]["uav"]["queue_max_bits"])), 0, 0.95)

        major, minors = act[0], act[1:]
        hv = np.zeros(2) if (freeze_hub or mode == "hover") else major[:2] * env.v_h_max
        env_action = {"hap_velocity_mps": hv,
                      "uav_velocity_mps": minors[:, :2] * env.v_u_max,
                      "beta": np.clip(minors[:, 2], 0.0, 1.0)}
        obs_dict, _, _, _, info = raw.step(env_action)
        obs_n = env._agent_obs(obs_dict)
        acc["train"] += info["training_cost"]; acc["src"] += info["src_cost_component"]
        acc["ovf"] += info["ovf_cost_component"]; acc["q"] += info["queue_cost_component"]
        acc["en"] += info["energy_cost_component"]; acc["U"] += info["U_src"]
        acc["A"] += float(np.sum(info["A_i"]))
    acc["w1"] /= H
    return acc


def _avg(env, policy, mode, args, **kw):
    runs = [_episode(env, policy, mode, args, seed=args.seed + 13 * i, **kw)
            for i in range(args.mec_eval_episodes)]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def main(args):
    parser = get_config()
    explicit_names = explicit_option_names(args)
    all_args = parse_args(args, parser)
    saved_args = load_model_config(all_args.model_dir)
    if saved_args:
        apply_saved_args(
            all_args,
            saved_args,
            explicit_names,
            skip={"model_dir", "mec_eval_controller", "mec_eval_episodes"},
        )
        print(f"loaded run config from {all_args.model_dir}")
    all_args.use_recurrent_policy = all_args.algorithm_name == "rmappo"
    all_args.use_naive_recurrent_policy = False
    torch.manual_seed(all_args.seed); np.random.seed(all_args.seed)
    device = torch.device("cpu")

    env = MECEnv(all_args)
    all_args.num_agents = env.num_agents
    ctrl = all_args.mec_eval_controller

    policy = None
    if ctrl == "policy":
        if all_args.model_dir is None:
            raise ValueError("--model_dir is required for --mec_eval_controller policy")
        from onpolicy.algorithms.mec.mec_policy import MECPolicy
        policy = MECPolicy(all_args, env.observation_space[0],
                           env.share_observation_space[0], env.action_space[0], device)
        sd = torch.load(str(all_args.model_dir) + "/actor.pt", map_location=device)
        policy.actor.load_state_dict(sd)
        policy.actor.eval()

    main_m = _avg(env, policy, ctrl, all_args)
    t = main_m["train"]
    H = int(env.cfg["base"]["episode_horizon_slots"])
    print(f"\n=== eval: controller={ctrl}  K={env.k}  {all_args.mec_eval_episodes} episodes ===")
    print(f"  accept rate          : {main_m['A'] / (main_m['A'] + main_m['U']) * 100:5.1f}%")
    print(f"  share coverage (src) : {main_m['src'] / t * 100:5.1f}%")
    print(f"  share overflow (ovf) : {main_m['ovf'] / t * 100:5.1f}%")
    print(f"  share queue          : {main_m['q'] / t * 100:5.1f}%")
    print(f"  share energy         : {main_m['en'] / t * 100:5.1f}%")
    print(f"  mean team cost / slot: {t / H:.4f}")

    if ctrl == "policy":
        fix = _avg(env, policy, "policy", all_args, freeze_hub=True)
        print(f"  hub ablation (freeze@centre): {(fix['train'] / t - 1) * 100:+5.1f}%  "
              f"[>0 => learned hub trajectory load-bearing]")

    print("\n  W1 demand-matching (mass-weighted nearest-UAV distance, lower=better):")
    base = {b: _avg(env, policy, b, all_args)["w1"]
            for b in ("hover", "random") if b != ctrl}
    print(f"    {ctrl:<10}: {main_m['w1']:8.1f} m")
    for b, v in base.items():
        print(f"    {b:<10}: {v:8.1f} m")
    best = min(base.values())
    print(f"  -> demand-matching: {(1 - main_m['w1'] / max(best, 1e-9)) * 100:+.0f}% vs best baseline "
          f"({'GOAL MET' if main_m['w1'] < 0.9 * best else 'not clearly better'})")


if __name__ == "__main__":
    main(sys.argv[1:])
