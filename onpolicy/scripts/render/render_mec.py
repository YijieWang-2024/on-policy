#!/usr/bin/env python
"""Render a MEC episode to an animated GIF.

Drives the ported finite-K HAP/UAV MEC env for one episode and draws each slot:
the drifting demand hotspot, the K UAVs with their sub-6 access coverage circles,
the mobile compute hub with its 60 GHz service circle, live backhaul/offload links
(only to UAVs inside the service circle), per-node queue fill, and a running cost
readout.

Policy source (--policy):
  auto      : load a trained shared actor from --model_dir/actor.pt; else heuristic
  heuristic : coverage(+ring)+offload heuristic (no model needed)
  random    : random actions

Examples:
  # trained policy
  python -m onpolicy.scripts.render.render_mec --model_dir <run>/models --mec_fleet_size 24
  # no model yet, just see the dynamics
  python -m onpolicy.scripts.render.render_mec --policy heuristic --mec_fleet_size 12 --out /tmp/mec.gif
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Circle    # noqa: E402
import imageio.v2 as imageio             # noqa: E402

from onpolicy.envs.mec.MEC_env import MECEnv, ACT_DIM     # noqa: E402
from onpolicy.envs.mec.config_loader import (             # noqa: E402
    access_coverage_radius_m, backhaul_service_radius_m)
from onpolicy.utils.run_config import (                    # noqa: E402
    apply_legacy_mec_arch_default, apply_saved_args,
    explicit_option_names, load_model_config)

DT = 1.0


def _hotspot_sigma_m(cfg):
    field = cfg["demand"].get("workload_field")
    if field is not None and "hotspot_sigma_m" in field:
        return float(field["hotspot_sigma_m"])
    return float(cfg["demand"]["activity_probability"]["hotspot_sigma_m"])


# --------------------------------------------------------------------- policies
def _heuristic_action(raw, ring, offset):
    st = raw.state
    R = raw.lx
    v_u = float(raw.cfg["env"]["uav"]["velocity_max_mps"])
    v_h = float(raw.cfg["env"]["hap"]["velocity_max_mps"])
    q_u_max = float(raw.cfg["env"]["uav"]["queue_max_bits"])

    def steer(cur, tgt, vmax):
        d = tgt - cur
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        return np.where(n > 0, d / np.maximum(n, 1e-9) * np.minimum(vmax, n / DT), 0.0)

    tgt = np.clip(st.demand_center_m + offset, [0, 0], [R, R])
    uv = steer(st.uav_xy_m, tgt, v_u)
    hv = steer(st.hap_xy_m[None, :], st.uav_xy_m.mean(0)[None, :], v_h)[0]
    beta = np.clip(st.uav_queue_bits / (0.2 * q_u_max), 0.0, 0.95)
    return hv, uv, beta


@np.errstate(all="ignore")
def _policy_action(policy, obs_n, rnn, masks, mec):
    import torch
    action, rnn = policy.act(obs_n, rnn, masks, deterministic=True)
    a = action.detach().cpu().numpy().reshape(mec.num_agents, ACT_DIM)
    if torch.is_tensor(rnn):
        rnn = rnn.detach().cpu().numpy()
    hv = a[0, :2] * mec.v_h_max
    uv = a[1:, :2] * mec.v_u_max
    beta = np.clip(a[1:, 2], 0.0, 1.0)
    return hv, uv, beta, rnn


# ------------------------------------------------------------------------ frame
def _draw(raw, info, cov_r, svc_r, step, cum_cost):
    R = raw.lx
    fig, ax = plt.subplots(figsize=(6, 6), dpi=96)
    ax.set_xlim(0, R); ax.set_ylim(0, R); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    cx, cy = info["demand_center_m"]
    sigma = _hotspot_sigma_m(raw.cfg)
    for r, alpha in ((2 * sigma, 0.06), (sigma, 0.10)):
        ax.add_patch(Circle((cx, cy), r, color="orange", alpha=alpha, lw=0))
    ax.plot(cx, cy, "*", color="darkorange", ms=14, label="demand hotspot")

    hx, hy = info["hap_xy_m"]
    if np.isfinite(svc_r):
        ax.add_patch(Circle((hx, hy), svc_r, fill=False, ls="--", ec="purple", lw=1.4, alpha=0.8))
    in_range = np.asarray(info["backhaul_in_range"], dtype=bool)
    beta = info["projected_action"]["beta"]

    uav_xy = info["uav_xy_m"]
    q_u_max = float(raw.cfg["env"]["uav"]["queue_max_bits"])
    qfill = np.clip(np.asarray(info["queue_bits_pre"]["uav"]) / q_u_max, 0, 1)
    for i, (ux, uy) in enumerate(uav_xy):
        ax.add_patch(Circle((ux, uy), cov_r, color="steelblue", alpha=0.05, lw=0))
        if in_range[i] and beta[i] > 0.02:                       # active offload link
            ax.plot([ux, hx], [uy, hy], color="purple", lw=0.6, alpha=0.35)
    sc = ax.scatter(uav_xy[:, 0], uav_xy[:, 1], c=qfill, cmap="RdYlGn_r",
                    vmin=0, vmax=1, s=42, edgecolors="k", linewidths=0.5, zorder=3)
    hq = info["queue_bits_pre"]["hap"] / float(raw.cfg["env"]["hap"]["queue_max_bits"])
    ax.plot(hx, hy, "P", color="purple", ms=16, mec="k", zorder=4)

    acc = float(np.sum(info["A_i"])) / 1e6
    ovf = (float(np.sum(info["D_i_U"])) + float(info["D_H"])) / 1e6
    n_in = int(in_range.sum())
    ax.set_title(
        f"t={step:3d}  cum_cost={cum_cost:6.2f}\n"
        f"accepted={acc:4.1f} Mb  overflow={ovf:4.1f} Mb  "
        f"hubQ={hq*100:3.0f}%  in-range={n_in}/{len(uav_xy)}",
        fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("UAV queue fill", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.plot([], [], "P", color="purple", ms=10, label="compute hub (+svc circle)")
    ax.plot([], [], "o", color="steelblue", label="UAV (+coverage)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return frame


# ------------------------------------------------------------------------- main
def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--mec_scenario", default="v2_iort_6km_mmwave")
    p.add_argument("--mec_fleet_size", type=int, default=None)
    p.add_argument("--policy", choices=["auto", "heuristic", "random"], default="auto")
    p.add_argument("--model_dir", default=None, help="dir containing actor.pt")
    p.add_argument("--algorithm_name", default="mappo", choices=["mappo", "rmappo", "ippo"])
    p.add_argument(
        "--mec_policy_arch",
        default="mean",
        choices=["legacy_mean", "mean", "flat", "set"],
    )
    p.add_argument("--episode_len", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260604)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", default="/tmp/mec_render.gif")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--layer_N", type=int, default=1)
    p.add_argument("--recurrent_N", type=int, default=1)
    explicit_names = explicit_option_names(argv)
    args = p.parse_args(argv)
    saved_args = load_model_config(args.model_dir)
    if saved_args:
        apply_saved_args(
            args,
            saved_args,
            explicit_names,
            skip={"model_dir", "policy", "episode_len", "fps", "out"},
        )
        print(f"loaded run config from {args.model_dir}")
    if apply_legacy_mec_arch_default(
        args, saved_args, explicit_names
    ):
        print("checkpoint predates architecture metadata; using legacy_mean")

    class _A:
        mec_scenario = args.mec_scenario
        mec_fleet_size = args.mec_fleet_size
    mec = MECEnv(_A())
    raw = mec.env
    cfg = mec.cfg
    cov_r = access_coverage_radius_m(cfg)
    svc_r = backhaul_service_radius_m(cfg)
    print(f"K={mec.k}  coverage_radius={cov_r:.0f}m  service_radius={svc_r:.0f}m  "
          f"region={raw.lx:.0f}m")

    # resolve policy source
    policy, rnn, masks = None, None, None
    mode = args.policy
    if mode == "auto":
        mode = "policy" if (args.model_dir and os.path.exists(
            os.path.join(args.model_dir, "actor.pt"))) else "heuristic"
    if mode == "policy":
        import torch
        from onpolicy.config import get_config
        base = get_config().parse_known_args([])[0]
        for k, v in vars(args).items():
            setattr(base, k, v)
        base.env_name = "MEC"
        base.use_centralized_V = True
        base.use_recurrent_policy = args.algorithm_name == "rmappo"
        base.use_naive_recurrent_policy = False
        from onpolicy.algorithms.mec.mec_policy import MECPolicy as Policy
        policy = Policy(base, mec.observation_space[0], mec.share_observation_space[0],
                        mec.action_space[0], device=torch.device("cpu"))
        sd = torch.load(os.path.join(args.model_dir, "actor.pt"), map_location="cpu")
        policy.actor.load_state_dict(sd)
        policy.actor.eval()
        rnn = np.zeros((mec.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.ones((mec.num_agents, 1), dtype=np.float32)
        print(f"loaded trained actor from {args.model_dir}/actor.pt")
    else:
        print(f"policy source: {mode} (no trained model)")

    R = raw.lx
    ring = min(_hotspot_sigma_m(cfg), 0.18 * R)
    ang = np.linspace(0, 2 * math.pi, mec.k, endpoint=False)
    offset = np.stack([ring * np.cos(ang), ring * np.sin(ang)], axis=1)

    obs_dict, _ = raw.reset(seed=args.seed)
    obs_n = mec._agent_obs(obs_dict)
    frames, cum = [], 0.0
    for t in range(args.episode_len):
        if mode == "policy":
            hv, uv, beta, rnn = _policy_action(policy, obs_n, rnn, masks, mec)
        elif mode == "heuristic":
            hv, uv, beta = _heuristic_action(raw, ring, offset)
        else:
            a = np.random.uniform(-1, 1, (mec.num_agents, ACT_DIM))
            hv, uv, beta = a[0, :2] * mec.v_h_max, a[1:, :2] * mec.v_u_max, 0.5 * (a[1:, 2] + 1)
        obs_dict, reward, _, trunc, info = raw.step(
            {"hap_velocity_mps": hv, "uav_velocity_mps": uv, "beta": beta})
        cum += -reward
        obs_n = mec._agent_obs(obs_dict)
        frames.append(_draw(raw, info, cov_r, svc_r, t, cum))
        if trunc:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"wrote {len(frames)} frames -> {args.out}  (cum_cost={cum:.2f})")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
