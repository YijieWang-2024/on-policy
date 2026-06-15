"""Health check on the PORTED env (spec section 6 acceptance, directional).

Runs the same coverage+offload heuristic the origin /tmp/mfmec_*.py scripts used,
but against onpolicy.envs.mec.finite_k_env loaded from the locked v2 YAML. The
point is migration correctness at the *system* level: cost shares land in the
right regime and all three action types are load-bearing. Numbers are checked
LOOSELY (directional), not to the percent -- the heuristic is not the learned
policy and the demand heading is random per seed. Permutation invariance is the
one strict check (it is a correctness property, not a tuning outcome).

Run: /opt/anaconda3/envs/marl/bin/python -m onpolicy.envs.mec.tests.test_health_check
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from onpolicy.envs.mec.config_loader import load_scenario       # noqa: E402
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv   # noqa: E402
from onpolicy.envs.mec.metrics import demand_matching_w1         # noqa: E402

SEEDS = (20260604, 20260605, 20260606)
DT = 1.0


def _steer(cur, tgt, vmax):
    d = tgt - cur
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return np.where(n > 0, d / np.maximum(n, 1e-9) * np.minimum(vmax, n / DT), 0.0)


def _rollout(cfg, hub="track", beta="qaware", uav="move", seed=0):
    """One heuristic episode; returns accumulated cost components + raw bits."""
    env = FiniteKHAPUAVMECEnv(cfg)
    k = cfg["env"]["fleet_size_k"]
    R = float(cfg["env"]["region"]["lx_m"])
    v_u = float(cfg["env"]["uav"]["velocity_max_mps"])
    v_h = float(cfg["env"]["hap"]["velocity_max_mps"])
    q_u_max = float(cfg["env"]["uav"]["queue_max_bits"])
    sigma = float(cfg["demand"]["activity_probability"]["hotspot_sigma_m"])
    # sunflower (equal-area) fill of the hotspot disk ~1.5 sigma: spreads UAVs
    # across the hotspot extent (incl. center), a sane demand-matcher for the W1 probe.
    rr = 1.5 * sigma * np.sqrt((np.arange(k) + 0.5) / k)
    ga = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(k)
    offset = np.stack([rr * np.cos(ga), rr * np.sin(ga)], axis=1)

    obs, _ = env.reset(seed=seed)

    acc = dict(train=0.0, src=0.0, ovf=0.0, q=0.0, en=0.0, U=0.0, A=0.0, B=0.0, w1=0.0)
    H = int(cfg["base"]["episode_horizon_slots"])
    for _ in range(H):
        ux = np.asarray(obs["uavs"]["xy_m"], float)
        q = np.asarray(obs["uavs"]["queue_bits"], float)
        hx = np.asarray(obs["hap"]["xy_m"], float)
        hot = np.asarray(obs["demand"]["hotspot_center_m"], float)
        acc["w1"] += demand_matching_w1(ux, env.grid_xy, env._demand_density(hot)[0])

        if uav == "hover":
            uv = np.zeros((k, 2))
        else:
            uv = _steer(ux, np.clip(hot + offset, [0, 0], [R, R]), v_u)
        ht = ux.mean(0) if hub == "track" else np.array([R / 2, R / 2])
        hv = _steer(hx[None, :], ht[None, :], v_h)[0]
        b = np.zeros(k) if beta == "zero" else np.clip(q / (0.2 * q_u_max), 0.0, 0.95)

        obs, _, _, _, info = env.step(
            {"hap_velocity_mps": hv, "uav_velocity_mps": uv, "beta": b})
        acc["train"] += info["training_cost"]
        acc["src"] += info["src_cost_component"]
        acc["ovf"] += info["ovf_cost_component"]
        acc["q"] += info["queue_cost_component"]
        acc["en"] += info["energy_cost_component"]
        acc["U"] += info["U_src"]
        acc["A"] += float(np.sum(info["A_i"]))
        acc["B"] += float(np.sum(info["B_i"]))
    acc["w1"] /= H  # per-step mean demand-matching distance [m]
    return acc


def _avg(cfg, **kw):
    runs = [_rollout(cfg, seed=s, **kw) for s in SEEDS]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def _perm_invariance(cfg, seed, perm):
    """Strict ENV symmetry: relabel UAV state AND actions by the same sigma ->
    identical team cost. Isolates the env (not the slot-indexed heuristic)."""
    k = cfg["env"]["fleet_size_k"]
    rng = np.random.default_rng(123)
    q0 = rng.uniform(0.0, float(cfg["env"]["uav"]["queue_max_bits"]), k)  # primed queues
    hv = rng.uniform(-5, 5, 2)
    uav_v = rng.uniform(-10, 10, (k, 2))
    beta = rng.uniform(0.0, 1.0, k)

    e1 = FiniteKHAPUAVMECEnv(cfg); e1.reset(seed=seed)
    e1.state.uav_queue_bits = q0.copy()
    _, _, _, _, i1 = e1.step({"hap_velocity_mps": hv, "uav_velocity_mps": uav_v, "beta": beta})

    e2 = FiniteKHAPUAVMECEnv(cfg); e2.reset(seed=seed)
    e2.state.uav_xy_m = e2.state.uav_xy_m[perm]
    e2.state.uav_queue_bits = q0[perm]
    _, _, _, _, i2 = e2.step({"hap_velocity_mps": hv,
                              "uav_velocity_mps": uav_v[perm], "beta": beta[perm]})
    return abs(i1["training_cost"] - i2["training_cost"]) / max(abs(i1["training_cost"]), 1e-9)


def run_health_check(fleet_size_k=24, verbose=True):
    cfg = load_scenario("v2_iort_6km_mmwave", fleet_size_k=fleet_size_k)
    main = _avg(cfg, hub="track", beta="qaware", uav="move")
    hubfix = _avg(cfg, hub="center", beta="qaware", uav="move")
    beta0 = _avg(cfg, hub="track", beta="zero", uav="move")
    hover = _avg(cfg, hub="track", beta="qaware", uav="hover")

    t = main["train"]
    cov, ovf, qsh, ensh = main["src"] / t, main["ovf"] / t, main["q"] / t, main["en"] / t
    offload_related = ovf + qsh
    hub_abl = hubfix["train"] / main["train"] - 1.0
    beta_abl = beta0["train"] / main["train"] - 1.0
    hover_abl = hover["train"] / main["train"] - 1.0

    # permutation invariance (strict env property): relabel state+actions by sigma
    perm = np.random.default_rng(0).permutation(fleet_size_k)
    perm_rel = max(_perm_invariance(cfg, s, perm) for s in SEEDS)

    if verbose:
        print(f"=== ported-env health check (K={fleet_size_k}, {len(SEEDS)} seeds, heuristic) ===")
        print(f"  accept rate            : {main['A'] / (main['A'] + main['U']) * 100:5.1f}%")
        print(f"  share coverage (src)   : {cov * 100:5.1f}%   [loose target 20-35]")
        print(f"  share overflow (ovf)   : {ovf * 100:5.1f}%")
        print(f"  share queue            : {qsh * 100:5.1f}%")
        print(f"  share offload-related  : {offload_related * 100:5.1f}%   [loose target 55-72]")
        print(f"  share energy           : {ensh * 100:5.1f}%   [small]")
        print(f"  hub ablation (fix-ctr) : {hub_abl * 100:+5.1f}%   [origin ref +23; directional >0]")
        print(f"  beta ablation (b=0)    : {beta_abl * 100:+5.1f}%   [origin ref +52; load-bearing]")
        print(f"  hover gate             : {hover_abl * 100:+5.1f}%   [origin ref +38; anti-hover]")
        print(f"  permutation rel-error  : {perm_rel:.2e}   [strict env symmetry, <1e-9]")
        print(f"  W1 demand-match (move) : {main['w1']:7.0f} m   [should be << hover]")
        print(f"  W1 demand-match (hover): {hover['w1']:7.0f} m   [UAVs stuck on init grid]")
        print(f"  W1 improvement (move)  : {(1 - main['w1'] / max(hover['w1'], 1e-9)) * 100:5.1f}%")

    # directional / loose asserts (per user: roughly-matching is fine, not to the percent)
    assert 0.08 <= cov <= 0.40, f"coverage share {cov:.2f} out of regime"
    assert 0.45 <= offload_related <= 0.85, f"offload-related share {offload_related:.2f} out of regime"
    assert ensh < 0.25, f"energy share {ensh:.2f} too large"
    assert hub_abl >= 0.03, f"hub trajectory not load-bearing ({hub_abl:+.2f})"
    assert beta_abl >= 0.20, f"beta not load-bearing ({beta_abl:+.2f})"
    assert hover_abl >= 0.08, f"hover not penalized ({hover_abl:+.2f})"
    assert perm_rel < 1e-9, f"env permutation symmetry violated ({perm_rel:.2e})"
    assert main["w1"] < hover["w1"], (
        f"demand-matcher should beat uniform hover on W1 "
        f"(move {main['w1']:.0f} m vs hover {hover['w1']:.0f} m)")
    return dict(cov=cov, offload=offload_related, energy=ensh, hub=hub_abl,
               beta=beta_abl, hover=hover_abl, perm=perm_rel,
               w1_move=main["w1"], w1_hover=hover["w1"])


def test_health_check():
    run_health_check(fleet_size_k=24, verbose=False)


if __name__ == "__main__":
    run_health_check(fleet_size_k=24)
    print("\nported-env health check PASS (directional regime + load-bearing + strict permutation)")
