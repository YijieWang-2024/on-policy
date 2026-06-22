"""Phase-1 inverse design (NO communication tuning yet; access set generous).
Find task/compute params so the hotspot OPTIMALLY needs ~5 UAVs: the 5th UAV on
the hotspot still pays off, but the 6th is better spent on the background.

Mechanism (per the soft service-split): a dense hotspot grid splits its load
across all covering UAVs, so each added UAV on the hotspot takes a SHARE -> its
marginal accepted/processed decreases with crowding; a UAV alone on background
owns its cell. The crossover n* is where hotspot-marginal == background-marginal.

We measure, on a STATIC central hotspot + uniform background, with access made
generous (wide per-UAV bandwidth so processing/sharing is the binding limit):
  cost(n) = team cost with n UAVs packed on the hotspot and (K-n) on a background
            grid; hub at hotspot centre; beta queue-aware; deploy-and-hold H slots.
n* = argmin_n cost(n). We want n* ~ 5 and cap > offered (bg also serviceable, ovf~0).
"""
import sys, math
import numpy as np
sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
from onpolicy.envs.mec.config_loader import load_scenario, derive_constants
from onpolicy.envs.mec.finite_k_env import FiniteKHAPUAVMECEnv

H, R = 50, 6000.0


def cfg_for(K, sigma, peak, bg, comp, access_mult):
    c = load_scenario("v3_iort_learnable", fleet_size_k=K)
    c["numerics"]["spatial_integral"]["grid_shape"] = [61, 61]
    pr = c["demand"]["process"]; pr["speed_mps"] = 0.0; pr["noise_std_m"] = 0.0
    pr["initial_center_frac"] = [0.5, 0.5]
    ap = c["demand"]["activity_probability"]
    ap["hotspot_sigma_m"] = float(sigma); ap["hotspot_peak_increment"] = float(peak)
    ap["base_probability"] = float(bg); ap["clip_probability"] = [0.0, float(peak + bg + 1e-6)]
    c["env"]["uav"]["cpu_frequency_hz"] *= comp; c["env"]["hap"]["cpu_frequency_hz"] *= comp
    # make ACCESS generous so phase-1 bottleneck is processing, not bandwidth:
    c["communication"]["access"]["total_bandwidth_hz"] *= access_mult
    c["derived"] = derive_constants(c)
    return c


def bg_grid(n):
    if n <= 0: return np.zeros((0, 2))
    g = int(math.ceil(math.sqrt(n))); pts = []
    for i in range(g):
        for j in range(g):
            if len(pts) < n:
                pts.append([R * (0.12 + 0.76 * i / max(g - 1, 1)), R * (0.12 + 0.76 * j / max(g - 1, 1))])
    return np.array(pts[:n])


def hot_pack(n, sigma):
    if n <= 0: return np.zeros((0, 2))
    rr = sigma * np.sqrt((np.arange(n) + 0.5) / n)
    ga = math.pi * (3 - math.sqrt(5)) * np.arange(n)
    return np.clip(np.array([R/2, R/2]) + np.stack([rr*np.cos(ga), rr*np.sin(ga)], 1), 0, R)


def offered(c):
    env = FiniteKHAPUAVMECEnv(c); _, info = env.reset(seed=7)
    _, fresh = env._demand_density(np.asarray(info["demand_center_m"], float))
    return float(np.sum(fresh)) * env.cell_area


def score(c, uav_xy):
    env = FiniteKHAPUAVMECEnv(c); _, info = env.reset(seed=7)
    env.state.uav_xy_m = uav_xy.copy(); env.state.hap_xy_m = np.asarray(info["demand_center_m"], float)
    qmax = float(c["env"]["uav"]["queue_max_bits"]); cost = ovf = acc = 0.0
    for _ in range(H):
        beta = np.clip(env.state.uav_queue_bits / (0.2 * qmax), 0, 0.95)
        _, r, _, _, i = env.step({"hap_velocity_mps": np.zeros(2),
            "uav_velocity_mps": np.zeros((len(uav_xy), 2)), "beta": beta})
        cost += i["training_cost"]; ovf += float(np.sum(i["D_i_U"])) + i["D_H"]; acc += float(np.sum(i["A_i"]))
    return cost / H, ovf / H / 1e6, acc / H / 1e6


# C_U at comp: base C_U=0.5 Mbit/slot * comp.  cap = (K*C_U + C_H)
def run(K, sigma, peak, bg, comp, access_mult, label):
    c = cfg_for(K, sigma, peak, bg, comp, access_mult)
    cu = c["derived"]["uav_compute_capacity_bits"]/1e6; ch = c["derived"]["hap_compute_capacity_bits"]/1e6
    off = offered(c)/1e6; cap = K*cu + ch
    costs = []
    for n in range(1, min(K, 9)+1):
        xy = np.vstack([hot_pack(n, sigma*0.8), bg_grid(K-n)])
        cst, ovf, acc = score(c, xy); costs.append((n, cst, ovf, acc))
    nstar = min(costs, key=lambda t: t[1])[0]
    print(f"\n[{label}] K={K} sig={sigma} peak={peak} bg={bg} C_U={cu:.2f} C_H={ch:.1f} cap={cap:.1f} offered={off:.1f} (cap/off={cap/off:.2f})  => n*={nstar}")
    print("   n :  " + " ".join(f"{n:5d}" for n,_,_,_ in costs))
    print("  cost: " + " ".join(f"{cst:5.2f}" for _,cst,_,_ in costs))
    print("  ovf : " + " ".join(f"{ovf:5.1f}" for _,_,ovf,_ in costs))
    return nstar


# CORRECT mechanism: hotspot must LOCALLY overload a single UAV (1 UAV on hotspot
# overflows) so ~5 are needed to share it; background sparse enough that 1 UAV
# suffices there; global cap >= total offered. Key diagnostic: ovf at n=1 must be
# >0 (single UAV can't drain the hotspot), ovf at n=5 ~ 0 (5 share it fine).
# Lever: hotspot peak high (dense local task) + modest C_U; background low.
run(16, 700, 0.006, 1e-5, 0.5, 4.0, "A tightHot weakCU")
run(16, 600, 0.008, 1e-5, 0.5, 4.0, "B")
run(16, 600, 0.008, 1e-5, 0.7, 4.0, "C strongerCU")
run(16, 500, 0.010, 1e-5, 0.7, 4.0, "D tighter+hotter")
run(16, 600, 0.006, 1e-5, 0.4, 4.0, "E weakestCU")
