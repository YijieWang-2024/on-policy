"""Head-to-head of the training reward & entropy curves for three runs:
  A = buggy,  ent 0.01 nominal   (v3_explorefix_k12/run1)   -- "buggy" curve
  B = fixed,  ent 0.01           (v3_logpfix_k12/run1)       -- fix alone
  C = fixed,  ent 0.003          (v3_logpfix_ent003/run1)    -- final
Tests the user's specific claims: (1) C jumpier than A? (2) C lower at end?
(3) C smaller end-slope than A (i.e. plateaued earlier)?
"""
import sys
import numpy as np
sys.path.insert(0, "F:/置换不变性")
from tb_dump import parse_logdir

BASE = "F:/置换不变性/YijieWang-2024-on-policy/onpolicy/scripts/results/MEC/v3_iort_learnable/mappo"
RUNS = {
    "A buggy ent.01 ": f"{BASE}/v3_explorefix_k12/run1/logs",
    "B fixed ent.01 ": f"{BASE}/v3_logpfix_k12/run1/logs",
    "C fixed ent.003": f"{BASE}/v3_logpfix_ent003/run1/logs",
}

def series(logdir, tag):
    s = parse_logdir(logdir)
    p = s.get(tag, [])
    return np.array([x[0] for x in p], float), np.array([x[1] for x in p], float)

def slope_last(steps, y, window):
    m = steps >= (steps.max() - window)
    if m.sum() < 3: return float("nan"), float("nan")
    A = np.vstack([steps[m], np.ones(m.sum())]).T
    sl, ic = np.linalg.lstsq(A, y[m], rcond=None)[0]
    resid = y[m] - (sl*steps[m] + ic)
    return sl*1e5, resid.std()

print(f"{'run':16s} | {'last3rd':>8s} {'last500k':>8s} {'last300k':>8s} | "
      f"{'slpFULL':>7s} {'slp500k':>7s} {'slp300k':>7s} | {'volFULL':>7s} {'vol500k':>7s} | {'sigma_end'}")
for name, d in RUNS.items():
    st, y = series(d, "average_episode_rewards")
    _, ent = series(d, "dist_entropy")
    n = len(y); t3 = n//3
    last3 = y[2*t3:].mean()
    last500 = y[st >= st.max()-500_000].mean()
    last300 = y[st >= st.max()-300_000].mean()
    A = np.vstack([st, np.ones(n)]).T
    slF, icF = np.linalg.lstsq(A, y, rcond=None)[0]
    volF = (y-(slF*st+icF)).std()
    sl5, vol5 = slope_last(st, y, 500_000)
    sl3, _    = slope_last(st, y, 300_000)
    # final entropy -> if MEC actor, dist_entropy is reported; sigma via exp not here, just show entropy
    ent_end = ent[-3:].mean() if len(ent) else float("nan")
    print(f"{name:16s} | {last3:8.1f} {last500:8.1f} {last300:8.1f} | "
          f"{slF*1e5:+7.2f} {sl5:+7.2f} {sl3:+7.2f} | {volF:7.2f} {vol5:7.2f} | ent_end={ent_end:.3f}")

print("\n--- downsampled reward trajectory (12 pts, step k / reward) ---")
for name, d in RUNS.items():
    st, y = series(d, "average_episode_rewards")
    idx = np.linspace(0, len(y)-1, 12).astype(int)
    traj = " ".join(f"{y[i]:.0f}" for i in idx)
    print(f"{name}: {traj}")
print("steps(k)  : " + " ".join(f"{series(RUNS['C fixed ent.003'],'average_episode_rewards')[0][i]/1000:.0f}"
      for i in np.linspace(0, len(series(RUNS['C fixed ent.003'],'average_episode_rewards')[0])-1,12).astype(int)))
