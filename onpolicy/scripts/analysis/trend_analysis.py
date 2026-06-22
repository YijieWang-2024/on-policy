"""Rigorous trend analysis of the full training run (not just endpoints).

For each key metric: print N, segment means (first/mid/last third), OLS slope
per 100k steps with a noise-relative t-ish ratio, and a smoothed trajectory.
Answers: is there ANY consistent learning trend, or flat noise?
"""
import sys
import numpy as np

sys.path.insert(0, "F:/置换不变性")
from tb_dump import parse_logdir

LOG = sys.argv[1]
s = parse_logdir(LOG)


def moving_avg(y, w):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def analyze(tag, scale=1.0, unit=""):
    pairs = s.get(tag, [])
    if not pairs:
        print(f"{tag}: (none)"); return
    steps = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float) * scale
    n = len(y)
    t3 = n // 3
    first, mid, last = y[:t3], y[t3:2*t3], y[2*t3:]
    # OLS slope per 100k steps
    A = np.vstack([steps, np.ones_like(steps)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (slope * steps + intercept)
    resid_std = resid.std()
    slope_100k = slope * 1e5
    # crude signal/noise: |total trend over run| vs residual noise
    total_trend = slope * (steps[-1] - steps[0])
    snr = abs(total_trend) / (resid_std + 1e-12)
    sm = moving_avg(y, 9)
    sm_idx = np.linspace(0, len(sm)-1, 8).astype(int)
    sm_str = " ".join(f"{sm[i]:.3g}" for i in sm_idx)
    print(f"\n{tag}  (n={n}, {steps[0]/1000:.0f}k..{steps[-1]/1000:.0f}k)  {unit}")
    print(f"  thirds: first={first.mean():.3g}±{first.std():.2g}  "
          f"mid={mid.mean():.3g}±{mid.std():.2g}  last={last.mean():.3g}±{last.std():.2g}")
    print(f"  OLS slope/100k = {slope_100k:+.3g}   total trend over run = {total_trend:+.3g}"
          f"   resid noise = {resid_std:.3g}   trend/noise = {snr:.2f}")
    print(f"  smoothed(w9): {sm_str}")


for tag, sc, u in [
    ("average_episode_rewards", 1.0, "(higher=better)"),
    ("mec/w1", 1.0, "m (LOWER=better, goal)"),
    ("mec/accepted", 1e-6, "Mbit/slot (higher=better)"),
    ("mec/U_src", 1e-6, "Mbit/slot (lower=better)"),
    ("mec/training_cost", 1.0, "(lower=better)"),
    ("mec/src_cost", 1.0, "share-cost"),
    ("mec/ovf_cost", 1.0, "share-cost"),
    ("dist_entropy", 1.0, "(exploration)"),
    ("explained_variance", 1.0, "(critic)"),
]:
    analyze(tag, sc, u)

print("\n\nNOTE: trend/noise >> 1 and consistent thirds => real learning; "
      "trend/noise < ~1 and overlapping thirds => flat noise.")
