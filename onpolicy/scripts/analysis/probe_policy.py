"""Probe a trained MECActor: exploration scale + does the mean action respond to
the demand direction (minor) / swarm-centroid direction (major)?

If the mean velocity ignores the demand offset, the policy never learned the
track-the-demand mapping -> confirms a learning-signal problem, not a code bug.
"""
import json, sys
from types import SimpleNamespace
import numpy as np
import torch

RUN = sys.argv[1]
cfg = json.load(open(RUN + "/models/config.json"))["all_args"]
args = SimpleNamespace(**cfg)

sys.path.insert(0, "F:/置换不变性/YijieWang-2024-on-policy")
from onpolicy.algorithms.mec.mec_policy import MECActor

try:
    from gym import spaces
except Exception:
    from gymnasium import spaces

obs_space = spaces.Box(-np.inf, np.inf, (14,), np.float32)
act_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
actor = MECActor(args, obs_space, act_space, torch.device("cpu"))
actor.load_state_dict(torch.load(RUN + "/models/actor.pt", map_location="cpu"))
actor.eval()

print("=== exploration scale (std = exp(logstd)) ===")
print("  major_logstd:", actor.major_logstd.data.numpy(), "-> sigma", np.exp(actor.major_logstd.data.numpy()))
print("  minor_logstd:", actor.minor_logstd.data.numpy(), "-> sigma", np.exp(actor.minor_logstd.data.numpy()))

R = 6000.0


def minor_obs(own_xy, demand_xy, demand_vel=(0, 0), descriptor_xy=(3000, 3000)):
    o = np.zeros(14, np.float32)
    o[0] = 0.0                                   # role = minor
    o[1:3] = np.array(own_xy) / R                # own xy
    o[3] = 0.3                                   # own queue (moderate)
    o[4:6] = np.array([3000, 3000]) / R          # hub xy (center)
    o[6] = 0.3                                    # hub queue
    o[7:9] = np.array(demand_xy) / R             # demand center
    o[9:11] = np.array(demand_vel) / 40.0        # demand velocity (norm by v_u_max)
    o[11:13] = np.array(descriptor_xy) / R       # swarm-centroid xy
    o[13] = 0.3                                   # swarm-centroid queue
    return o


def major_obs(hub_xy, demand_xy, centroid_xy):
    o = np.zeros(14, np.float32)
    o[0] = 1.0                                   # role = major
    o[1:3] = np.array(hub_xy) / R
    o[3] = 0.3
    o[4:6] = np.array(hub_xy) / R                # major own block == hub
    o[6] = 0.3
    o[7:9] = np.array(demand_xy) / R
    o[9:11] = 0.0
    o[11:13] = np.array(centroid_xy) / R
    o[13] = 0.3
    return o


rnn = np.zeros((1, args.recurrent_N, args.hidden_size), np.float32)
masks = np.ones((1, 1), np.float32)


def mean_action(o):
    with torch.no_grad():
        a, _, _ = actor(o[None, :], rnn, masks, deterministic=True)
    return a.numpy()[0]


print("\n=== minor: does mean velocity point TOWARD the demand? (own at center 3000,3000) ===")
print("  demand dir       mean[vx, vy, beta]   (want vx,vy to point toward demand)")
for name, dxy in [("EAST (+x)", (5000, 3000)), ("WEST (-x)", (1000, 3000)),
                  ("NORTH(+y)", (3000, 5000)), ("SOUTH(-y)", (3000, 1000)),
                  ("NE       ", (5000, 5000))]:
    a = mean_action(minor_obs((3000, 3000), dxy))
    print(f"  {name}   [{a[0]:+.3f}, {a[1]:+.3f}, beta={a[2]:.3f}]")

print("\n=== major: does mean velocity point TOWARD the swarm centroid? (hub at center) ===")
for name, cxy in [("centroid EAST", (5000, 3000)), ("centroid WEST", (1000, 3000)),
                  ("centroid NORTH", (3000, 5000)), ("centroid SOUTH", (3000, 1000))]:
    a = mean_action(major_obs((3000, 3000), (3000, 3000), cxy))
    print(f"  {name:16s} [{a[0]:+.3f}, {a[1]:+.3f}]")

print("\n=== sensitivity magnitude (max |mean velocity| across the probes above) ===")
mags = [np.linalg.norm(mean_action(minor_obs((3000, 3000), d))[:2])
        for d in [(5000, 3000), (1000, 3000), (3000, 5000), (3000, 1000)]]
print(f"  minor |v| range: {min(mags):.3f} .. {max(mags):.3f}  (0 => ignores demand)")
