"""Shared-policy runner for the MEC environment.

Reuses the MPE shared runner wholesale (Gymnasium termination semantics,
truncation bootstrap, GAE, logging) and only adds D4: a Box action passthrough.
MEC actions are continuous (`[-1,1]^3` per agent); the env does the projection,
so the runner forwards raw actions unchanged instead of one-hot expanding.
"""

from __future__ import annotations

import numpy as np

from onpolicy.runner.shared.mpe_runner import MPERunner


class MECRunner(MPERunner):
    def _actions_to_env(self, actions, envs):
        action_space = envs.action_space[0]
        if action_space.__class__.__name__ == "Box":
            return actions  # [threads, agents, act_dim] -> env projects internally
        return super()._actions_to_env(actions, envs)

    def log_env(self, env_infos, total_num_steps):
        """Also surface team MEC metrics (carried on the major agent's info slot)."""
        keys = ("training_cost", "src_cost", "ovf_cost", "queue_cost",
                "energy_cost", "accepted", "offloaded", "overflow", "U_src")
        for key in keys:
            vals = []
            for info in getattr(self, "_last_infos", []):
                major = info.get("agent_infos", [{}])[0]
                if key in major and major[key] is not None:
                    vals.append(major[key])
            if vals:
                env_infos[f"mec/{key}"] = vals
        super().log_env(env_infos, total_num_steps)

    def insert(self, data):
        # stash the latest infos so log_env can read team metrics
        self._last_infos = data[4]
        super().insert(data)
