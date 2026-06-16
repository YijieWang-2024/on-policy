"""Major-minor shared-parameter actor + policy for MAPPO on the MEC env.

Design (see docs/mec_env_port_spec.md section 8, D1a folded into one module):
- ONE shared trunk over the 14-d agent obs (role flag + own state + hub public
  state + demand + permutation-invariant population descriptor).
- TWO role-routed heads:
    * major head  -> v^H : 2-d diagonal Gaussian (the hub trajectory).
    * minor head  -> v_i : 2-d diagonal Gaussian, and beta_i : Beta(a,b) on [0,1].
  All K UAVs are routed through the SAME minor head => parameter sharing across
  the fleet + permutation equivariance; the hub is the single heterogeneous agent.
- The stored action is a unified 3-vector [vx, vy, beta]; the major's beta slot is
  a dummy 0 and is ignored in its log-prob/entropy (only the 2 velocity dims count).

This matches the R_Actor / R_MAPPOPolicy interface exactly, so it drops into the
stock R_MAPPO trainer with no change to the PPO loop. The critic is reused
(R_Critic) on the centralized share_obs. Velocities are unsquashed (the env
projects to the velocity disk, the library's Box convention); beta uses a Beta
distribution (spec R6) so it is naturally bounded on [0,1] with no clipping bias.

The Gaussian velocity heads' initial log-std is set by ``args.mec_logstd_init``
(default -1.9 => sigma~0.15). A small initial sigma keeps exploration local so the
learned mean dominates; a large sigma (the old fixed 0 => sigma 1) made samples
saturate the velocity disk and the swarm careen at max speed (see spec section 10).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import check, init
from onpolicy.utils.util import get_shape_from_obs_space, update_linear_schedule

_EPS = 1e-6


class MECActor(nn.Module):
    """Role-routed major/minor actor with shared minor parameters."""

    VEL_DIM = 2          # velocity action dims (vx, vy)
    ACT_DIM = 3          # stored action = [vx, vy, beta]

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)
        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size,
                                self._recurrent_N, self._use_orthogonal)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=0.01)

        h = self.hidden_size
        logstd_init = float(getattr(args, "mec_logstd_init", 0.0))
        # major head: hub velocity (2-d Gaussian)
        self.major_mean = init_(nn.Linear(h, self.VEL_DIM))
        self.major_logstd = nn.Parameter(torch.full((self.VEL_DIM,), logstd_init))
        # minor head: UAV velocity (2-d Gaussian) + offload beta (Beta on [0,1])
        self.minor_mean = init_(nn.Linear(h, self.VEL_DIM))
        self.minor_logstd = nn.Parameter(torch.full((self.VEL_DIM,), logstd_init))
        self.minor_beta = init_(nn.Linear(h, 2))   # -> softplus + 1 => (alpha, beta) >= 1

        self.to(device)

    # ---------------------------------------------------------------- helpers
    def _features(self, obs, rnn_states, masks):
        feats = self.base(obs)
        if self._use_recurrent_policy or self._use_naive_recurrent_policy:
            feats, rnn_states = self.rnn(feats, rnn_states, masks)
        return feats, rnn_states

    def _dists(self, feats):
        major_std = torch.exp(self.major_logstd).expand_as(self.major_mean(feats))
        major = Normal(self.major_mean(feats), major_std)
        minor_std = torch.exp(self.minor_logstd).expand_as(self.minor_mean(feats))
        minor_v = Normal(self.minor_mean(feats), minor_std)
        ab = torch.nn.functional.softplus(self.minor_beta(feats)) + 1.0
        minor_b = Beta(ab[:, 0:1], ab[:, 1:2])
        return major, minor_v, minor_b

    # ---------------------------------------------------------------- forward
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        is_major = (obs[:, 0:1] > 0.5).float()                # role flag in obs[:,0]

        feats, rnn_states = self._features(obs, rnn_states, masks)
        major, minor_v, minor_b = self._dists(feats)

        if deterministic:
            v_major = major.mean
            v_minor = minor_v.mean
            b_minor = minor_b.mean
        else:
            v_major = major.rsample()
            v_minor = minor_v.rsample()
            b_minor = minor_b.rsample()

        v = is_major * v_major + (1.0 - is_major) * v_minor   # [batch, 2]
        beta = (1.0 - is_major) * b_minor                     # major's beta slot = 0 (dummy)
        actions = torch.cat([v, beta], dim=-1)                # [batch, 3]

        lp_major = major.log_prob(v_major).sum(-1, keepdim=True)
        lp_minor = (minor_v.log_prob(v_minor).sum(-1, keepdim=True)
                    + minor_b.log_prob(b_minor.clamp(_EPS, 1.0 - _EPS)))
        action_log_probs = is_major * lp_major + (1.0 - is_major) * lp_minor
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        is_major = (obs[:, 0:1] > 0.5).float()

        feats, _ = self._features(obs, rnn_states, masks)
        major, minor_v, minor_b = self._dists(feats)

        vel = action[:, :self.VEL_DIM]
        b = action[:, self.VEL_DIM:self.VEL_DIM + 1].clamp(_EPS, 1.0 - _EPS)

        lp_major = major.log_prob(vel).sum(-1, keepdim=True)
        ent_major = major.entropy().sum(-1, keepdim=True)
        lp_minor = (minor_v.log_prob(vel).sum(-1, keepdim=True) + minor_b.log_prob(b))
        ent_minor = minor_v.entropy().sum(-1, keepdim=True) + minor_b.entropy()

        action_log_probs = is_major * lp_major + (1.0 - is_major) * lp_minor
        entropy = is_major * ent_major + (1.0 - is_major) * ent_minor
        if self._use_policy_active_masks and active_masks is not None:
            dist_entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy


class MECPolicy(R_MAPPOPolicy):
    """R_MAPPOPolicy with the major-minor MECActor; critic reused unchanged."""

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = MECActor(args, obs_space, act_space, device)
        self.critic = R_Critic(args, cent_obs_space, device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
