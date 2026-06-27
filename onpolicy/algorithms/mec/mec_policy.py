"""Information-matched major-minor MEC population policies."""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal

try:
    from gym import spaces
except Exception:  # pragma: no cover
    from gymnasium import spaces

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import check, init
from onpolicy.algorithms.mec.set_networks import (
    PopulationReconstructionDecoder,
    FusionMLP,
    build_population_encoder,
    chamfer_set_loss,
    population_encoder_type,
    population_representation_dim,
)
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    LEGACY_AGENT_OBS_DIM,
    OWN_SLICE,
    PHYSICAL_PUBLIC_SLICE,
    PHYSICAL_PUBLIC_STATE_DIM,
    PUBLIC_SLICE,
    PUBLIC_STATE_DIM,
    ROLE_INDEX,
    UAV_STATE_DIM,
    team_state_dim,
)
from onpolicy.utils.util import get_shape_from_obs_space, update_linear_schedule

_EPS = 1e-6
_ROLE = ROLE_INDEX
_OWN = OWN_SLICE
_PUBLIC = PUBLIC_SLICE


class MECActor(nn.Module):
    """Legacy mean-descriptor actor with a shared role-conditioned trunk."""

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
        self._use_rolewise_loss = bool(
            getattr(args, "mec_rolewise_loss", False)
        )
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
        if self._use_rolewise_loss:
            weights = (
                active_masks
                if self._use_policy_active_masks and active_masks is not None
                else torch.ones_like(entropy)
            )
            major_weights = weights * is_major
            minor_weights = weights * (1.0 - is_major)
            role_entropies = []
            if major_weights.sum() > 0:
                role_entropies.append(
                    (entropy * major_weights).sum() / major_weights.sum()
                )
            if minor_weights.sum() > 0:
                role_entropies.append(
                    (entropy * minor_weights).sum() / minor_weights.sum()
                )
            dist_entropy = torch.stack(role_entropies).mean()
        elif self._use_policy_active_masks and active_masks is not None:
            dist_entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy


def _canonical_to_legacy_rows(obs, num_agents, tpdv):
    """Append the historical UAV mean descriptor to canonical local rows."""
    obs = check(obs).to(**tpdv)
    if obs.shape[-1] == LEGACY_AGENT_OBS_DIM:
        return obs
    if obs.shape[-1] != AGENT_OBS_DIM:
        raise ValueError(
            f"legacy adapter expected obs dim {AGENT_OBS_DIM} or "
            f"{LEGACY_AGENT_OBS_DIM}, got {obs.shape[-1]}"
        )
    if obs.shape[0] % num_agents != 0:
        raise ValueError(
            "legacy mean adapter requires complete K+1 agent groups"
        )
    team = obs.reshape(-1, num_agents, AGENT_OBS_DIM)
    mean = team[:, 1:, _OWN].mean(dim=1, keepdim=True)
    mean = mean.expand(-1, num_agents, -1)
    historical_rows = torch.cat(
        [
            team[:, :, :_OWN.stop],
            team[:, :, PHYSICAL_PUBLIC_SLICE],
            mean,
        ],
        dim=-1,
    )
    return historical_rows.reshape(
        -1, LEGACY_AGENT_OBS_DIM
    )


class MECLegacyMeanActor(MECActor):
    """Load historical 14-D actor weights while consuming canonical rows."""

    def __init__(
        self, args, obs_space, action_space, num_agents, device
    ):
        legacy_obs_space = spaces.Box(
            -float("inf"),
            float("inf"),
            (LEGACY_AGENT_OBS_DIM,),
            dtype=float,
        )
        self.num_agents = int(num_agents)
        super().__init__(
            args, legacy_obs_space, action_space, device
        )

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        return super().forward(
            _canonical_to_legacy_rows(
                obs, self.num_agents, self.tpdv
            ),
            rnn_states,
            masks,
            available_actions,
            deterministic,
        )

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        return super().evaluate_actions(
            _canonical_to_legacy_rows(
                obs, self.num_agents, self.tpdv
            ),
            rnn_states,
            action,
            masks,
            available_actions,
            active_masks,
        )


class MECLegacyMeanCritic(R_Critic):
    """Load historical ordered-flat critic weights from canonical state."""

    def __init__(self, args, num_agents, device):
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.canonical_state_dim = team_state_dim(self.num_agents)
        self.legacy_state_dim = (
            LEGACY_AGENT_OBS_DIM * self.num_agents
        )
        legacy_space = spaces.Box(
            -float("inf"),
            float("inf"),
            (self.legacy_state_dim,),
            dtype=float,
        )
        super().__init__(args, legacy_space, device)

    def _adapt_central_obs(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        if cent_obs.shape[-1] == self.legacy_state_dim:
            return cent_obs
        if cent_obs.shape[-1] != self.canonical_state_dim:
            raise ValueError(
                "legacy critic expected centralized state dim "
                f"{self.canonical_state_dim} or "
                f"{self.legacy_state_dim}, got "
                f"{cent_obs.shape[-1]}"
            )
        if cent_obs.shape[0] % self.num_agents != 0:
            raise ValueError(
                "legacy critic requires complete K+1 agent groups"
            )

        grouped = cent_obs.reshape(
            -1, self.num_agents, self.canonical_state_dim
        )
        state = grouped[:, 0]
        public = state[:, :PHYSICAL_PUBLIC_STATE_DIM]
        uavs = state[:, PUBLIC_STATE_DIM:].reshape(
            -1, self.num_uavs, UAV_STATE_DIM
        )
        mean = uavs.mean(dim=1)
        batch = state.shape[0]

        major = torch.cat(
            [
                torch.ones(
                    batch, 1, dtype=state.dtype, device=state.device
                ),
                public[:, :UAV_STATE_DIM],
                public,
                mean,
            ],
            dim=-1,
        ).unsqueeze(1)
        minor = torch.cat(
            [
                torch.zeros(
                    batch,
                    self.num_uavs,
                    1,
                    dtype=state.dtype,
                    device=state.device,
                ),
                uavs,
                public.unsqueeze(1).expand(-1, self.num_uavs, -1),
                mean.unsqueeze(1).expand(-1, self.num_uavs, -1),
            ],
            dim=-1,
        )
        legacy = torch.cat([major, minor], dim=1).reshape(
            batch, self.legacy_state_dim
        )
        return legacy.unsqueeze(1).expand(
            -1, self.num_agents, -1
        ).reshape(-1, self.legacy_state_dim)

    def forward(self, cent_obs, rnn_states, masks):
        return super().forward(
            self._adapt_central_obs(cent_obs), rnn_states, masks
        )


class _GroupedRoleActor(nn.Module):
    """Common action distributions for grouped MEC population actors."""

    VEL_DIM = 2
    ACT_DIM = 3

    def __init__(
        self, args, obs_space, action_space, num_agents, device
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.obs_dim = int(get_shape_from_obs_space(obs_space)[0])
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_rolewise_loss = bool(
            getattr(args, "mec_rolewise_loss", False)
        )
        self.tpdv = dict(dtype=torch.float32, device=device)
        if (
            args.use_recurrent_policy
            or args.use_naive_recurrent_policy
        ):
            raise NotImplementedError(
                "MEC population policies currently support feed-forward "
                "MAPPO only"
            )

        init_method = [
            nn.init.xavier_uniform_, nn.init.orthogonal_
        ][args.use_orthogonal]

        def init_(module):
            return init(
                module,
                init_method,
                lambda x: nn.init.constant_(x, 0),
                gain=0.01,
            )

        logstd_init = float(
            getattr(args, "mec_logstd_init", 0.0)
        )
        self.major_mean = init_(
            nn.Linear(self.hidden_size, self.VEL_DIM)
        )
        self.major_logstd = nn.Parameter(
            torch.full((self.VEL_DIM,), logstd_init)
        )
        self.minor_mean = init_(
            nn.Linear(self.hidden_size, self.VEL_DIM)
        )
        self.minor_logstd = nn.Parameter(
            torch.full((self.VEL_DIM,), logstd_init)
        )
        self.minor_beta = init_(nn.Linear(self.hidden_size, 2))
        self.to(device)

    def _reshape_team(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected obs dim {self.obs_dim}, got {obs.shape[-1]}"
            )
        if obs.shape[0] % self.num_agents != 0:
            raise ValueError(
                "grouped MEC policy requires complete K+1 agent groups"
            )
        return obs.reshape(-1, self.num_agents, self.obs_dim)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _dists(self, features):
        major_mean = self.major_mean(features)
        major_std = torch.exp(self.major_logstd).expand_as(
            major_mean
        )
        minor_mean = self.minor_mean(features)
        minor_std = torch.exp(self.minor_logstd).expand_as(
            minor_mean
        )
        ab = torch.nn.functional.softplus(
            self.minor_beta(features)
        ) + 1.0
        return (
            Normal(major_mean, major_std),
            Normal(minor_mean, minor_std),
            Beta(ab[:, 0:1], ab[:, 1:2]),
        )

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        is_major = (obs[:, _ROLE:_ROLE + 1] > 0.5).float()
        features = self._features(obs)
        major, minor_v, minor_b = self._dists(features)

        if deterministic:
            v_major = major.mean
            v_minor = minor_v.mean
            b_minor = minor_b.mean
        else:
            v_major = major.rsample()
            v_minor = minor_v.rsample()
            b_minor = minor_b.rsample()

        velocity = (
            is_major * v_major + (1.0 - is_major) * v_minor
        )
        beta = (1.0 - is_major) * b_minor
        actions = torch.cat([velocity, beta], dim=-1)
        lp_major = major.log_prob(v_major).sum(-1, keepdim=True)
        lp_minor = (
            minor_v.log_prob(v_minor).sum(-1, keepdim=True)
            + minor_b.log_prob(
                b_minor.clamp(_EPS, 1.0 - _EPS)
            )
        )
        log_probs = (
            is_major * lp_major + (1.0 - is_major) * lp_minor
        )
        return actions, log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        is_major = (obs[:, _ROLE:_ROLE + 1] > 0.5).float()
        features = self._features(obs)
        major, minor_v, minor_b = self._dists(features)

        velocity = action[:, :self.VEL_DIM]
        beta = action[
            :, self.VEL_DIM:self.VEL_DIM + 1
        ].clamp(_EPS, 1.0 - _EPS)
        lp_major = major.log_prob(velocity).sum(-1, keepdim=True)
        ent_major = major.entropy().sum(-1, keepdim=True)
        lp_minor = (
            minor_v.log_prob(velocity).sum(-1, keepdim=True)
            + minor_b.log_prob(beta)
        )
        ent_minor = (
            minor_v.entropy().sum(-1, keepdim=True)
            + minor_b.entropy()
        )
        log_probs = (
            is_major * lp_major + (1.0 - is_major) * lp_minor
        )
        entropy = (
            is_major * ent_major + (1.0 - is_major) * ent_minor
        )
        return log_probs, self._reduce_entropy(
            entropy, is_major, active_masks
        )

    def _reduce_entropy(self, entropy, is_major, active_masks):
        weights = (
            active_masks
            if self._use_policy_active_masks
            and active_masks is not None
            else torch.ones_like(entropy)
        )
        if self._use_rolewise_loss:
            major_weights = weights * is_major
            minor_weights = weights * (1.0 - is_major)
            major_entropy = (
                entropy * major_weights
            ).sum() / major_weights.sum().clamp_min(1.0)
            minor_entropy = (
                entropy * minor_weights
            ).sum() / minor_weights.sum().clamp_min(1.0)
            return 0.5 * (major_entropy + minor_entropy)
        if (
            self._use_policy_active_masks
            and active_masks is not None
        ):
            return (entropy * weights).sum() / weights.sum()
        return entropy.mean()


class _PopulationActor(_GroupedRoleActor):
    """Role-specific actor whose only varying part is population encoding."""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        num_agents,
        representation_dim,
        device,
    ):
        super().__init__(
            args, obs_space, action_space, num_agents, device
        )
        self.representation_dim = int(representation_dim)
        use_relu = bool(args.use_ReLU)
        self.major_fusion = FusionMLP(
            PUBLIC_STATE_DIM + self.representation_dim,
            self.hidden_size,
            use_relu,
            layer_N=args.layer_N,
            use_orthogonal=args.use_orthogonal,
            use_feature_normalization=args.use_feature_normalization,
        )
        self.minor_fusion = FusionMLP(
            UAV_STATE_DIM
            + PUBLIC_STATE_DIM
            + self.representation_dim,
            self.hidden_size,
            use_relu,
            layer_N=args.layer_N,
            use_orthogonal=args.use_orthogonal,
            use_feature_normalization=args.use_feature_normalization,
        )
        self.to(device)

    def _representation(self, uavs):
        raise NotImplementedError

    def population_representation(self, obs):
        obs = check(obs).to(**self.tpdv)
        team = self._reshape_team(obs)
        return self._representation(team[:, 1:, _OWN])

    def _features(self, obs):
        team = self._reshape_team(obs)
        public = team[:, 0, _PUBLIC]
        uavs = team[:, 1:, _OWN]
        representation = self._representation(uavs)
        major = self.major_fusion(
            torch.cat([public, representation], dim=-1)
        )
        public_uav = public.unsqueeze(1).expand(
            -1, self.num_uavs, -1
        )
        representation_uav = representation.unsqueeze(1).expand(
            -1, self.num_uavs, -1
        )
        minor = self.minor_fusion(
            torch.cat(
                [uavs, public_uav, representation_uav], dim=-1
            )
        )
        return torch.cat(
            [major.unsqueeze(1), minor], dim=1
        ).reshape(-1, self.hidden_size)


class MECMeanActor(_PopulationActor):
    """Information-matched mean-descriptor baseline."""

    def __init__(
        self, args, obs_space, action_space, num_agents, device
    ):
        super().__init__(
            args,
            obs_space,
            action_space,
            num_agents,
            UAV_STATE_DIM,
            device,
        )

    def _representation(self, uavs):
        return uavs.mean(dim=1)


class MECFlatActor(_PopulationActor):
    """Ordered full-state baseline with the same role-specific interface."""

    def __init__(
        self, args, obs_space, action_space, num_agents, device
    ):
        super().__init__(
            args,
            obs_space,
            action_space,
            num_agents,
            UAV_STATE_DIM * (int(num_agents) - 1),
            device,
        )

    def _representation(self, uavs):
        return uavs.reshape(uavs.shape[0], -1)


class MECSetActor(_PopulationActor):
    """Invariant population encoder with the common actor readout."""

    def __init__(
        self, args, obs_space, action_space, num_agents, device
    ):
        model_dim = int(getattr(args, "mec_set_dim", 64))
        num_heads = int(getattr(args, "mec_set_heads", 4))
        num_seeds = int(
            getattr(args, "mec_set_num_seeds", 4)
        )
        if model_dim <= 0 or num_heads <= 0 or num_seeds <= 0:
            raise ValueError(
                "MEC set dimensions, heads, and seeds must be positive"
            )
        if model_dim % num_heads != 0:
            raise ValueError(
                "mec_set_dim must be divisible by mec_set_heads"
            )
        representation_dim = population_representation_dim(args)
        self.set_actor_context = str(
            getattr(args, "mec_set_actor_context", "pooled")
        ).lower()
        if self.set_actor_context not in {"pooled", "relational"}:
            raise ValueError(
                "mec_set_actor_context must be one of: "
                "pooled, relational"
            )
        if (
            population_encoder_type(args) == "flat_mlp"
            and self.set_actor_context == "relational"
        ):
            raise ValueError(
                "mec_set_encoder_type=flat_mlp supports only "
                "mec_set_actor_context=pooled"
            )
        self.set_actor_encoder_mode = str(
            getattr(args, "mec_set_actor_encoder", "shared")
        ).lower()
        if self.set_actor_encoder_mode not in {"shared", "separate"}:
            raise ValueError(
                "mec_set_actor_encoder must be one of: shared, separate"
            )
        super().__init__(
            args,
            obs_space,
            action_space,
            num_agents,
            representation_dim,
            device,
        )
        if self.set_actor_encoder_mode == "shared":
            self.population_encoder = build_population_encoder(
                args,
                atom_dim=UAV_STATE_DIM,
                use_relu=bool(args.use_ReLU),
                num_atoms=self.num_uavs,
            )
        else:
            self.major_population_encoder = build_population_encoder(
                args,
                atom_dim=UAV_STATE_DIM,
                use_relu=bool(args.use_ReLU),
                num_atoms=self.num_uavs,
            )
            self.minor_population_encoder = build_population_encoder(
                args,
                atom_dim=UAV_STATE_DIM,
                use_relu=bool(args.use_ReLU),
                num_atoms=self.num_uavs,
            )
        self.use_reconstruction_aux = bool(
            getattr(args, "mec_set_reconstruction_coef", 0.0) > 0.0
        )
        if (
            self.use_reconstruction_aux
            and self.set_actor_encoder_mode != "shared"
        ):
            raise ValueError(
                "mec_set_reconstruction_coef currently requires "
                "mec_set_actor_encoder=shared"
            )
        if self.use_reconstruction_aux:
            self.reconstruction_decoder = PopulationReconstructionDecoder(
                representation_dim=representation_dim,
                num_atoms=self.num_uavs,
                atom_dim=UAV_STATE_DIM,
                hidden_dim=self.hidden_size,
                use_relu=bool(args.use_ReLU),
                layer_N=args.layer_N,
                use_orthogonal=args.use_orthogonal,
            )
        if self.set_actor_context == "relational":
            self.minor_fusion = FusionMLP(
                UAV_STATE_DIM
                + PUBLIC_STATE_DIM
                + representation_dim
                + model_dim,
                self.hidden_size,
                bool(args.use_ReLU),
                layer_N=args.layer_N,
                use_orthogonal=args.use_orthogonal,
                use_feature_normalization=args.use_feature_normalization,
            )
        self.to(device)

    def population_representation(self, obs):
        obs = check(obs).to(**self.tpdv)
        team = self._reshape_team(obs)
        uavs = team[:, 1:, _OWN]
        if self.set_actor_encoder_mode == "shared":
            descriptor = self.population_encoder(uavs)
        else:
            descriptor = self.major_population_encoder(uavs)
        return descriptor.reshape(descriptor.shape[0], -1)

    def population_descriptor(self, obs):
        obs = check(obs).to(**self.tpdv)
        team = self._reshape_team(obs)
        if self.set_actor_encoder_mode == "shared":
            return self.population_encoder(team[:, 1:, _OWN])
        return self.major_population_encoder(team[:, 1:, _OWN])

    def reconstruct_population(self, uavs):
        if not self.use_reconstruction_aux:
            raise RuntimeError("Set reconstruction auxiliary is disabled")
        descriptor = self.population_encoder(uavs)
        representation = descriptor.reshape(descriptor.shape[0], -1)
        return self.reconstruction_decoder(representation)

    def reconstruction_chamfer_loss(self, uavs):
        return chamfer_set_loss(self.reconstruct_population(uavs), uavs)

    def _representation(self, uavs):
        if self.set_actor_encoder_mode != "shared":
            raise RuntimeError(
                "separate Set actor encoders use role-specific features"
            )
        descriptor = self.population_encoder(uavs)
        return descriptor.reshape(descriptor.shape[0], -1)

    def _features(self, obs):
        if self.set_actor_encoder_mode == "separate":
            return self._features_separate_encoders(obs)
        if self.set_actor_context != "relational":
            return super()._features(obs)

        team = self._reshape_team(obs)
        public = team[:, 0, _PUBLIC]
        uavs = team[:, 1:, _OWN]
        descriptor, tokens = self.population_encoder.forward_with_tokens(
            uavs
        )
        representation = descriptor.reshape(descriptor.shape[0], -1)
        major = self.major_fusion(
            torch.cat([public, representation], dim=-1)
        )
        public_uav = public.unsqueeze(1).expand(
            -1, self.num_uavs, -1
        )
        representation_uav = representation.unsqueeze(1).expand(
            -1, self.num_uavs, -1
        )
        minor = self.minor_fusion(
            torch.cat(
                [uavs, public_uav, representation_uav, tokens],
                dim=-1,
            )
        )
        return torch.cat(
            [major.unsqueeze(1), minor], dim=1
        ).reshape(-1, self.hidden_size)

    def _features_separate_encoders(self, obs):
        team = self._reshape_team(obs)
        public = team[:, 0, _PUBLIC]
        uavs = team[:, 1:, _OWN]

        major_descriptor = self.major_population_encoder(uavs)
        major_representation = major_descriptor.reshape(
            major_descriptor.shape[0], -1
        )
        major = self.major_fusion(
            torch.cat([public, major_representation], dim=-1)
        )

        if self.set_actor_context == "relational":
            minor_descriptor, tokens = (
                self.minor_population_encoder.forward_with_tokens(uavs)
            )
        else:
            minor_descriptor = self.minor_population_encoder(uavs)
            tokens = None
        minor_representation = minor_descriptor.reshape(
            minor_descriptor.shape[0], -1
        )
        public_uav = public.unsqueeze(1).expand(
            -1, self.num_uavs, -1
        )
        representation_uav = minor_representation.unsqueeze(1).expand(
            -1, self.num_uavs, -1
        )
        minor_inputs = [uavs, public_uav, representation_uav]
        if tokens is not None:
            minor_inputs.append(tokens)
        minor = self.minor_fusion(torch.cat(minor_inputs, dim=-1))
        return torch.cat(
            [major.unsqueeze(1), minor], dim=1
        ).reshape(-1, self.hidden_size)


class MECTeamCritic(nn.Module):
    """One team value using the same population representation as the actor."""

    def __init__(
        self,
        args,
        cent_obs_space,
        num_agents,
        architecture,
        population_encoder=None,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.central_state_dim = team_state_dim(self.num_agents)
        self.architecture = architecture
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        if (
            args.use_recurrent_policy
            or args.use_naive_recurrent_policy
        ):
            raise NotImplementedError(
                "MEC population critic currently supports feed-forward "
                "MAPPO only"
            )

        use_relu = bool(args.use_ReLU)
        if architecture == "set":
            self.set_critic_encoder_mode = str(
                getattr(args, "mec_set_critic_encoder", "actor_detached")
            ).lower()
            representation_dim = population_representation_dim(args)
            if self.set_critic_encoder_mode == "actor_detached":
                if population_encoder is None:
                    raise ValueError(
                        "actor_detached set critic requires the public "
                        "population encoder"
                    )
                self.__dict__["_population_encoder_ref"] = weakref.ref(
                    population_encoder
                )
            elif self.set_critic_encoder_mode == "shared_grad":
                if population_encoder is None:
                    raise ValueError(
                        "shared_grad set critic requires the public "
                        "population encoder"
                    )
                self.population_encoder = population_encoder
            elif self.set_critic_encoder_mode == "separate":
                self.population_encoder = build_population_encoder(
                    args,
                    atom_dim=UAV_STATE_DIM,
                    use_relu=use_relu,
                    num_atoms=self.num_uavs,
                )
            else:
                raise ValueError(
                    "mec_set_critic_encoder must be one of: "
                    "actor_detached, separate, shared_grad"
                )
        elif architecture == "flat":
            representation_dim = UAV_STATE_DIM * self.num_uavs
        elif architecture == "mean":
            representation_dim = UAV_STATE_DIM
        else:
            raise ValueError(
                f"unsupported team critic architecture: {architecture}"
            )
        self.readout = FusionMLP(
            PUBLIC_STATE_DIM + representation_dim,
            self.hidden_size,
            use_relu,
            layer_N=args.layer_N,
            use_orthogonal=args.use_orthogonal,
            use_feature_normalization=args.use_feature_normalization,
        )

        init_method = [
            nn.init.xavier_uniform_, nn.init.orthogonal_
        ][args.use_orthogonal]

        def init_(module):
            return init(
                module,
                init_method,
                lambda x: nn.init.constant_(x, 0),
            )

        if self._use_popart:
            self.v_out = init_(
                PopArt(self.hidden_size, 1, device=device)
            )
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def _team_from_central_obs(self, cent_obs):
        if cent_obs.shape[-1] != self.central_state_dim:
            raise ValueError(
                "expected canonical centralized state dim "
                f"{self.central_state_dim}, "
                f"got {cent_obs.shape[-1]}"
            )
        if cent_obs.shape[0] % self.num_agents != 0:
            raise ValueError(
                "team critic requires complete K+1 agent groups"
            )
        grouped = cent_obs.reshape(
            -1, self.num_agents, self.central_state_dim
        )
        state = grouped[:, 0]
        public = state[:, :PUBLIC_STATE_DIM]
        uavs = state[:, PUBLIC_STATE_DIM:].reshape(
            -1, self.num_uavs, UAV_STATE_DIM
        )
        return public, uavs

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        public, uavs = self._team_from_central_obs(cent_obs)

        if self.architecture == "set":
            if self.set_critic_encoder_mode == "actor_detached":
                encoder = self._population_encoder_ref()
                if encoder is None:
                    raise RuntimeError(
                        "public population encoder is no longer available"
                    )
                with torch.no_grad():
                    descriptor = encoder(uavs)
            else:
                descriptor = self.population_encoder(uavs)
            representation = descriptor.reshape(
                descriptor.shape[0], -1
            )
        elif self.architecture == "mean":
            representation = uavs.mean(dim=1)
        else:
            representation = uavs.reshape(uavs.shape[0], -1)

        features = self.readout(
            torch.cat([public, representation], dim=-1)
        )
        team_values = self.v_out(features)
        values = team_values.unsqueeze(1).expand(
            -1, self.num_agents, -1
        ).reshape(-1, 1)
        return values, rnn_states


class MECPolicy(R_MAPPOPolicy):
    """MEC policy factory preserving the standard MAPPO policy interface."""

    def __init__(
        self,
        args,
        obs_space,
        cent_obs_space,
        act_space,
        device=torch.device("cpu"),
        num_agents=None,
    ):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        obs_dim = int(get_shape_from_obs_space(obs_space)[0])
        cent_dim = int(
            get_shape_from_obs_space(cent_obs_space)[0]
        )
        if num_agents is not None:
            self.num_agents = int(num_agents)
        elif (
            cent_dim >= PUBLIC_STATE_DIM + UAV_STATE_DIM
            and (cent_dim - PUBLIC_STATE_DIM) % UAV_STATE_DIM == 0
        ):
            self.num_agents = (
                (cent_dim - PUBLIC_STATE_DIM) // UAV_STATE_DIM + 1
            )
        elif cent_dim % LEGACY_AGENT_OBS_DIM == 0:
            self.num_agents = cent_dim // LEGACY_AGENT_OBS_DIM
        else:
            raise ValueError(
                "cannot infer MEC fleet size from centralized "
                f"observation dim {cent_dim}; pass num_agents"
            )
        self.architecture = str(
            getattr(args, "mec_policy_arch", "mean")
        ).lower()
        self.uses_grouped_batches = True
        set_critic_encoder_mode = str(
            getattr(args, "mec_set_critic_encoder", "actor_detached")
        ).lower()
        set_actor_encoder_mode = str(
            getattr(args, "mec_set_actor_encoder", "shared")
        ).lower()
        if (
            self.architecture == "set"
            and set_actor_encoder_mode == "separate"
            and set_critic_encoder_mode != "separate"
        ):
            raise ValueError(
                "mec_set_actor_encoder=separate requires "
                "mec_set_critic_encoder=separate so the critic does not "
                "implicitly choose one actor encoder"
            )
        self.recompute_values_after_actor_update = (
            self.architecture == "set"
            and set_critic_encoder_mode in {"actor_detached", "shared_grad"}
        )

        expected_obs_dim = AGENT_OBS_DIM
        expected_cent_dim = team_state_dim(self.num_agents)
        if self.architecture != "legacy_mean":
            if obs_dim != expected_obs_dim:
                raise ValueError(
                    f"{self.architecture} expects local obs dim "
                    f"{expected_obs_dim}, got {obs_dim}"
                )
            if cent_dim != expected_cent_dim:
                raise ValueError(
                    f"{self.architecture} expects centralized state dim "
                    f"{expected_cent_dim}, got {cent_dim}"
                )

        if self.architecture == "legacy_mean":
            self.actor = MECLegacyMeanActor(
                args,
                obs_space,
                act_space,
                self.num_agents,
                device,
            )
            self.critic = MECLegacyMeanCritic(
                args, self.num_agents, device
            )
        elif self.architecture == "mean":
            self.actor = MECMeanActor(
                args,
                obs_space,
                act_space,
                self.num_agents,
                device,
            )
            self.critic = MECTeamCritic(
                args,
                cent_obs_space,
                self.num_agents,
                "mean",
                device=device,
            )
        elif self.architecture == "flat":
            self.actor = MECFlatActor(
                args,
                obs_space,
                act_space,
                self.num_agents,
                device,
            )
            self.critic = MECTeamCritic(
                args,
                cent_obs_space,
                self.num_agents,
                "flat",
                device=device,
            )
        elif self.architecture == "set":
            self.actor = MECSetActor(
                args,
                obs_space,
                act_space,
                self.num_agents,
                device,
            )
            population_encoder = (
                self.actor.population_encoder
                if set_actor_encoder_mode == "shared"
                else None
            )
            self.critic = MECTeamCritic(
                args,
                cent_obs_space,
                self.num_agents,
                "set",
                population_encoder=population_encoder,
                device=device,
            )
        else:
            raise ValueError(
                "mec_policy_arch must be one of: "
                "legacy_mean, mean, flat, set"
            )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def load_set_pretrained_actor(self, actor_state_dict):
        """Load only Set representation weights from a pretrained actor."""
        if self.architecture != "set":
            raise ValueError("Set pretraining weights require set policy")
        if getattr(self.actor, "set_actor_encoder_mode", "shared") != "shared":
            raise ValueError(
                "Set pretraining weights currently require "
                "mec_set_actor_encoder=shared"
            )
        actor_state = self.actor.state_dict()
        allowed_prefixes = (
            "population_encoder.",
            "reconstruction_decoder.",
        )
        filtered = {
            key: value
            for key, value in actor_state_dict.items()
            if key.startswith(allowed_prefixes)
            and key in actor_state
            and actor_state[key].shape == value.shape
        }
        encoder_keys = [
            key for key in filtered if key.startswith("population_encoder.")
        ]
        if not encoder_keys:
            raise ValueError(
                "pretrained actor does not contain compatible "
                "population_encoder weights"
            )
        actor_state.update(filtered)
        self.actor.load_state_dict(actor_state)
        return sorted(filtered)

    def set_set_encoder_trainable(self, trainable: bool):
        if self.architecture != "set":
            return 0
        if getattr(self.actor, "set_actor_encoder_mode", "shared") != "shared":
            return 0
        count = 0
        for parameter in self.actor.population_encoder.parameters():
            parameter.requires_grad_(bool(trainable))
            count += 1
        return count

    def mec_set_reconstruction_loss(self, cent_obs):
        if self.architecture != "set" or not getattr(
            self.actor, "use_reconstruction_aux", False
        ):
            raise RuntimeError("MEC Set reconstruction auxiliary is disabled")
        cent_obs = check(cent_obs).to(
            dtype=torch.float32, device=self.device
        )
        if cent_obs.shape[-1] != team_state_dim(self.num_agents):
            raise ValueError(
                "reconstruction auxiliary expects canonical centralized "
                f"state dim {team_state_dim(self.num_agents)}, "
                f"got {cent_obs.shape[-1]}"
            )
        if cent_obs.shape[0] % self.num_agents != 0:
            raise ValueError(
                "reconstruction auxiliary requires complete K+1 groups"
            )
        grouped = cent_obs.reshape(
            -1, self.num_agents, cent_obs.shape[-1]
        )
        team_state = grouped[:, 0]
        uavs = team_state[:, PUBLIC_STATE_DIM:].reshape(
            -1, self.num_agents - 1, UAV_STATE_DIM
        )
        return self.actor.reconstruction_chamfer_loss(uavs)
