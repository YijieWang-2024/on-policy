"""Permutation-aware building blocks for MEC population policies."""

from __future__ import annotations

import torch
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.util import init


def _activation(use_relu: bool) -> nn.Module:
    return nn.ReLU() if use_relu else nn.Tanh()


class ResidualAttentionBlock(nn.Module):
    """Multi-head attention block with residual FFN and layer normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_relu: bool,
        use_orthogonal: bool,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            _activation(use_relu),
            nn.Linear(2 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self._init_parameters(use_relu, use_orthogonal)

    def _init_parameters(
        self, use_relu: bool, use_orthogonal: bool
    ) -> None:
        init_method = [
            nn.init.xavier_uniform_, nn.init.orthogonal_
        ][use_orthogonal]
        act_gain = nn.init.calculate_gain(["tanh", "relu"][use_relu])

        init(
            self.ffn[0],
            init_method,
            lambda x: nn.init.constant_(x, 0),
            gain=act_gain,
        )
        init(
            self.ffn[2],
            init_method,
            lambda x: nn.init.constant_(x, 0),
            gain=1.0,
        )
        init_method(self.attn.in_proj_weight, gain=1.0)
        if self.attn.in_proj_bias is not None:
            nn.init.constant_(self.attn.in_proj_bias, 0)
        init(
            self.attn.out_proj,
            init_method,
            lambda x: nn.init.constant_(x, 0),
            gain=1.0,
        )

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.attn(
            query, key_value, key_value, need_weights=False
        )
        hidden = self.norm1(query + attended)
        return self.norm2(hidden + self.ffn(hidden))


class SetAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention over a set."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_relu: bool,
        use_orthogonal: bool,
    ):
        super().__init__()
        self.block = ResidualAttentionBlock(
            dim, num_heads, use_relu, use_orthogonal
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.block(tokens, tokens)


class PopulationEncoder(nn.Module):
    """Encode an unordered UAV state set into invariant latent slots."""

    def __init__(
        self,
        atom_dim: int,
        model_dim: int,
        num_heads: int,
        num_seeds: int,
        element_blocks: int,
        latent_blocks: int,
        use_relu: bool,
        layer_N: int,
        use_orthogonal: bool,
        use_feature_normalization: bool,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_seeds = num_seeds
        self._use_feature_normalization = bool(
            use_feature_normalization
        )
        if self._use_feature_normalization:
            self.atom_norm = nn.LayerNorm(atom_dim)
        self.atom_encoder = MLPLayer(
            atom_dim,
            model_dim,
            int(layer_N),
            bool(use_orthogonal),
            bool(use_relu),
        )
        self.element_blocks = nn.ModuleList(
            SetAttentionBlock(
                model_dim, num_heads, use_relu, use_orthogonal
            )
            for _ in range(element_blocks)
        )
        self.population_seeds = nn.Parameter(
            torch.empty(1, num_seeds, model_dim)
        )
        nn.init.normal_(self.population_seeds, std=0.02)
        self.pool = ResidualAttentionBlock(
            model_dim, num_heads, use_relu, use_orthogonal
        )
        self.latent_blocks = nn.ModuleList(
            SetAttentionBlock(
                model_dim, num_heads, use_relu, use_orthogonal
            )
            for _ in range(latent_blocks)
        )

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return invariant slots with shape ``[batch, M, model_dim]``."""
        tokens = self.encode_tokens(atoms)
        return self.pool_tokens(tokens)

    def encode_tokens(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return permutation-equivariant UAV tokens."""
        if self._use_feature_normalization:
            atoms = self.atom_norm(atoms)
        tokens = self.atom_encoder(atoms)
        for block in self.element_blocks:
            tokens = block(tokens)
        return tokens

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return permutation-invariant latent slots from encoded tokens."""
        seeds = self.population_seeds.expand(tokens.shape[0], -1, -1)
        latent = self.pool(seeds, tokens)
        for block in self.latent_blocks:
            latent = block(latent)
        return latent

    def forward_with_tokens(
        self, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return invariant slots and equivariant per-UAV tokens."""
        tokens = self.encode_tokens(atoms)
        return self.pool_tokens(tokens), tokens


class MeanPoolPopulationEncoder(nn.Module):
    """Simple permutation-invariant encoder: self-attention + mean pooling."""

    def __init__(
        self,
        atom_dim: int,
        model_dim: int,
        num_heads: int,
        element_blocks: int,
        use_relu: bool,
        layer_N: int,
        use_orthogonal: bool,
        use_feature_normalization: bool,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_seeds = 1
        self._use_feature_normalization = bool(
            use_feature_normalization
        )
        if self._use_feature_normalization:
            self.atom_norm = nn.LayerNorm(atom_dim)
        self.atom_encoder = MLPLayer(
            atom_dim,
            model_dim,
            int(layer_N),
            bool(use_orthogonal),
            bool(use_relu),
        )
        self.element_blocks = nn.ModuleList(
            SetAttentionBlock(
                model_dim, num_heads, use_relu, use_orthogonal
            )
            for _ in range(element_blocks)
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def encode_tokens(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return permutation-equivariant UAV tokens."""
        if self._use_feature_normalization:
            atoms = self.atom_norm(atoms)
        tokens = self.atom_encoder(atoms)
        for block in self.element_blocks:
            tokens = block(tokens)
        return tokens

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return an invariant mean-pooled population descriptor."""
        return self.output_norm(tokens.mean(dim=1))

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return invariant descriptor with shape ``[batch, model_dim]``."""
        return self.pool_tokens(self.encode_tokens(atoms))

    def forward_with_tokens(
        self, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return invariant descriptor and equivariant per-UAV tokens."""
        tokens = self.encode_tokens(atoms)
        return self.pool_tokens(tokens), tokens


class FlatMLPPopulationEncoder(nn.Module):
    """Ordered fixed-K population encoder using one plain MLP."""

    def __init__(
        self,
        atom_dim: int,
        num_atoms: int,
        model_dim: int,
        use_relu: bool,
        layer_N: int,
        use_orthogonal: bool,
        use_feature_normalization: bool,
    ):
        super().__init__()
        if int(num_atoms) <= 0:
            raise ValueError("flat_mlp population encoder requires num_atoms")
        self.atom_dim = int(atom_dim)
        self.num_atoms = int(num_atoms)
        self.model_dim = int(model_dim)
        self.flat_dim = self.atom_dim * self.num_atoms
        self._use_feature_normalization = bool(
            use_feature_normalization
        )
        if self._use_feature_normalization:
            self.atom_norm = nn.LayerNorm(self.flat_dim)
        self.atom_encoder = MLPLayer(
            self.flat_dim,
            self.model_dim,
            int(layer_N),
            bool(use_orthogonal),
            bool(use_relu),
        )

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return ordered fixed-K descriptor with shape ``[batch, model_dim]``."""
        if atoms.shape[1] != self.num_atoms:
            raise ValueError(
                f"flat_mlp expected {self.num_atoms} atoms, "
                f"got {atoms.shape[1]}"
            )
        features = atoms.reshape(atoms.shape[0], self.flat_dim)
        if self._use_feature_normalization:
            features = self.atom_norm(features)
        return self.atom_encoder(features)


class PopulationReconstructionDecoder(nn.Module):
    """Decode a population descriptor into an unordered fixed-K atom set."""

    def __init__(
        self,
        representation_dim: int,
        num_atoms: int,
        atom_dim: int,
        hidden_dim: int,
        use_relu: bool,
        layer_N: int,
        use_orthogonal: bool,
    ):
        super().__init__()
        if int(num_atoms) <= 0:
            raise ValueError("reconstruction decoder requires num_atoms")
        self.num_atoms = int(num_atoms)
        self.atom_dim = int(atom_dim)
        self.trunk = MLPLayer(
            int(representation_dim),
            int(hidden_dim),
            int(layer_N),
            bool(use_orthogonal),
            bool(use_relu),
        )
        init_method = [
            nn.init.xavier_uniform_, nn.init.orthogonal_
        ][bool(use_orthogonal)]
        self.out = init(
            nn.Linear(int(hidden_dim), self.num_atoms * self.atom_dim),
            init_method,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01,
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        decoded = self.out(self.trunk(representation))
        return decoded.reshape(
            representation.shape[0], self.num_atoms, self.atom_dim
        )


def chamfer_set_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Symmetric squared Chamfer loss for unordered fixed-size atom sets."""
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("Chamfer set loss expects [batch, atoms, dim]")
    if prediction.shape[0] != target.shape[0]:
        raise ValueError("Chamfer set loss batch sizes must match")
    if prediction.shape[2] != target.shape[2]:
        raise ValueError("Chamfer set loss atom dimensions must match")
    distances = torch.cdist(prediction, target, p=2).pow(2)
    pred_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_pred = distances.min(dim=1).values.mean(dim=1)
    return 0.5 * (pred_to_target + target_to_pred).mean()


def population_encoder_type(args) -> str:
    return str(getattr(args, "mec_set_encoder_type", "latent_slots")).lower()


def population_representation_dim(args) -> int:
    model_dim = int(getattr(args, "mec_set_dim", 64))
    if population_encoder_type(args) in {"mean_pool", "flat_mlp"}:
        return model_dim
    return model_dim * int(getattr(args, "mec_set_num_seeds", 4))


def build_population_encoder(
    args,
    *,
    atom_dim: int,
    use_relu: bool,
    num_atoms: int | None = None,
) -> nn.Module:
    encoder_type = population_encoder_type(args)
    model_dim = int(getattr(args, "mec_set_dim", 64))
    num_heads = int(getattr(args, "mec_set_heads", 4))
    if encoder_type == "latent_slots":
        return PopulationEncoder(
            atom_dim=atom_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_seeds=int(getattr(args, "mec_set_num_seeds", 4)),
            element_blocks=int(
                getattr(args, "mec_set_element_blocks", 2)
            ),
            latent_blocks=int(
                getattr(args, "mec_set_latent_blocks", 1)
            ),
            use_relu=use_relu,
            layer_N=int(getattr(args, "layer_N", 1)),
            use_orthogonal=bool(
                getattr(args, "use_orthogonal", True)
            ),
            use_feature_normalization=bool(
                getattr(args, "use_feature_normalization", True)
            ),
        )
    if encoder_type == "mean_pool":
        return MeanPoolPopulationEncoder(
            atom_dim=atom_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            element_blocks=int(
                getattr(args, "mec_set_element_blocks", 2)
            ),
            use_relu=use_relu,
            layer_N=int(getattr(args, "layer_N", 1)),
            use_orthogonal=bool(
                getattr(args, "use_orthogonal", True)
            ),
            use_feature_normalization=bool(
                getattr(args, "use_feature_normalization", True)
            ),
        )
    if encoder_type == "flat_mlp":
        if num_atoms is None:
            raise ValueError(
                "flat_mlp population encoder requires num_atoms"
            )
        return FlatMLPPopulationEncoder(
            atom_dim=atom_dim,
            num_atoms=int(num_atoms),
            model_dim=model_dim,
            use_relu=use_relu,
            layer_N=int(getattr(args, "layer_N", 1)),
            use_orthogonal=bool(
                getattr(args, "use_orthogonal", True)
            ),
            use_feature_normalization=bool(
                getattr(args, "use_feature_normalization", True)
            ),
        )
    raise ValueError(
        "mec_set_encoder_type must be one of: latent_slots, "
        "mean_pool, flat_mlp"
    )


class FusionMLP(nn.Module):
    """Role-specific MLP shared by aligned MEC architectures.

    Keep the same optimization contract as the legacy MAPPO ``MLPBase``:
    optional input LayerNorm, configured orthogonal/Xavier initialization,
    ``layer_N`` hidden blocks, and per-hidden-layer LayerNorm.  The aligned
    Mean/Flat/Set policies differ in what population descriptor they feed into
    this readout, not in the basic PPO-friendly MLP numerics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_relu: bool,
        *,
        layer_N: int = 1,
        use_orthogonal: bool = True,
        use_feature_normalization: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self._use_feature_normalization = bool(
            use_feature_normalization
        )
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(self.input_dim)
        self.mlp = MLPLayer(
            self.input_dim,
            self.hidden_dim,
            int(layer_N),
            bool(use_orthogonal),
            bool(use_relu),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self._use_feature_normalization:
            features = self.feature_norm(features)
        return self.mlp(features)
