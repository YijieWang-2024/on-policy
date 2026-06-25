"""Permutation-aware building blocks for MEC population policies."""

from __future__ import annotations

import torch
import torch.nn as nn


def _activation(use_relu: bool) -> nn.Module:
    return nn.ReLU() if use_relu else nn.Tanh()


class ResidualAttentionBlock(nn.Module):
    """Multi-head attention block with residual FFN and layer normalization."""

    def __init__(self, dim: int, num_heads: int, use_relu: bool):
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

    def __init__(self, dim: int, num_heads: int, use_relu: bool):
        super().__init__()
        self.block = ResidualAttentionBlock(dim, num_heads, use_relu)

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
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_seeds = num_seeds
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, model_dim),
            _activation(use_relu),
            nn.Linear(model_dim, model_dim),
            _activation(use_relu),
        )
        self.element_blocks = nn.ModuleList(
            SetAttentionBlock(model_dim, num_heads, use_relu)
            for _ in range(element_blocks)
        )
        self.population_seeds = nn.Parameter(
            torch.empty(1, num_seeds, model_dim)
        )
        nn.init.normal_(self.population_seeds, std=0.02)
        self.pool = ResidualAttentionBlock(
            model_dim, num_heads, use_relu
        )
        self.latent_blocks = nn.ModuleList(
            SetAttentionBlock(model_dim, num_heads, use_relu)
            for _ in range(latent_blocks)
        )

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """Return invariant slots with shape ``[batch, M, model_dim]``."""
        tokens = self.atom_encoder(atoms)
        for block in self.element_blocks:
            tokens = block(tokens)
        seeds = self.population_seeds.expand(atoms.shape[0], -1, -1)
        latent = self.pool(seeds, tokens)
        for block in self.latent_blocks:
            latent = block(latent)
        return latent


class FusionMLP(nn.Module):
    """Small role-specific MLP shared by aligned MEC architectures."""

    def __init__(
        self, input_dim: int, hidden_dim: int, use_relu: bool
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(use_relu),
            nn.Linear(hidden_dim, hidden_dim),
            _activation(use_relu),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
