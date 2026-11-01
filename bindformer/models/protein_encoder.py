from __future__ import annotations

import torch
from torch import nn

from bindformer.models.layers import BiasedTransformer
from bindformer.models.rbf import RadialBasisBias
from bindformer.utils.pdb_utils import UNKNOWN_AA_INDEX


class ProteinEncoder(nn.Module):
    """Transformer over pocket residues; adds a learned amino-acid embedding when `aa` is given.

    With `rbf_kernels > 0` and CA coordinates, residue-residue distances enter
    self-attention as an additive per-head bias, giving the encoder pocket geometry
    rather than an unordered residue set.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        rbf_kernels: int = 0,
        rbf_cutoff: float = 20.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.aa_embedding = nn.Embedding(UNKNOWN_AA_INDEX + 1, hidden_dim)
        self.rbf = RadialBasisBias(num_heads, rbf_kernels, rbf_cutoff) if rbf_kernels > 0 else None
        self.encoder = BiasedTransformer(hidden_dim, num_heads, num_layers, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        aa: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        if aa is not None:
            h = h + self.aa_embedding(aa)
        bias = None
        if self.rbf is not None and coords is not None:
            bias = self.rbf(torch.cdist(coords, coords))
        return self.encoder(h, mask, bias)
