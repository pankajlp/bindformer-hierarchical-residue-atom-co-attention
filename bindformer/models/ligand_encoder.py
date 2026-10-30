from __future__ import annotations

import torch
from torch import nn

from bindformer.data.ligand_featurizer import MAX_SPD
from bindformer.models.layers import BiasedTransformer


class LigandEncoder(nn.Module):
    """Transformer over ligand atoms with a learned per-head shortest-path-distance bias.

    The topological distance bias (Graphormer-style) gives the encoder the bond graph;
    without `spd` it falls back to a plain set transformer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.spd_bias = nn.Embedding(MAX_SPD + 2, num_heads)
        nn.init.zeros_(self.spd_bias.weight)
        self.encoder = BiasedTransformer(hidden_dim, num_heads, num_layers, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        spd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        bias = None
        if spd is not None:
            bias = self.spd_bias(spd.clamp(0, MAX_SPD + 1)).permute(0, 3, 1, 2)  # [B, H, N, N]
        return self.encoder(h, mask, bias)
