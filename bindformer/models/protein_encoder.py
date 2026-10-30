from __future__ import annotations

import torch
from torch import nn

from bindformer.models.layers import BiasedTransformer
from bindformer.utils.pdb_utils import UNKNOWN_AA_INDEX


class ProteinEncoder(nn.Module):
    """Transformer over pocket residues; adds a learned amino-acid embedding when `aa` is given."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.aa_embedding = nn.Embedding(UNKNOWN_AA_INDEX + 1, hidden_dim)
        self.encoder = BiasedTransformer(hidden_dim, num_heads, num_layers, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, aa: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        if aa is not None:
            h = h + self.aa_embedding(aa)
        return self.encoder(h, mask)
