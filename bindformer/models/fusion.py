from __future__ import annotations

import torch
from torch import nn

from bindformer.models.pooling import AttentionPool


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x)).sum(dim=1) / denom


class FusionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        attention_pool: bool = False,
        pool_scorer_hidden: int = 0,
        pool_temperature: float = 1.0,
    ):
        super().__init__()
        def make_pool():
            return AttentionPool(hidden_dim, pool_scorer_hidden, pool_temperature) if attention_pool else None

        self.protein_pool = make_pool()
        self.ligand_pool = make_pool()
        layers = []
        dim = hidden_dim * 2
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.affinity = nn.Linear(hidden_dim, 1)
        self.contact = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, protein, ligand, protein_mask=None, ligand_mask=None):
        pool_p = masked_mean if self.protein_pool is None else self.protein_pool
        pool_l = masked_mean if self.ligand_pool is None else self.ligand_pool
        pooled = torch.cat([pool_p(protein, protein_mask), pool_l(ligand, ligand_mask)], dim=-1)
        z = self.trunk(pooled)
        affinity = self.affinity(z).squeeze(-1)
        pair = torch.cat(
            [
                protein.unsqueeze(2).expand(-1, -1, ligand.shape[1], -1),
                ligand.unsqueeze(1).expand(-1, protein.shape[1], -1, -1),
            ],
            dim=-1,
        )
        contacts = self.contact(pair).squeeze(-1)
        return {"affinity": affinity, "contact_logits": contacts}

