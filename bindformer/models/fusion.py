from __future__ import annotations

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x)).sum(dim=1) / denom


class FusionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        dim = hidden_dim * 2
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.affinity = nn.Linear(hidden_dim, 1)
        self.contact = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, protein, ligand, protein_mask=None, ligand_mask=None):
        pooled = torch.cat([masked_mean(protein, protein_mask), masked_mean(ligand, ligand_mask)], dim=-1)
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

