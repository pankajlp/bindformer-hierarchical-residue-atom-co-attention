from __future__ import annotations

import torch
from torch import nn

from bindformer.models.fusion import FusionHead
from bindformer.models.ligand_encoder import LigandEncoder
from bindformer.models.protein_encoder import ProteinEncoder
from bindformer.models.raa_attention import ResidueAtomAttention
from bindformer.utils.sparse_mask import build_distance_attention_mask


def _zero_padding(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero padded positions so the pooled and pairwise heads never read padding."""

    if mask is None:
        return x
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))


class BindFormer(nn.Module):
    def __init__(
        self,
        protein_dim: int = 1280,
        ligand_dim: int = 74,
        hidden_dim: int = 256,
        num_heads: int = 8,
        protein_layers: int = 3,
        ligand_layers: int = 3,
        raa_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        pocket_radius_angstrom: float = 10.0,
        rbf_kernels: int = 0,
        protein_rbf_cutoff: float = 20.0,
        raa_rbf_cutoff: float | None = None,
        attention_pool: bool = False,
        pool_scorer_hidden: int = 0,
        pool_temperature: float = 1.0,
    ):
        super().__init__()
        self.pocket_radius_angstrom = pocket_radius_angstrom
        # Co-attention is already masked at the pocket radius, so bias over that same range.
        raa_cutoff = pocket_radius_angstrom if raa_rbf_cutoff is None else raa_rbf_cutoff
        self.protein_encoder = ProteinEncoder(
            protein_dim, hidden_dim, num_heads, protein_layers, dropout, rbf_kernels, protein_rbf_cutoff
        )
        self.ligand_encoder = LigandEncoder(ligand_dim, hidden_dim, num_heads, ligand_layers, dropout)
        self.raa = ResidueAtomAttention(hidden_dim, num_heads, raa_layers, dropout, rbf_kernels, raa_cutoff)
        self.fusion = FusionHead(
            hidden_dim, fusion_layers, dropout, attention_pool, pool_scorer_hidden, pool_temperature
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        protein_mask = batch.get("protein_mask")
        ligand_mask = batch.get("ligand_mask")
        protein = _zero_padding(
            self.protein_encoder(batch["protein_x"], protein_mask, batch.get("protein_aa"), batch.get("protein_coords")),
            protein_mask,
        )
        ligand = _zero_padding(self.ligand_encoder(batch["ligand_x"], ligand_mask, batch.get("ligand_spd")), ligand_mask)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = build_distance_attention_mask(
                batch["protein_coords"],
                batch["ligand_coords"],
                self.pocket_radius_angstrom,
                protein_mask,
                ligand_mask,
            )
        distances = None
        if self.raa.rbf is not None:
            distances = torch.cdist(batch["protein_coords"], batch["ligand_coords"])
        protein, ligand, attentions = self.raa(protein, ligand, attention_mask, distances)
        out = self.fusion(protein, ligand, protein_mask, ligand_mask)
        out["raa_attentions"] = attentions
        out["attention_mask"] = attention_mask
        return out

