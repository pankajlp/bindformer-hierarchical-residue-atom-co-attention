from __future__ import annotations

import torch


def build_distance_attention_mask(
    protein_coords: torch.Tensor,
    ligand_coords: torch.Tensor,
    radius: float = 10.0,
    protein_mask: torch.Tensor | None = None,
    ligand_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a boolean `[batch, residues, atoms]` mask for residue-atom attention."""

    distances = torch.cdist(protein_coords, ligand_coords)
    mask = distances <= radius
    if protein_mask is not None:
        mask = mask & protein_mask.unsqueeze(-1)
    if ligand_mask is not None:
        mask = mask & ligand_mask.unsqueeze(1)
    return mask

