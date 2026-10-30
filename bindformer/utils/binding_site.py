from __future__ import annotations

import numpy as np
import torch

from bindformer.utils.pdb_utils import ProteinStructure


def select_binding_site_residues(
    protein_coords: torch.Tensor,
    ligand_coords: torch.Tensor,
    radius: float = 10.0,
) -> torch.Tensor:
    """Heuristic pocket selection by ligand proximity."""

    return torch.cdist(protein_coords, ligand_coords).amin(dim=-1) <= radius


def residue_atom_min_distances(structure: ProteinStructure, ligand_coords: np.ndarray) -> np.ndarray:
    """`[residues, ligand_atoms]` minimum heavy-atom distance between each residue and ligand atom."""

    ligand_coords = np.asarray(ligand_coords, dtype=np.float32)
    out = np.empty((len(structure.residues), len(ligand_coords)), dtype=np.float32)
    for i, res in enumerate(structure.residues):
        atoms = np.asarray(res.atom_coords, dtype=np.float32)
        out[i] = np.linalg.norm(atoms[:, None, :] - ligand_coords[None, :, :], axis=-1).min(axis=0)
    return out


def crop_pocket(
    structure: ProteinStructure,
    ligand_coords: np.ndarray,
    cutoff: float = 10.0,
    contact_cutoff: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select residues with any heavy atom within `cutoff` of the ligand.

    Returns the selected residue indices and a `[pocket_residues, ligand_atoms]`
    float contact map (1 where the residue-atom heavy-atom distance <= `contact_cutoff`).
    """

    dist = residue_atom_min_distances(structure, ligand_coords)
    pocket_idx = np.nonzero(dist.min(axis=1) <= cutoff)[0]
    contacts = (dist[pocket_idx] <= contact_cutoff).astype(np.float32)
    return pocket_idx, contacts
