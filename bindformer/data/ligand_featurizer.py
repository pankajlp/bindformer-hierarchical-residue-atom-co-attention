from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Same layout as DGL-LifeSci's CanonicalAtomFeaturizer (74 dims).
ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I", "B", "V",
    "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni",
    "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]
DEGREES = list(range(11))
IMPLICIT_VALENCES = list(range(7))
TOTAL_HS = list(range(5))
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
ATOM_FEATURE_DIM = len(ATOM_TYPES) + len(DEGREES) + len(IMPLICIT_VALENCES) + 2 + len(HYBRIDIZATIONS) + 1 + len(TOTAL_HS)
BOND_FEATURE_DIM = 6  # single, double, triple, aromatic, conjugated, in ring
MAX_SPD = 8  # shortest-path distances above this (or disconnected) are clipped to MAX_SPD + 1


@dataclass
class LigandFeatures:
    x: torch.Tensor  # [atoms, 74]
    coords: torch.Tensor  # [atoms, 3]
    atom_symbols: list[str]
    edge_index: torch.Tensor  # [2, 2 * bonds]
    edge_attr: torch.Tensor  # [2 * bonds, 6]
    spd: torch.Tensor  # [atoms, atoms] uint8 topological distance
    source: str = "unknown"  # which parser succeeded: sdf, mol2, or sdf_partial (unsanitized)


def _one_hot(value, choices) -> list[float]:
    return [float(value == c) for c in choices]


def _implicit_valence(atom) -> int:
    try:
        from rdkit.Chem import ValenceType

        return atom.GetValence(ValenceType.IMPLICIT)
    except (ImportError, AttributeError):
        return atom.GetImplicitValence()


def atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(_implicit_valence(atom), IMPLICIT_VALENCES)
        + [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
        + _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic())]
        + _one_hot(atom.GetTotalNumHs(), TOTAL_HS)
    )


def bond_features(bond) -> list[float]:
    bt = str(bond.GetBondType())
    return [
        float(bt == "SINGLE"),
        float(bt == "DOUBLE"),
        float(bt == "TRIPLE"),
        float(bt == "AROMATIC"),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def _heavy_atoms_only(mol):
    """Drop every hydrogen, keeping the conformer.

    `removeHs=True` is not enough: RDKit spares hydrogens it considers meaningful, and
    the CASF-2016 coreset SDFs carry explicit hydrogens where the PDBbind general and
    refined SDFs carry none. Left alone that silently makes the test ligands about
    twice as long as the training ones. `RemoveAllHs` strips them unconditionally.
    """

    from rdkit import Chem

    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        return mol
    stripped = Chem.RemoveAllHs(mol, sanitize=False)
    return stripped if stripped is not None and stripped.GetNumConformers() else mol


def load_ligand(path: str | Path):
    """Load a ligand with heavy atoms only, trying progressively looser parsing.

    PDBbind SDF files occasionally fail sanitization; the matching `.mol2` file is
    tried next, then an unsanitized SDF with partial sanitization.
    """

    from rdkit import Chem

    path = Path(path)
    candidates = [path]
    if path.suffix == ".sdf":
        candidates.append(path.with_suffix(".mol2"))

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".sdf":
            mol = next(iter(Chem.SDMolSupplier(str(candidate), removeHs=True, sanitize=True)), None)
        else:
            mol = Chem.MolFromMol2File(str(candidate), removeHs=True, sanitize=True)
        if mol is not None and mol.GetNumConformers():
            mol = _heavy_atoms_only(mol)
            mol.SetProp("bf_source", candidate.suffix.lstrip("."))
            return mol

    if path.suffix == ".sdf" and path.exists():
        mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)), None)
        if mol is not None:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
                catchErrors=True,
            )
            mol = _heavy_atoms_only(Chem.RemoveHs(mol, sanitize=False))
            if mol.GetNumConformers():
                mol.SetProp("bf_source", "sdf_partial")
                return mol
    raise ValueError(f"Could not parse ligand from {path}")


class LigandFeaturizer:
    """Heavy-atom ligand featurizer: 74-dim atom features, bond graph, and topological distances."""

    def __init__(self, feature_dim: int = ATOM_FEATURE_DIM):
        if feature_dim != ATOM_FEATURE_DIM:
            raise ValueError(f"LigandFeaturizer produces {ATOM_FEATURE_DIM}-dim features, got {feature_dim}")
        self.feature_dim = feature_dim

    def featurize_sdf(self, sdf_path: str | Path) -> LigandFeatures:
        return self.featurize_mol(load_ligand(sdf_path))

    def featurize_mol(self, mol) -> LigandFeatures:
        from rdkit import Chem

        conf = mol.GetConformer()
        atoms = list(mol.GetAtoms())
        x = torch.tensor([atom_features(a) for a in atoms], dtype=torch.float32).reshape(len(atoms), ATOM_FEATURE_DIM)
        coords = torch.tensor(conf.GetPositions(), dtype=torch.float32).reshape(len(atoms), 3)

        src, dst, attrs = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feats = bond_features(bond)
            src += [i, j]
            dst += [j, i]
            attrs += [feats, feats]
        edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
        edge_attr = torch.tensor(attrs, dtype=torch.float32).reshape(-1, BOND_FEATURE_DIM)

        spd = np.asarray(Chem.GetDistanceMatrix(mol))
        spd = np.where(spd > MAX_SPD, MAX_SPD + 1, spd).astype(np.uint8)

        return LigandFeatures(
            x=x,
            coords=coords,
            atom_symbols=[a.GetSymbol() for a in atoms],
            edge_index=edge_index,
            edge_attr=edge_attr,
            spd=torch.from_numpy(spd),
            source=mol.GetProp("bf_source") if mol.HasProp("bf_source") else "unknown",
        )
