"""Ligands must reach the model as heavy atoms only, whatever the source file holds.

PDBbind's general and refined SDFs carry no hydrogens while the CASF-2016 coreset
SDFs do, and RDKit's `removeHs=True` does not strip them. Unnoticed, that made the
test ligands roughly twice as long as the training ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bindformer.data.ligand_featurizer import LigandFeaturizer, load_ligand
from tests.conftest import write_ligand


@pytest.fixture
def sdf_with_hydrogens(tmp_path: Path) -> Path:
    path = tmp_path / "lig.sdf"
    write_ligand(path, "CC(=O)Oc1ccccc1C(=O)O", seed=1)  # conftest writes these with explicit Hs
    return path


def test_source_file_really_has_hydrogens(sdf_with_hydrogens: Path):
    """Guard the fixture itself: without Hs in the file the test below proves nothing."""

    assert " H " in sdf_with_hydrogens.read_text() or "H   0" in sdf_with_hydrogens.read_text()


def test_loaded_ligand_has_no_hydrogens(sdf_with_hydrogens: Path):
    mol = load_ligand(sdf_with_hydrogens)
    assert not [a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    assert mol.GetNumConformers() == 1, "stripping hydrogens must keep the pose"


def test_featurized_atoms_are_heavy_only(sdf_with_hydrogens: Path):
    lig = LigandFeaturizer().featurize_sdf(sdf_with_hydrogens)
    assert "H" not in lig.atom_symbols
    assert lig.x.shape[0] == len(lig.atom_symbols) == lig.coords.shape[0]
    assert lig.spd.shape == (len(lig.atom_symbols), len(lig.atom_symbols))


def test_heavy_atom_count_matches_the_molecule(sdf_with_hydrogens: Path):
    from rdkit import Chem

    expected = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O").GetNumAtoms()
    assert LigandFeaturizer().featurize_sdf(sdf_with_hydrogens).x.shape[0] == expected
