from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccc2[nH]ccc2c1",
    "CN1CCN(CC1)c1ccc(cc1)C(N)=O",
    "OC(=O)CCc1ccccc1",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "NC(=O)c1cccnc1",
    "CCOC(=O)c1ccc(O)cc1",
    "c1ccc(cc1)-c1ccncc1",
]
RESIDUES = ["ALA", "GLY", "SER", "LEU", "PHE", "LYS", "ASP", "TYR", "MSE", "HIS"]


def pdb_atom_line(record, serial, name, resname, chain, resseq, xyz, element, altloc=" ", icode=" "):
    return (
        f"{record:<6}{serial:>5} {name:<4}{altloc}{resname:>3} {chain}{resseq:>4}{icode}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{0.0:6.2f}{'':10}{element:>2}\n"
    )


def write_protein(path: Path, ligand_center: np.ndarray, rng: np.random.Generator, n_residues: int = 40) -> None:
    """Write a toy two-chain protein: residues scattered 3-20 A from the ligand, plus waters and H atoms."""

    lines, serial = [], 1
    for i in range(n_residues):
        chain = "A" if i < n_residues // 2 else "B"
        resname = RESIDUES[i % len(RESIDUES)]
        record = "HETATM" if resname == "MSE" else "ATOM"
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        ca = ligand_center + direction * rng.uniform(3.0, 20.0)
        for name, element, offset in (("N", "N", (-1.2, 0.4, 0)), ("CA", "C", (0, 0, 0)), ("C", "C", (1.2, 0.4, 0)), ("O", "O", (1.8, 1.2, 0)), ("H", "H", (-1.8, 0.0, 0.3))):
            lines.append(pdb_atom_line(record, serial, name, resname, chain, i + 1, ca + np.array(offset), element))
            serial += 1
    # Alternate location B for the first residue's CA must be ignored.
    lines.append(pdb_atom_line("ATOM", serial, "CA", RESIDUES[0], "A", 1, ligand_center + 100.0, "C", altloc="B"))
    serial += 1
    for j in range(3):
        lines.append(pdb_atom_line("HETATM", serial, "O", "HOH", "W", 900 + j, ligand_center + rng.normal(size=3), "O"))
        serial += 1
    lines.append(pdb_atom_line("HETATM", serial, "ZN", "ZN", "A", 999, ligand_center + 2.0, "ZN"))
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def write_ligand(path: Path, smiles: str, seed: int) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return mol.GetConformer().GetPositions().mean(axis=0)


@pytest.fixture
def pdbbind_tree(tmp_path: Path) -> dict:
    """A miniature PDBbind v2020 + CASF-2016 directory tree."""

    rng = np.random.default_rng(0)
    root = tmp_path / "pdbbind"
    general = root / "v2020-other-PL"
    refined = root / "refined-set"
    casf = root / "CASF-2016"
    (root / "index").mkdir(parents=True)
    (casf / "power_scoring").mkdir(parents=True)

    codes = [f"{i}ab{chr(97 + i)}" for i in range(1, 9)] + [f"{i}cd{chr(97 + i)}" for i in range(1, 9)]
    core_codes = codes[-3:]
    refined_codes = codes[:5]
    index_general, index_refined, core_lines = [], [], []
    for n, code in enumerate(codes):
        if code in core_codes:
            base = casf / "coreset"
        elif code in refined_codes:
            base = refined
        else:
            base = general
        cdir = base / code
        cdir.mkdir(parents=True)
        smiles = SMILES[n % len(SMILES)]
        center = write_ligand(cdir / f"{code}_ligand.sdf", smiles, seed=n)
        write_protein(cdir / f"{code}_protein.pdb", center, rng)
        pk = 4.0 + 0.3 * n
        year = 2010 + n
        line = f"{code}  2.00  {year}   {pk:.2f}  Kd={10 ** (9 - pk):.1f}nM  // {code}.pdf (LIG)\n"
        index_general.append(line)
        if code in refined_codes:
            index_refined.append(line)
        if code in core_codes:
            core_lines.append(f"{code}  2.00  {year}  {pk:.2f}  Kd=1nM  1\n")

    header = "# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name\n"
    (root / "index" / "INDEX_general_PL_data.2020").write_text(header + "".join(index_general), encoding="utf-8")
    (root / "index" / "INDEX_refined_data.2020").write_text(header + "".join(index_refined), encoding="utf-8")
    (casf / "power_scoring" / "CoreSet.dat").write_text("#code  resl  year  logKa  Ka  target\n" + "".join(core_lines), encoding="utf-8")
    return {"root": root, "codes": codes, "core_codes": core_codes, "refined_codes": refined_codes}
