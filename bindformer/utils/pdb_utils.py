from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
# Common modified residues mapped to their parent amino acid.
MODIFIED_TO_PARENT = {
    "MSE": "MET", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS",
    "CME": "CYS", "KCX": "LYS", "LLP": "LYS", "HYP": "PRO", "MLY": "LYS",
    "CSD": "CYS", "OCS": "CYS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_AA_INDEX = len(AMINO_ACIDS)


@dataclass
class Residue:
    chain: str
    resseq: str
    icode: str
    resname: str
    one_letter: str
    atom_names: list[str] = field(default_factory=list)
    atom_coords: list[list[float]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.chain}:{self.resseq}{self.icode}:{self.resname}"

    @property
    def representative_coord(self) -> np.ndarray:
        """CA coordinate, or the residue heavy-atom centroid when CA is missing."""

        if "CA" in self.atom_names:
            return np.asarray(self.atom_coords[self.atom_names.index("CA")], dtype=np.float32)
        return np.asarray(self.atom_coords, dtype=np.float32).mean(axis=0)


@dataclass
class ProteinStructure:
    residues: list[Residue]

    def chain_sequences(self) -> dict[str, tuple[str, list[int]]]:
        """Map chain id -> (observed sequence, residue indices in `self.residues`)."""

        chains: dict[str, tuple[list[str], list[int]]] = {}
        for i, res in enumerate(self.residues):
            seq, idx = chains.setdefault(res.chain, ([], []))
            seq.append(res.one_letter)
            idx.append(i)
        return {chain: ("".join(seq), idx) for chain, (seq, idx) in chains.items()}

    def residue_coords(self) -> np.ndarray:
        if not self.residues:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([r.representative_coord for r in self.residues]).astype(np.float32)

    def aa_indices(self) -> np.ndarray:
        return np.asarray([AA_TO_INDEX.get(r.one_letter, UNKNOWN_AA_INDEX) for r in self.residues], dtype=np.int64)


def parse_protein_pdb(pdb_path: str | Path) -> ProteinStructure:
    """Parse amino-acid residues from a PDB file.

    Keeps the first model only, standard plus common modified amino acids, heavy
    atoms only, and the first alternate location. Waters, ions and other HETATM
    groups are dropped.
    """

    residues: dict[tuple[str, str, str], Residue] = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            record = line[:6]
            if record not in ("ATOM  ", "HETATM") or len(line) < 54:
                continue
            resname = line[17:20].strip()
            parent = resname if resname in THREE_TO_ONE else MODIFIED_TO_PARENT.get(resname)
            if parent is None:
                continue
            altloc = line[16]
            if altloc not in (" ", "A", "1"):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if (element or atom_name.lstrip("0123456789")[:1]).upper() in ("H", "D"):
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            res = residues.get(key)
            if res is None:
                res = Residue(chain, resseq, icode, resname, THREE_TO_ONE[parent])
                residues[key] = res
            if atom_name in res.atom_names:
                continue
            res.atom_names.append(atom_name)
            res.atom_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return ProteinStructure([r for r in residues.values() if r.atom_coords])


def extract_residue_coordinates(pdb_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Extract one coordinate per amino-acid residue, preferring CA atoms."""

    structure = parse_protein_pdb(pdb_path)
    return [r.key for r in structure.residues], structure.residue_coords()


def pairwise_distances(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
