from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from bindformer.data.pdbbind_index import load_splits


class PDBbindDataset(Dataset):
    """Dataset over preprocessed PDBbind `.pt` complex caches.

    With `splits.json` present (written by `scripts/preprocess.py`), the split's
    complex codes are loaded from `processed_dir/complexes/<code>.pt`. Otherwise
    falls back to globbing `processed_dir/<split>/*.pt`.

    Required cache keys are `protein_x`, `protein_coords`, `ligand_x`,
    `ligand_coords`, and `affinity`. Optional keys include `ligand_spd`,
    `protein_aa`, and `contacts`.
    """

    def __init__(self, processed_dir: str | Path, split: str = "train", cache_ext: str = ".pt"):
        self.processed_dir = Path(processed_dir)
        self.split = split
        splits_path = self.processed_dir / "splits.json"
        if splits_path.exists():
            codes = load_splits(splits_path).get(split, [])
            complex_dir = self.processed_dir / "complexes"
            self.files = [complex_dir / f"{code}{cache_ext}" for code in codes]
            self.files = [f for f in self.files if f.exists()]
        else:
            split_dir = self.processed_dir / split
            root = split_dir if split_dir.exists() else self.processed_dir
            self.files = sorted(root.glob(f"*{cache_ext}"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = torch.load(self.files[index], map_location="cpu", weights_only=True)
        item.setdefault("complex_id", self.files[index].stem)
        item["protein_x"] = item["protein_x"].float()
        return item


def collate_complexes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-sized pockets and ligands into dense tensors."""

    protein_n = max(item["protein_x"].shape[0] for item in batch)
    ligand_n = max(item["ligand_x"].shape[0] for item in batch)
    p_dim = batch[0]["protein_x"].shape[-1]
    l_dim = batch[0]["ligand_x"].shape[-1]
    bsz = len(batch)

    out = {
        "protein_x": torch.zeros(bsz, protein_n, p_dim),
        "ligand_x": torch.zeros(bsz, ligand_n, l_dim),
        "protein_coords": torch.zeros(bsz, protein_n, 3),
        "ligand_coords": torch.zeros(bsz, ligand_n, 3),
        "protein_mask": torch.zeros(bsz, protein_n, dtype=torch.bool),
        "ligand_mask": torch.zeros(bsz, ligand_n, dtype=torch.bool),
        "affinity": torch.tensor([float(item["affinity"]) for item in batch], dtype=torch.float32),
        "complex_id": [item.get("complex_id", str(i)) for i, item in enumerate(batch)],
    }
    has_spd = all("ligand_spd" in item for item in batch)
    has_aa = all("protein_aa" in item for item in batch)
    has_contacts = all("contacts" in item for item in batch)
    if has_spd:
        out["ligand_spd"] = torch.zeros(bsz, ligand_n, ligand_n, dtype=torch.long)
    if has_aa:
        out["protein_aa"] = torch.zeros(bsz, protein_n, dtype=torch.long)
    if has_contacts:
        out["contacts"] = torch.zeros(bsz, protein_n, ligand_n)

    for i, item in enumerate(batch):
        pn = item["protein_x"].shape[0]
        ln = item["ligand_x"].shape[0]
        out["protein_x"][i, :pn] = item["protein_x"]
        out["ligand_x"][i, :ln] = item["ligand_x"]
        out["protein_coords"][i, :pn] = item["protein_coords"]
        out["ligand_coords"][i, :ln] = item["ligand_coords"]
        out["protein_mask"][i, :pn] = True
        out["ligand_mask"][i, :ln] = True
        if has_spd:
            out["ligand_spd"][i, :ln, :ln] = item["ligand_spd"].long()
        if has_aa:
            out["protein_aa"][i, :pn] = item["protein_aa"]
        if has_contacts:
            out["contacts"][i, :pn, :ln] = item["contacts"]
    return out
