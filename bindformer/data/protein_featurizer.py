from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bindformer.utils.binding_site import crop_pocket
from bindformer.utils.pdb_utils import AMINO_ACIDS, UNKNOWN_AA_INDEX, parse_protein_pdb

ESM2_DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
ONEHOT_DIM = UNKNOWN_AA_INDEX + 1


@dataclass
class ProteinFeatures:
    x: torch.Tensor  # [pocket_residues, feature_dim]
    coords: torch.Tensor  # [pocket_residues, 3] CA coordinates
    aa: torch.Tensor  # [pocket_residues] amino-acid index (20 = unknown)
    residue_ids: list[str]
    contacts: torch.Tensor  # [pocket_residues, ligand_atoms]
    num_residues_total: int


class OneHotEmbedder:
    """Residue-identity one-hot features; the no-download fallback for ESM-2."""

    name = "onehot"
    dim = ONEHOT_DIM

    def embed(self, sequence: str) -> np.ndarray:
        out = np.zeros((len(sequence), self.dim), dtype=np.float32)
        for i, aa in enumerate(sequence):
            out[i, AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else UNKNOWN_AA_INDEX] = 1.0
        return out


class Esm2Embedder:
    """Per-residue ESM-2 embeddings via Hugging Face `transformers`, cached on disk per sequence.

    Sequences longer than the model context are embedded in overlapping windows and
    the overlapping positions are averaged.
    """

    name = "esm2"

    def __init__(
        self,
        model_name: str = ESM2_DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        max_len: int = 1022,
        stride: int = 511,
    ):
        from transformers import AutoTokenizer, EsmModel

        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.dim = self.model.config.hidden_size
        self.max_len = max_len
        self.stride = stride
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, sequence: str) -> np.ndarray:
        cache_path = None
        if self.cache_dir:
            digest = hashlib.sha1(f"{self.model_name}:{sequence}".encode()).hexdigest()
            cache_path = self.cache_dir / f"{digest}.npy"
            if cache_path.exists():
                return np.load(cache_path).astype(np.float32)

        total = np.zeros((len(sequence), self.dim), dtype=np.float32)
        counts = np.zeros((len(sequence), 1), dtype=np.float32)
        starts = list(range(0, max(1, len(sequence) - self.max_len + self.stride), self.stride)) or [0]
        for start in starts:
            window = sequence[start : start + self.max_len]
            total[start : start + len(window)] += self._embed_window(window)
            counts[start : start + len(window)] += 1
            if start + self.max_len >= len(sequence):
                break
        emb = total / np.maximum(counts, 1)

        if cache_path is not None:
            np.save(cache_path, emb.astype(np.float16))
        return emb

    @torch.no_grad()
    def _embed_window(self, window: str) -> np.ndarray:
        tokens = self.tokenizer(window, return_tensors="pt").to(self.device)
        hidden = self.model(**tokens).last_hidden_state[0, 1 : len(window) + 1]
        return hidden.float().cpu().numpy()


class ProteinFeaturizer:
    """Pocket-level residue features from a PDB structure and a ligand pose.

    Embeddings are computed on each full observed chain (so ESM-2 sees sequence
    context), then sliced down to the pocket residues.
    """

    def __init__(self, embedder=None, pocket_cutoff: float = 10.0, contact_cutoff: float = 4.0):
        self.embedder = embedder or OneHotEmbedder()
        self.pocket_cutoff = pocket_cutoff
        self.contact_cutoff = contact_cutoff

    @property
    def feature_dim(self) -> int:
        return self.embedder.dim

    def featurize_pocket(self, pdb_path: str | Path, ligand_coords) -> ProteinFeatures:
        structure = parse_protein_pdb(pdb_path)
        if not structure.residues:
            raise ValueError(f"No amino-acid residues parsed from {pdb_path}")
        ligand_coords = np.asarray(ligand_coords, dtype=np.float32)
        pocket_idx, contacts = crop_pocket(structure, ligand_coords, self.pocket_cutoff, self.contact_cutoff)
        if len(pocket_idx) == 0:
            raise ValueError(f"No residues within {self.pocket_cutoff} A of the ligand in {pdb_path}")

        pocket_set = set(pocket_idx.tolist())
        residue_x: dict[int, np.ndarray] = {}
        for sequence, indices in structure.chain_sequences().values():
            if not pocket_set.intersection(indices):
                continue
            emb = self.embedder.embed(sequence)
            for pos, res_i in enumerate(indices):
                if res_i in pocket_set:
                    residue_x[res_i] = emb[pos]

        x = np.stack([residue_x[i] for i in pocket_idx]).astype(np.float32)
        coords = structure.residue_coords()[pocket_idx]
        aa = structure.aa_indices()[pocket_idx]
        return ProteinFeatures(
            x=torch.from_numpy(x),
            coords=torch.from_numpy(coords),
            aa=torch.from_numpy(aa),
            residue_ids=[structure.residues[i].key for i in pocket_idx],
            contacts=torch.from_numpy(contacts),
            num_residues_total=len(structure.residues),
        )


def build_embedder(kind: str, model_name: str = ESM2_DEFAULT_MODEL, device: str = "cpu", cache_dir=None):
    if kind == "onehot":
        return OneHotEmbedder()
    if kind == "esm2":
        return Esm2Embedder(model_name, device=device, cache_dir=cache_dir)
    raise ValueError(f"Unknown protein feature type: {kind}")
