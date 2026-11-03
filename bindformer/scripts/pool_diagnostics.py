"""Report how far a trained AttentionPool is from plain mean pooling.

    python scripts/pool_diagnostics.py checkpoints_attn_s42/best.pt [--split casf2016]

Two numbers per pool, averaged over the split:

* `entropy/uniform` - attention entropy divided by log(n_tokens). 1.000 is exactly
  uniform, which *is* mean pooling, so a value at 1.000 means the layer never learned
  to prefer any token and a null result against the mean-pool baseline says nothing.
* `top x uniform` - the largest weight times the token count. 1.0 is uniform; 3.0
  means the top token takes three times its even share.

A first attempt at attention pooling sat at 0.999 / 1.2 after a full run, which is
why this check exists: run it before spending seeds on a comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from bindformer.data import PDBbindDataset, collate_complexes
from bindformer.models import BindFormer


def load_model(checkpoint: Path) -> tuple[BindFormer, dict]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ck["config"]
    model = BindFormer(
        protein_dim=config["features"]["protein_dim"],
        ligand_dim=config["features"]["ligand_dim"],
        pocket_radius_angstrom=config["features"]["pocket_radius_angstrom"],
        **config["model"],
    )
    model.load_state_dict(ck["model"])
    return model.eval(), config


def pool_stats(pool, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], list[float]]:
    logits = pool.score(tokens).squeeze(-1) / pool.temperature
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 2)
    weights = torch.softmax(logits, dim=-1)
    n = mask.sum(-1).float()
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(-1)
    return (entropy / n.log()).tolist(), (weights.max(-1).values * n).tolist()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Measure how far attention pooling is from uniform.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--split", default="casf2016")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args(argv)

    loader = DataLoader(
        PDBbindDataset(args.processed_dir, args.split),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_complexes,
    )
    mean = lambda xs: sum(xs) / max(1, len(xs))

    for path in args.checkpoints:
        model, _ = load_model(Path(path))
        if model.fusion.protein_pool is None:
            print(f"{path}: no attention pooling in this checkpoint")
            continue
        acc: dict[str, tuple[list[float], list[float]]] = {"protein": ([], []), "ligand": ([], [])}
        with torch.no_grad():
            for batch in loader:
                protein = model.protein_encoder(
                    batch["protein_x"], batch["protein_mask"], batch.get("protein_aa"), batch.get("protein_coords")
                )
                ligand = model.ligand_encoder(batch["ligand_x"], batch["ligand_mask"], batch.get("ligand_spd"))
                for name, pool, tokens, mask in (
                    ("protein", model.fusion.protein_pool, protein, batch["protein_mask"]),
                    ("ligand", model.fusion.ligand_pool, ligand, batch["ligand_mask"]),
                ):
                    ent, top = pool_stats(pool, tokens, mask)
                    acc[name][0].extend(ent)
                    acc[name][1].extend(top)
        parts = [
            f"{name} entropy/uniform={mean(ent):.3f} top x uniform={mean(top):.2f}"
            for name, (ent, top) in acc.items()
        ]
        print(f"{path}\n  " + "\n  ".join(parts))


if __name__ == "__main__":
    main()
