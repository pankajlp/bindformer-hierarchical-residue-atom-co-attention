"""Does the model actually use the ligand's pose, or only what is in the pocket?

The pocket crop and the co-attention mask are both built from the ligand's true
coordinates, so a model can score well while ignoring geometry: pocket composition
plus ligand identity may be enough. This perturbs the pose at evaluation time and
watches what happens to CASF scoring power.

    python scripts/pose_sensitivity.py --processed_dir data/processed_h \
        checkpoints checkpoints_esm_s43 checkpoints_esm_s44

Protein and ligand tokens are left exactly as cached; only `ligand_coords` moves, so
the only thing that changes is the geometry entering the distance mask (and the RBF
bias, where a model has one). Perturbations:

* `rotate`      - random rotation about the ligand centroid. The ligand stays in the
                  pocket and keeps its shape, but every contact is rearranged.
* `shift <d>`   - translation by d Angstrom in a random direction.
* `scramble`    - independent random rotation *and* a 10 A shift: no real contacts left.

What the numbers mean. A scorer that reads the pose should lose Pearson as the pose
degrades, and `scramble` should approach the no-interaction limit. A flat row means
the pose is decoration and the score comes from the pocket and the ligand alone.
`mean |delta pred|` is in pK units, so it is directly comparable to the RMSE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from bindformer.data import PDBbindDataset, collate_complexes
from bindformer.models import BindFormer


def random_rotations(n: int, generator: torch.Generator) -> torch.Tensor:
    """`n` uniformly random 3x3 rotation matrices, via QR of a Gaussian matrix."""

    a = torch.randn(n, 3, 3, generator=generator)
    q, r = torch.linalg.qr(a)
    # Fix the sign so these are rotations (det +1), not reflections.
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q = q * sign
    flip = torch.where(torch.det(q) < 0, -1.0, 1.0).view(n, 1, 1)
    return q * flip


def perturb(coords: torch.Tensor, mask: torch.Tensor, kind: str, distance: float, generator: torch.Generator):
    """Move each ligand in the batch. `coords` is [B, A, 3]; padding rows are ignored."""

    out = coords.clone()
    bsz = coords.shape[0]
    valid = mask.unsqueeze(-1)
    counts = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
    centroid = (coords * valid).sum(1, keepdim=True) / counts

    if kind in ("rotate", "scramble"):
        rot = random_rotations(bsz, generator).to(coords.dtype)
        out = torch.einsum("bij,baj->bai", rot, out - centroid) + centroid
    if kind in ("shift", "scramble"):
        direction = torch.randn(bsz, 1, 3, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        out = out + direction.to(coords.dtype) * distance
    return torch.where(valid, out, coords)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def score(model, loader, device, kind: str, distance: float, seed: int, csv_path: Path | None = None):
    generator = torch.Generator().manual_seed(seed)
    preds, targets, base, ids = [], [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if kind != "none":
            moved = perturb(
                batch["ligand_coords"].cpu(), batch["ligand_mask"].cpu(), kind, distance, generator
            ).to(device)
            reference = model(batch)["affinity"].float().cpu()
            base.append(reference)
            batch = {**batch, "ligand_coords": moved}
        out = model(batch)
        preds.append(out["affinity"].float().cpu())
        targets.append(batch["affinity"].float().cpu())
        ids.extend(batch["complex_id"])
    pred = torch.cat(preds).numpy()
    target = torch.cat(targets).numpy()
    shift = float((torch.cat(base) - torch.cat(preds)).abs().mean()) if base else 0.0
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = _csv.writer(handle)
            writer.writerow(["complex_id", "pred_pk", "true_pk"])
            writer.writerows(zip(ids, pred.tolist(), target.tolist()))
    return pearson(pred, target), rmse, shift


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Measure how much the model's score depends on the pose.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--processed_dir", default="data/processed_h")
    parser.add_argument("--split", default="casf2016")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--write_predictions",
        help="Directory for per-condition predictions, laid out for scripts/compare.py",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    loader = DataLoader(
        PDBbindDataset(args.processed_dir, args.split),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_complexes,
    )
    conditions = [
        ("true pose", "none", 0.0),
        ("rotated in place", "rotate", 0.0),
        ("shifted 2 A", "shift", 2.0),
        ("shifted 5 A", "shift", 5.0),
        ("shifted 10 A", "shift", 10.0),
        ("shifted 25 A", "shift", 25.0),
        ("scrambled", "scramble", 10.0),
    ]
    rows: dict[str, list[tuple[float, float, float]]] = {name: [] for name, _, _ in conditions}

    for run in args.run_dirs:
        state = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
        config = state["config"]
        model = BindFormer(
            protein_dim=config["features"]["protein_dim"],
            ligand_dim=config["features"]["ligand_dim"],
            pocket_radius_angstrom=config["features"]["pocket_radius_angstrom"],
            **config["model"],
        )
        model.load_state_dict(state["model"])
        model = model.to(device).eval()
        for name, kind, distance in conditions:
            csv_path = None
            if args.write_predictions:
                slug = name.replace(" ", "_")
                csv_path = Path(args.write_predictions) / f"{slug}__{Path(run).name}" / f"predictions_{args.split}.csv"
            rows[name].append(score(model, loader, device, kind, distance, args.seed, csv_path))
        print(f"scored {run}")

    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n{len(args.run_dirs)} model(s), {args.split}, mean over runs\n")
    print(f"{'pose':<18} {'Pearson':>8} {'RMSE':>7} {'mean |d pred|':>14}")
    for name, _, _ in conditions:
        rs, rmses, shifts = zip(*rows[name])
        print(f"{name:<18} {mean(rs):>8.3f} {mean(rmses):>7.3f} {mean(shifts):>14.3f}")
    print("\nA flat Pearson column means the score does not depend on where the ligand is.")


if __name__ == "__main__":
    main()
