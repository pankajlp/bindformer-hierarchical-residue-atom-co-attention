"""Move a ligand without changing what it is.

Shared by `scripts/pose_sensitivity.py`, which perturbs poses to ask whether a trained
model uses them, and by training, which needs decoy poses as negatives. Only
coordinates change: atom features, the bond graph and the protein are untouched, so a
decoy differs from the native pose in geometry alone.
"""

from __future__ import annotations

import torch

KINDS = ("none", "rotate", "shift", "scramble")


def random_rotations(n: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """`n` uniformly random 3x3 rotation matrices, via QR of a Gaussian matrix."""

    a = torch.randn(n, 3, 3, generator=generator, device=device)
    q, r = torch.linalg.qr(a)
    # Fix the sign so these come out as rotations (det +1), not reflections.
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q = q * sign
    flip = torch.where(torch.det(q) < 0, -1.0, 1.0).view(n, 1, 1)
    return q * flip


def centroids(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    counts = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
    return (coords * valid).sum(1, keepdim=True) / counts


def perturb(
    coords: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
    distance: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Move each ligand in a batch. `coords` is [B, A, 3]; padded rows are left alone.

    `rotate` spins the ligand about its own centroid, so it stays in the pocket and
    keeps its shape and its distance to the protein - only which atom faces which
    residue changes. `shift` translates it `distance` Angstrom in a random direction.
    `scramble` does both.
    """

    if kind not in KINDS:
        raise ValueError(f"unknown perturbation {kind!r}, expected one of {KINDS}")
    if kind == "none":
        return coords
    # A generator is tied to one device, and a CPU generator is what keeps a run
    # reproducible across devices, so draw on its device and move the result back.
    if generator is not None and generator.device != coords.device:
        moved = perturb(coords.to(generator.device), mask.to(generator.device), kind, distance, generator)
        return moved.to(coords.device)
    out = coords
    valid = mask.unsqueeze(-1)
    if kind in ("rotate", "scramble"):
        centroid = centroids(coords, mask)
        rot = random_rotations(coords.shape[0], generator, coords.device).to(coords.dtype)
        out = torch.einsum("bij,baj->bai", rot, out - centroid) + centroid
    if kind in ("shift", "scramble"):
        direction = torch.randn(coords.shape[0], 1, 3, generator=generator, device=coords.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        out = out + direction.to(coords.dtype) * distance
    return torch.where(valid, out, coords)


def decoy_batch(
    batch: dict,
    kind: str = "rotate",
    distance: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict:
    """A shallow copy of `batch` whose ligand sits in a different pose.

    `attention_mask` is dropped if present so the model rebuilds it from the moved
    coordinates; leaving a stale mask in place would hand the decoy the native pose's
    contacts and make the task trivially unlearnable.
    """

    moved = perturb(batch["ligand_coords"], batch["ligand_mask"], kind, distance, generator)
    out = {key: value for key, value in batch.items() if key != "attention_mask"}
    out["ligand_coords"] = moved
    return out
