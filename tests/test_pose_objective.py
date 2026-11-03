"""Decoy poses, the pose head, and the discrimination loss.

A decoy must differ from the native pose in geometry *only* — same protein, same atom
features, same bond graph — or the task could be solved without reading the pose.
"""

from __future__ import annotations

import pytest
import torch

from bindformer.models import BindFormer
from bindformer.training.losses import pose_discrimination_loss
from bindformer.utils.poses import centroids, decoy_batch, perturb, random_rotations
from tests.test_rbf import make_batch


def build(pose_head: bool, seed: int = 0) -> BindFormer:
    torch.manual_seed(seed)
    return BindFormer(protein_dim=32, hidden_dim=32, num_heads=4, protein_layers=2, ligand_layers=2, raa_layers=2, fusion_layers=1, dropout=0.0, pose_head=pose_head)


def test_random_rotations_are_rotations():
    r = random_rotations(32, torch.Generator().manual_seed(0))
    assert torch.allclose(r @ r.transpose(-1, -2), torch.eye(3).expand(32, 3, 3), atol=1e-5)
    assert (torch.det(r) > 0.99).all()


def test_rotation_keeps_the_ligand_where_it_was():
    batch = make_batch()
    coords, mask = batch["ligand_coords"], batch["ligand_mask"]
    moved = perturb(coords, mask, "rotate", generator=torch.Generator().manual_seed(1))
    assert torch.allclose(centroids(coords, mask), centroids(moved, mask), atol=1e-4)
    for i in range(coords.shape[0]):
        keep = mask[i]
        before = torch.cdist(coords[i][keep], coords[i][keep])
        after = torch.cdist(moved[i][keep], moved[i][keep])
        assert torch.allclose(before, after, atol=1e-4), "rotation must not deform the ligand"


def test_shift_moves_by_the_requested_distance():
    batch = make_batch()
    coords, mask = batch["ligand_coords"], batch["ligand_mask"]
    moved = perturb(coords, mask, "shift", 7.0, torch.Generator().manual_seed(2))
    delta = (centroids(moved, mask) - centroids(coords, mask)).norm(dim=-1).flatten()
    assert torch.allclose(delta, torch.full_like(delta, 7.0), atol=1e-3)


def test_padded_atoms_are_untouched():
    batch = make_batch()
    coords, mask = batch["ligand_coords"], batch["ligand_mask"]
    moved = perturb(coords, mask, "scramble", 5.0, torch.Generator().manual_seed(3))
    assert torch.equal(moved[~mask], coords[~mask])


def test_decoy_changes_only_the_coordinates():
    batch = make_batch()
    decoy = decoy_batch(batch, "rotate", generator=torch.Generator().manual_seed(4))
    assert not torch.allclose(decoy["ligand_coords"], batch["ligand_coords"])
    for key in ("ligand_x", "protein_x", "protein_coords", "ligand_spd", "protein_mask", "ligand_mask"):
        assert torch.equal(decoy[key], batch[key]), f"{key} must be identical in a decoy"


def test_decoy_drops_a_stale_attention_mask():
    """Keeping the native mask would hand the decoy the true contacts."""

    batch = dict(make_batch())
    batch["attention_mask"] = torch.ones(3, 9, 6, dtype=torch.bool)
    decoy = decoy_batch(batch, "rotate", generator=torch.Generator().manual_seed(5))
    assert "attention_mask" not in decoy


def test_decoy_actually_changes_the_contact_pattern():
    batch = make_batch()
    decoy = decoy_batch(batch, "rotate", generator=torch.Generator().manual_seed(6))
    before = torch.cdist(batch["protein_coords"], batch["ligand_coords"]) <= 10.0
    after = torch.cdist(decoy["protein_coords"], decoy["ligand_coords"]) <= 10.0
    assert not torch.equal(before, after), "a decoy the mask cannot see is not a negative"


def test_unknown_perturbation_is_rejected():
    batch = make_batch()
    with pytest.raises(ValueError):
        perturb(batch["ligand_coords"], batch["ligand_mask"], "wiggle", 1.0)


def test_pose_head_present_only_when_requested():
    assert build(False).fusion.pose is None
    assert build(True).fusion.pose is not None
    assert "pose_logit" not in build(False)(make_batch())
    assert "pose_logit" in build(True)(make_batch())


def test_discrimination_loss_rewards_separating_native_from_decoy():
    confident = pose_discrimination_loss(torch.full((8,), 5.0), torch.full((8,), -5.0))
    confused = pose_discrimination_loss(torch.zeros(8), torch.zeros(8))
    backwards = pose_discrimination_loss(torch.full((8,), -5.0), torch.full((8,), 5.0))
    assert confident < confused < backwards
    assert confused == pytest.approx(0.6931, abs=1e-3)  # log 2


def test_pose_loss_reaches_the_encoder_not_just_the_head():
    model = build(True)
    native = model(make_batch())
    decoy = model(decoy_batch(make_batch(), "rotate", generator=torch.Generator().manual_seed(7)))
    pose_discrimination_loss(native["pose_logit"], decoy["pose_logit"]).backward()
    grad = model.protein_encoder.input_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "geometry signal must reach the encoder"
