from __future__ import annotations

import pytest
import torch

from bindformer.models import BindFormer
from bindformer.models.rbf import RadialBasisBias


def make_batch(bsz: int = 3, n_res: int = 9, n_atom: int = 6, protein_dim: int = 32, ligand_dim: int = 74):
    g = torch.Generator().manual_seed(7)
    protein_mask = torch.ones(bsz, n_res, dtype=torch.bool)
    ligand_mask = torch.ones(bsz, n_atom, dtype=torch.bool)
    protein_mask[1, -3:] = False  # ragged batch: padding must not leak
    ligand_mask[2, -2:] = False
    return {
        "protein_x": torch.randn(bsz, n_res, protein_dim, generator=g),
        "protein_coords": torch.randn(bsz, n_res, 3, generator=g) * 6.0,
        "protein_aa": torch.randint(0, 20, (bsz, n_res), generator=g),
        "protein_mask": protein_mask,
        "ligand_x": torch.randn(bsz, n_atom, ligand_dim, generator=g),
        "ligand_coords": torch.randn(bsz, n_atom, 3, generator=g) * 2.0,
        "ligand_spd": torch.randint(0, 5, (bsz, n_atom, n_atom), generator=g),
        "ligand_mask": ligand_mask,
    }


def build(rbf_kernels: int, seed: int = 0) -> BindFormer:
    torch.manual_seed(seed)
    return BindFormer(protein_dim=32, hidden_dim=32, num_heads=4, protein_layers=2, ligand_layers=2, raa_layers=2, fusion_layers=1, dropout=0.0, rbf_kernels=rbf_kernels)


def test_bias_is_zero_at_init():
    bias = RadialBasisBias(4, 16, 20.0)(torch.rand(2, 5, 7) * 30)
    assert torch.equal(bias, torch.zeros_like(bias))


def test_rbf_model_matches_baseline_at_init():
    """Zero-init makes the RBF model identical to the plain one, so any gain is learned."""

    batch = make_batch()
    with_rbf, without = build(16).eval(), build(0).eval()
    shared = {k: v for k, v in with_rbf.state_dict().items() if "rbf" not in k}
    missing, unexpected = without.load_state_dict(shared, strict=False)
    assert not unexpected and not [k for k in missing if "rbf" not in k]
    with torch.no_grad():
        a, b = with_rbf(batch)["affinity"], without(batch)["affinity"]
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_bias_changes_predictions_once_trained():
    batch = make_batch()
    model = build(16).eval()
    with torch.no_grad():
        before = model(batch)["affinity"].clone()
        for mod in (model.protein_encoder.rbf, model.raa.rbf):
            torch.nn.init.normal_(mod.proj.weight, 0.0, 0.5)
        after = model(batch)["affinity"]
    assert not torch.allclose(before, after, atol=1e-4)


def test_gradients_reach_both_bias_modules():
    model = build(16)
    out = model(make_batch())
    out["affinity"].sum().backward()
    for name, mod in (("protein", model.protein_encoder.rbf), ("raa", model.raa.rbf)):
        grad = mod.proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"{name} bias received no gradient"


def test_outputs_stay_finite_with_padding():
    model = build(16).eval()
    for mod in (model.protein_encoder.rbf, model.raa.rbf):
        torch.nn.init.normal_(mod.proj.weight, 0.0, 1.0)
    with torch.no_grad():
        out = model(make_batch())
    assert torch.isfinite(out["affinity"]).all()
    assert torch.isfinite(out["contact_logits"]).all()


def test_rbf_is_off_by_default():
    model = BindFormer(protein_dim=32, hidden_dim=32, num_heads=4)
    assert model.protein_encoder.rbf is None and model.raa.rbf is None


def test_invalid_kernel_count_is_rejected():
    with pytest.raises(ValueError):
        RadialBasisBias(4, 0)
