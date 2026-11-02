from __future__ import annotations

import torch

from bindformer.models.fusion import FusionHead, masked_mean
from bindformer.models.pooling import AttentionPool
from tests.test_rbf import make_batch

from bindformer.models import BindFormer


def test_uniform_attention_equals_mean_pooling():
    """Zero-init scorer -> uniform weights -> exactly masked_mean, padding included."""

    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 7, 16, generator=g)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[1, 4:] = False
    mask[2, 1:] = False
    pool = AttentionPool(16)
    assert torch.allclose(pool(x, mask), masked_mean(x, mask), atol=1e-6)


def test_pooling_handles_a_fully_masked_row():
    x = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0] = False
    out = AttentionPool(8)(x, mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.zeros(8))


def test_pooling_ignores_padded_values():
    """Changing a padded token must not move the pooled vector."""

    x = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 3:] = False
    pool = AttentionPool(8)
    torch.nn.init.normal_(pool.score.weight, 0.0, 1.0)
    before = pool(x, mask).clone()
    x[0, 4] += 50.0
    assert torch.allclose(before, pool(x, mask), atol=1e-6)


def build(attention_pool: bool, seed: int = 0) -> BindFormer:
    torch.manual_seed(seed)
    return BindFormer(protein_dim=32, hidden_dim=32, num_heads=4, protein_layers=2, ligand_layers=2, raa_layers=2, fusion_layers=1, dropout=0.0, attention_pool=attention_pool)


def test_model_matches_mean_pool_baseline_at_init():
    batch = make_batch()
    pooled, plain = build(True).eval(), build(False).eval()
    shared = {k: v for k, v in pooled.state_dict().items() if "_pool" not in k}
    missing, unexpected = plain.load_state_dict(shared, strict=False)
    assert not unexpected and not [k for k in missing if "_pool" not in k]
    with torch.no_grad():
        a, b = pooled(batch)["affinity"], plain(batch)["affinity"]
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_gradients_reach_both_pools():
    model = build(True)
    model(make_batch())["affinity"].sum().backward()
    for name in ("protein_pool", "ligand_pool"):
        grad = getattr(model.fusion, name).score.weight.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, f"{name} received no gradient"


def test_attention_pool_is_off_by_default():
    assert FusionHead(16, 1, 0.0).protein_pool is None
