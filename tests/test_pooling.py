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


def test_mlp_scorer_also_equals_mean_pooling_at_init():
    g = torch.Generator().manual_seed(11)
    x = torch.randn(3, 9, 16, generator=g)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[2, 5:] = False
    pool = AttentionPool(16, scorer_hidden=32, temperature=0.5)
    assert torch.allclose(pool(x, mask), masked_mean(x, mask), atol=1e-6)


def test_nonpositive_temperature_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        AttentionPool(16, temperature=0.0)


def test_temperature_sharpens_the_weights():
    """Same scorer, lower temperature -> further from uniform."""

    torch.manual_seed(5)
    x = torch.randn(1, 12, 16)
    sharp, soft = AttentionPool(16, temperature=0.25), AttentionPool(16, temperature=1.0)
    w = torch.randn(1, 16) * 0.5
    for pool in (sharp, soft):
        with torch.no_grad():
            pool.score.weight.copy_(w)
    spread = lambda p: p(x).abs().std()
    assert spread(sharp) != spread(soft)


def test_pool_params_get_a_larger_learning_rate():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindformer"))
    from scripts.train import build_optimizer

    model = build(True)
    opt = build_optimizer(model, {"lr": 1e-4, "weight_decay": 0.0, "pool_lr_mult": 20.0})
    lrs = sorted(group["lr"] for group in opt.param_groups)
    assert lrs == [1e-4, 2e-3]
    n_pool = sum(1 for n, _ in model.named_parameters() if "_pool" in n)
    assert len(opt.param_groups[-1]["params"]) == n_pool


def test_optimizer_has_one_group_without_pooling():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindformer"))
    from scripts.train import build_optimizer

    opt = build_optimizer(build(False), {"lr": 1e-4, "weight_decay": 0.0, "pool_lr_mult": 20.0})
    assert len(opt.param_groups) == 1
