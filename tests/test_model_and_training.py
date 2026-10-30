from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scipy import stats

from bindformer.data import PDBbindDataset, collate_complexes
from bindformer.models import BindFormer
from bindformer.models.ligand_encoder import LigandEncoder
from bindformer.scripts import preprocess, train
from bindformer.training.metrics import casf_sd, pearson_r, spearman_rho


def test_metrics_match_scipy():
    rng = np.random.default_rng(1)
    target = rng.normal(6, 2, size=200)
    pred = target + rng.normal(0, 1, size=200)
    pred[:10] = pred[0]  # ties for the rank correlation
    p, t = torch.tensor(pred), torch.tensor(target)
    assert pearson_r(p, t) == pytest.approx(stats.pearsonr(pred, target)[0], abs=1e-9)
    assert spearman_rho(p, t) == pytest.approx(stats.spearmanr(pred, target)[0], abs=1e-9)
    fit = stats.linregress(pred, target)
    resid = target - (fit.slope * pred + fit.intercept)
    assert casf_sd(p, t) == pytest.approx(np.sqrt((resid**2).sum() / (len(pred) - 1)), abs=1e-9)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_ligand_encoder_is_padding_invariant(mode):
    torch.manual_seed(0)
    enc = LigandEncoder(74, 32, 4, 2, 0.0)
    getattr(enc, mode)()
    torch.nn.init.normal_(enc.spd_bias.weight)
    x = torch.randn(1, 5, 74)
    spd = torch.randint(0, 9, (1, 5, 5))
    alone = enc(x, torch.ones(1, 5, dtype=torch.bool), spd)

    # Batch of 2 where our molecule is padded next to a larger one.
    x_pad = torch.cat([torch.cat([x, torch.randn(1, 3, 74)], dim=1), torch.randn(1, 8, 74)])
    spd_pad = torch.randint(0, 9, (2, 8, 8))
    spd_pad[0, :5, :5] = spd
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 8])
    with torch.no_grad():
        padded = enc(x_pad, mask, spd_pad)
    assert not padded.isnan().any()
    torch.testing.assert_close(alone.detach(), padded[:1, :5], atol=1e-5, rtol=1e-5)


def test_preprocess_train_eval_end_to_end(pdbbind_tree, tmp_path):
    out = tmp_path / "processed"
    preprocess.main([
        "--pdbbind_root", str(pdbbind_tree["root"]),
        "--out_dir", str(out),
        "--protein_features", "onehot",
        "--val_fraction", "0.25",
    ])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["protein_dim"] == 21 and meta["ligand_dim"] == 74
    assert meta["split_sizes"] == {"train": 10, "val": 3, "casf2016": 3}
    assert not (out / "failures.csv").exists()

    ds = PDBbindDataset(out, "train")
    assert len(ds) == 10
    batch = collate_complexes([ds[0], ds[1], ds[2]])
    assert batch["ligand_spd"].shape[1:] == batch["ligand_x"].shape[1:2] * 2
    assert batch["contacts"].shape == (3, batch["protein_x"].shape[1], batch["ligand_x"].shape[1])
    assert batch["contacts"].sum() > 0

    model = BindFormer(protein_dim=21, ligand_dim=74, hidden_dim=32, num_heads=4)
    outputs = model(batch)
    assert outputs["affinity"].shape == (3,)
    assert outputs["attention_mask"].any()

    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 0\ndevice: cpu\n"
        f"data: {{processed_dir: '{out.as_posix()}', cache_ext: .pt, train_split: train, val_split: val, test_split: casf2016, num_workers: 0}}\n"
        "features: {protein_dim: 1280, ligand_dim: 74, pocket_radius_angstrom: 10.0}\n"
        "model: {hidden_dim: 32, num_heads: 4, dropout: 0.0, protein_layers: 1, ligand_layers: 1, raa_layers: 1, fusion_layers: 1}\n"
        "training: {batch_size: 4, epochs: 3, lr: 0.001, weight_decay: 0.0, warmup_epochs: 1, early_stopping_patience: 5, selection_metric: rmse, amp: true, grad_clip_norm: 5.0, checkpoint_dir: unused}\n"
        "loss: {affinity_weight: 1.0, contact_weight: 0.2, contact_pos_weight: 10.0}\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "ckpt"
    train.main(["--config", str(config), "--checkpoint_dir", str(ckpt)])
    assert (ckpt / "best.pt").exists() and (ckpt / "last.pt").exists()
    assert len((ckpt / "log.jsonl").read_text().splitlines()) == 3
    results = json.loads((ckpt / "results.json").read_text())
    assert results["casf2016"]["n"] == 3 and results["val"]["n"] == 3
    assert (ckpt / "predictions_casf2016.csv").exists()

    eval_dir = tmp_path / "eval"
    train.main(["--config", str(config), "--mode", "eval", "--checkpoint", str(ckpt / "best.pt"), "--checkpoint_dir", str(eval_dir)])
    eval_results = json.loads((eval_dir / "results.json").read_text())
    assert eval_results["casf2016"]["rmse"] == pytest.approx(results["casf2016"]["rmse"], rel=1e-5)
