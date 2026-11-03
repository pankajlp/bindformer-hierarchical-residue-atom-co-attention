from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from bindformer.data import PDBbindDataset, collate_complexes
from bindformer.models import BindFormer
from bindformer.training import Trainer
from bindformer.training.losses import BindFormerLoss


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_meta(config: dict) -> dict:
    """Take feature dimensions from the preprocessing run so model and caches always agree."""

    meta_path = Path(config["data"]["processed_dir"]) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config["features"]["protein_dim"] = meta["protein_dim"]
        config["features"]["ligand_dim"] = meta["ligand_dim"]
        config["features"]["pocket_radius_angstrom"] = meta.get("pocket_cutoff", config["features"]["pocket_radius_angstrom"])
    return config


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(default_device())
    return torch.device(name)


def build_model(config: dict) -> BindFormer:
    return BindFormer(
        protein_dim=config["features"]["protein_dim"],
        ligand_dim=config["features"]["ligand_dim"],
        pocket_radius_angstrom=config["features"]["pocket_radius_angstrom"],
        **config["model"],
    )


def make_loader(dataset, config: dict, shuffle: bool) -> DataLoader | None:
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_complexes,
        persistent_workers=config["data"]["num_workers"] > 0,
    )


def build_optimizer(model: BindFormer, train_cfg: dict) -> torch.optim.Optimizer:
    """AdamW, with the attention-pool scorer optionally on a larger learning rate.

    A zero-initialized scorer at the model's own LR stayed at uniform weights for a
    whole run, so `pool_lr_mult` lets it move while the rest of the model is unchanged.
    """

    lr, mult = train_cfg["lr"], train_cfg.get("pool_lr_mult", 1.0)
    pool = [p for name, p in model.named_parameters() if "_pool" in name]
    rest = [p for name, p in model.named_parameters() if "_pool" not in name]
    groups = [{"params": rest, "lr": lr}]
    if pool:
        groups.append({"params": pool, "lr": lr * mult})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=train_cfg["weight_decay"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate BindFormer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--processed_dir", help="Override data.processed_dir")
    parser.add_argument("--checkpoint_dir", help="Override training.checkpoint_dir")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--seed", type=int, help="Override seed")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.processed_dir:
        config["data"]["processed_dir"] = args.processed_dir
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    config = apply_meta(config)

    torch.manual_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    model = build_model(config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])

    data_cfg = config["data"]
    train_cfg = config["training"]
    processed_dir = data_cfg["processed_dir"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = build_optimizer(model, train_cfg)
    trainer = Trainer(
        model,
        optimizer,
        device,
        BindFormerLoss(**{k: v for k, v in config["loss"].items() if k != "pose_weight"}),
        amp=train_cfg["amp"],
        grad_clip_norm=train_cfg["grad_clip_norm"],
        warmup_epochs=train_cfg.get("warmup_epochs", 1),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        selection_metric=train_cfg.get("selection_metric", "rmse"),
        pose_weight=config.get("loss", {}).get("pose_weight", 0.0),
        decoy_kind=config.get("decoy", {}).get("kind", "rotate"),
        decoy_distance=config.get("decoy", {}).get("distance", 0.0),
        seed=config.get("seed", 42),
    )

    if args.mode == "train":
        train_ds = PDBbindDataset(processed_dir, data_cfg["train_split"])
        if len(train_ds) == 0:
            raise RuntimeError(f"No training caches found under {processed_dir}. Run scripts/preprocess.py first.")
        val_loader = make_loader(PDBbindDataset(processed_dir, data_cfg["val_split"]), config, shuffle=False)
        print(f"train={len(train_ds)} val={0 if val_loader is None else len(val_loader.dataset)} device={device}")
        trainer.fit(make_loader(train_ds, config, shuffle=True), val_loader, train_cfg["epochs"], checkpoint_dir, {"config": config})
        best = checkpoint_dir / "best.pt"
        final = best if best.exists() else checkpoint_dir / "last.pt"
        model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True)["model"])
        print(f"Loaded {final.name} for final evaluation")

    results = {}
    for split in (data_cfg["val_split"], data_cfg["test_split"]):
        loader = make_loader(PDBbindDataset(processed_dir, split), config, shuffle=False)
        if loader is not None:
            results[split] = trainer.evaluate(loader, checkpoint_dir / f"predictions_{split}.csv")
    (checkpoint_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
