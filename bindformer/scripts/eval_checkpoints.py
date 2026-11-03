"""Score existing checkpoints on one split, without retraining.

    python scripts/eval_checkpoints.py --processed_dir data/processed_h \
        --split casf2016 --suffix _hfix checkpoints checkpoints_rbf_s42 ...

For each run directory, loads `best.pt` (falling back to `last.pt`), evaluates it on
the split from `--processed_dir`, and writes `predictions_<split>.csv` plus
`results.json` into `<run_dir><suffix>`. That layout is what scripts/compare.py takes
as one arm of a comparison, so corrected scores drop straight into the same tooling.

Use it when the data changed but the training did not — re-featurizing the test set
after a preprocessing fix, for instance.
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
from bindformer.training import Trainer
from bindformer.training.losses import BindFormerLoss


def resolve_checkpoint(run_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"no best.pt or last.pt in {run_dir}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on a split.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--split", default="casf2016")
    parser.add_argument("--suffix", default="_eval", help="Appended to each run dir for the output")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dataset = PDBbindDataset(args.processed_dir, args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"no cached complexes for split {args.split} in {args.processed_dir}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_complexes)
    meta = json.loads((Path(args.processed_dir) / "meta.json").read_text())
    print(f"{len(dataset)} complexes from {args.processed_dir}:{args.split} on {device}\n")

    for run in args.run_dirs:
        run_dir = Path(run)
        checkpoint = resolve_checkpoint(run_dir)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = state["config"]
        if config["features"]["protein_dim"] != meta["protein_dim"]:
            print(
                f"{run_dir.name}: SKIPPED - trained on {config['features']['protein_dim']}-dim protein "
                f"features, this cache has {meta['protein_dim']}"
            )
            continue
        model = BindFormer(
            protein_dim=config["features"]["protein_dim"],
            ligand_dim=config["features"]["ligand_dim"],
            pocket_radius_angstrom=config["features"]["pocket_radius_angstrom"],
            **config["model"],
        )
        model.load_state_dict(state["model"])
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.0), device, BindFormerLoss(), amp=False)
        out_dir = Path(f"{run}{args.suffix}")
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = trainer.evaluate(loader, out_dir / f"predictions_{args.split}.csv")
        (out_dir / "results.json").write_text(json.dumps({args.split: metrics}, indent=2), encoding="utf-8")
        print(
            f"{run_dir.name:<26} epoch {state.get('epoch', '?'):>3}  "
            f"pearson={metrics['pearson_r']:.4f} rmse={metrics['rmse']:.4f} sd={metrics['sd']:.4f}"
        )


if __name__ == "__main__":
    main()
