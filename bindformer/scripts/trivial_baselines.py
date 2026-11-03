"""How much of the affinity label is predictable without modelling binding at all?

PDBbind pK correlates with ligand size, so part of any model's scoring power comes
from counting atoms rather than from protein-ligand interaction. This fits
deliberately trivial predictors on the train split and scores them on CASF-2016,
giving the floor any real model has to beat.

    python scripts/trivial_baselines.py --processed_dir data/processed

Predictors, in increasing order of cheating:

* `heavy_atoms`     - ligand heavy-atom count alone, no fitting.
* `atom_counts`     - ridge on per-element counts (C, N, O, S, halogens, ...).
* `ligand_features` - ridge on the mean and sum of the 74-dim atom features.
* `pocket_size`     - number of pocket residues alone, which uses the true pose only
                      through the 10 A crop and so measures pocket-size leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sklearn.linear_model import RidgeCV


def load_split(
    processed_dir: Path, split: str, ids: list[str], symbol_index: dict[str, int] | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """`symbol_index` is shared across splits so element-count columns line up."""

    heavy, pocket, feat_mean, feat_sum, pk = [], [], [], [], []
    if symbol_index is None:
        symbol_index = {}
    rows = []
    for code in ids:
        item = torch.load(processed_dir / "complexes" / f"{code}.pt", weights_only=False)
        symbols = item["atom_symbols"]
        counts: dict[str, int] = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1
            symbol_index.setdefault(s, len(symbol_index))  # unseen elements extend the shared vocabulary
        x = item["ligand_x"].float()
        rows.append(counts)
        heavy.append(len(symbols))
        pocket.append(item["protein_x"].shape[0])
        feat_mean.append(x.mean(0).numpy())
        feat_sum.append(x.sum(0).numpy())
        pk.append(item["affinity"])
    elements = np.zeros((len(rows), len(symbol_index)), dtype=np.float32)
    for i, counts in enumerate(rows):
        for s, c in counts.items():
            elements[i, symbol_index[s]] = c
    return {
        "heavy_atoms": np.array(heavy, dtype=np.float32)[:, None],
        "pocket_size": np.array(pocket, dtype=np.float32)[:, None],
        "atom_counts": elements,
        "ligand_features": np.concatenate([np.stack(feat_mean), np.stack(feat_sum)], axis=1),
    }, np.array(pk, dtype=np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Trivial no-binding baselines for CASF-2016.")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="casf2016")
    parser.add_argument("--train_limit", type=int, default=4000, help="Subsample train for speed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    splits = json.loads((processed / "splits.json").read_text())
    splits = splits.get("splits", splits)
    rng = np.random.default_rng(args.seed)
    train_ids = list(splits[args.train_split])
    if args.train_limit and len(train_ids) > args.train_limit:
        train_ids = [train_ids[i] for i in rng.choice(len(train_ids), args.train_limit, replace=False)]
    test_ids = list(splits[args.test_split])

    print(f"fitting on {len(train_ids)} train complexes, scoring {len(test_ids)} {args.test_split}\n")
    symbols: dict[str, int] = {}
    Xtr, ytr = load_split(processed, args.train_split, train_ids, symbols)
    Xte, yte = load_split(processed, args.test_split, test_ids, symbols)
    # Train may have elements CASF lacks, and vice versa; pad both to the shared width.
    width = len(symbols)
    for X in (Xtr, Xte):
        got = X["atom_counts"].shape[1]
        if got < width:
            X["atom_counts"] = np.pad(X["atom_counts"], ((0, 0), (0, width - got)))

    print(f"{'predictor':<18} {'CASF Pearson':>13} {'RMSE':>7}")
    for name in ("heavy_atoms", "pocket_size", "atom_counts", "ligand_features"):
        a, b = Xtr[name], Xte[name]
        if a.shape[1] == 1:
            # Single feature: a monotone fit cannot change Pearson, so use it directly.
            r = pearson(b[:, 0], yte)
            scale = np.polyfit(a[:, 0], ytr, 1)
            pred = np.polyval(scale, b[:, 0])
        else:
            model = RidgeCV(alphas=np.logspace(-3, 4, 25)).fit(a, ytr)
            pred = model.predict(b)
            r = pearson(pred, yte)
        print(f"{name:<18} {r:>13.3f} {rmse(pred, yte):>7.3f}")

    print(f"\nfor reference: BindFormer baseline is 0.748 +/- 0.011 Pearson, RMSE 1.44")
    print(f"label spread on {args.test_split}: sd {yte.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
