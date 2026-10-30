"""Featurize PDBbind complexes into per-complex `.pt` caches plus `splits.json` and `meta.json`.

Expected input (any nesting under --pdbbind_root):
    <code>/<code>_protein.pdb, <code>/<code>_ligand.sdf (+ optional .mol2)
    index/INDEX_general_PL_data.2020, index/INDEX_refined_data.2020
CASF-2016 complexes are found the same way; pass --coreset pointing at CASF-2016
(or its `power_scoring/CoreSet.dat`) to define the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from bindformer.data.ligand_featurizer import ATOM_FEATURE_DIM, LigandFeaturizer
from bindformer.data.pdbbind_index import (
    SplitConfig,
    find_complex_dirs,
    load_core_set,
    load_pdbbind_index,
    make_splits,
    save_splits,
)
from bindformer.data.protein_featurizer import ESM2_DEFAULT_MODEL, ProteinFeaturizer, build_embedder


def featurize_complex(code: str, complex_dir: Path, pk: float, ligand_featurizer, protein_featurizer) -> dict:
    ligand = ligand_featurizer.featurize_sdf(complex_dir / f"{code}_ligand.sdf")
    protein = protein_featurizer.featurize_pocket(complex_dir / f"{code}_protein.pdb", ligand.coords.numpy())
    return {
        "complex_id": code,
        "affinity": float(pk),
        "protein_x": protein.x.half() if protein.x.shape[-1] > 64 else protein.x,
        "protein_coords": protein.coords,
        "protein_aa": protein.aa,
        "residue_ids": protein.residue_ids,
        "num_residues_total": protein.num_residues_total,
        "ligand_x": ligand.x,
        "ligand_coords": ligand.coords,
        "ligand_spd": ligand.spd,
        "ligand_edge_index": ligand.edge_index,
        "ligand_edge_attr": ligand.edge_attr,
        "atom_symbols": ligand.atom_symbols,
        "ligand_source": ligand.source,
        "contacts": protein.contacts,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Featurize PDBbind complexes for BindFormer.")
    parser.add_argument("--pdbbind_root", "--data_dir", dest="pdbbind_root", default="data/raw")
    parser.add_argument("--coreset", default=None, help="CASF-2016 dir or CoreSet.dat (default: search pdbbind_root)")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--protein_features", choices=["esm2", "onehot"], default="esm2")
    parser.add_argument("--esm_model", default=ESM2_DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pocket_cutoff", type=float, default=10.0)
    parser.add_argument("--contact_cutoff", type=float, default=4.0)
    parser.add_argument("--split_strategy", choices=["random", "time"], default="random")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_min_year", type=int, default=2019)
    parser.add_argument("--exact_only", action="store_true", help="Drop affinities reported with <, >, ~")
    parser.add_argument("--refined_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Featurize at most N complexes (debugging)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # PDBbind SDF sanitization errors are expected; fallbacks handle them
    root = Path(args.pdbbind_root)
    out_dir = Path(args.out_dir)
    complex_out = out_dir / "complexes"
    complex_out.mkdir(parents=True, exist_ok=True)

    entries = load_pdbbind_index(root)
    core = load_core_set(args.coreset if args.coreset else root)
    if not core:
        print("WARNING: no CASF-2016 CoreSet.dat found; the casf2016 split will be empty.")
    # Core-set affinities fill in complexes missing from the general/refined index.
    for code, entry in core.items():
        entries.setdefault(code, entry)

    complex_dirs = find_complex_dirs(root)
    if args.coreset and Path(args.coreset).is_dir():
        for code, path in find_complex_dirs(args.coreset).items():
            complex_dirs.setdefault(code, path)
    todo = sorted(code for code in complex_dirs if code in entries)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(entries)} index entries, {len(core)} core-set codes, {len(complex_dirs)} complex dirs, {len(todo)} to featurize")

    embedder = build_embedder(args.protein_features, args.esm_model, args.device, cache_dir=out_dir / "esm_cache")
    ligand_featurizer = LigandFeaturizer()
    protein_featurizer = ProteinFeaturizer(embedder, args.pocket_cutoff, args.contact_cutoff)

    failures = []
    for code in tqdm(todo, desc="featurize"):
        target = complex_out / f"{code}.pt"
        if target.exists() and not args.overwrite:
            continue
        try:
            item = featurize_complex(code, complex_dirs[code], entries[code].pk, ligand_featurizer, protein_featurizer)
            torch.save(item, target)
        except Exception as exc:  # noqa: BLE001 - record and continue over a large dataset
            failures.append((code, type(exc).__name__, str(exc)))

    available = {p.stem for p in complex_out.glob("*.pt")}
    split_config = SplitConfig(
        strategy=args.split_strategy,
        val_fraction=args.val_fraction,
        val_min_year=args.val_min_year,
        seed=args.seed,
        exact_only=args.exact_only,
        refined_only=args.refined_only,
    )
    splits = make_splits(entries, set(core), available, split_config)
    save_splits(splits, out_dir / "splits.json", split_config)

    meta = {
        "protein_features": args.protein_features,
        "esm_model": args.esm_model if args.protein_features == "esm2" else None,
        "protein_dim": protein_featurizer.feature_dim,
        "ligand_dim": ATOM_FEATURE_DIM,
        "pocket_cutoff": args.pocket_cutoff,
        "contact_cutoff": args.contact_cutoff,
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if failures:
        with open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([("code", "error", "message"), *failures])
    print(f"Featurized {len(available)} complexes ({len(failures)} failures) into {out_dir.resolve()}")
    print(f"Splits: {meta['split_sizes']}")


if __name__ == "__main__":
    main()
