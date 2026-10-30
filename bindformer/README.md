# BindFormer

**Hierarchical Residue-Atom Co-Attention for Interpretable Protein-Ligand Binding Prediction**

BindFormer predicts protein-ligand binding affinity (pK) from a complex structure. Pocket residues (ESM-2 embeddings) and ligand heavy atoms (74-dim atom features plus a bond-graph attention bias) are encoded separately, then exchanged through bidirectional residue-atom co-attention restricted by a distance mask. An auxiliary head predicts residue-atom contacts.

The model scores a **given pose**: the pocket and attention mask come from the ligand's crystal coordinates. It does not predict pockets or poses.

## Project Layout

```text
bindformer/
├── configs/default.yaml
├── data/          # INDEX/CoreSet parsing, splits, featurizers, dataset
├── models/        # encoders, residue-atom attention, fusion head
├── training/      # loss, metrics, trainer
├── utils/         # PDB parsing, pocket cropping, masks, plotting
├── scripts/       # preprocess.py, train.py
└── notebooks/
tests/             # synthetic PDBbind-shaped fixtures; run with pytest
```

## Setup

```bash
conda create -n bindformer python=3.10
conda activate bindformer
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r bindformer/requirements.txt
pytest tests
```

## Data

Download PDBbind v2020 (general + refined sets and the `index/` folder) and CASF-2016 from [PDBbind](http://www.pdbbind.org.cn/) after registering. Extract everything under one root; nesting doesn't matter:

```text
data/raw/
├── index/INDEX_general_PL_data.2020, INDEX_refined_data.2020
├── v2020-other-PL/<code>/<code>_protein.pdb, <code>_ligand.sdf, <code>_ligand.mol2
├── refined-set/<code>/...
└── CASF-2016/coreset/<code>/...,  CASF-2016/power_scoring/CoreSet.dat
```

Run from inside `bindformer/`:

```bash
python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed
```

This writes:

- `data/processed/complexes/<code>.pt`: 10 Å pocket residues, ligand atoms, topological distances, 4 Å contact labels, pK.
- `splits.json`: `train` / `val` / `casf2016`. The 285 CASF-2016 core complexes never enter train or val.
- `meta.json`: feature dimensions, which `train.py` reads automatically.
- `failures.csv`: complexes that could not be parsed, if any.

Useful flags:

- `--protein_features onehot` skips ESM-2 (fast, no model download).
- `--split_strategy time --val_min_year 2019` makes a temporal val split.
- `--exact_only` drops `<`, `>`, `~` affinities; `--refined_only` uses only the refined set.
- `--limit 200` runs a quick subset.

ESM-2 (`facebook/esm2_t33_650M_UR50D`, ~2.5 GB) is downloaded from Hugging Face on first use. Embeddings are cached per unique chain sequence in `data/processed/esm_cache/`, so proteins shared across complexes are embedded once.

## Train

```bash
python scripts/train.py --config configs/default.yaml
```

This uses a cosine LR schedule with warmup and early stopping on validation RMSE. Logs go to `checkpoints/log.jsonl`, and the run saves `best.pt` and `last.pt`. When training ends, it evaluates val and CASF-2016 into `results.json` and `predictions_<split>.csv`.

## Evaluate

```bash
python scripts/train.py --config configs/default.yaml --mode eval --checkpoint checkpoints/best.pt --checkpoint_dir runs/eval
```

Reported metrics are RMSE, MAE, Pearson R, Spearman ρ, and the CASF scoring-power SD (the residual standard deviation of the linear fit).

## Core Idea

Residue-Atom Attention (RAA) restricts cross-attention to residue-atom pairs within a distance cutoff. Proteins are cropped to the pocket (residues with any heavy atom within 10 Å of the ligand) before encoding, so encoder and co-attention cost scale with pocket size `k` rather than the whole protein.
