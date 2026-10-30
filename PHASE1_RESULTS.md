# BindFormer Phase 1 — baseline on real PDBbind v2020 / CASF-2016

Run on Apple M5 Max (MPS), PDBbind v2020, splits identical across both runs
(train 17,238 / val 1,915 / casf2016 285, zero overlap).

## CASF-2016 scoring power (285 complexes)

| protein features | Pearson R | RMSE  | SD    | Spearman | MAE   |
|------------------|-----------|-------|-------|----------|-------|
| **ESM-2 650M**   | **0.691** | 1.570 | 1.572 | 0.673    | 1.243 |
| one-hot          | 0.551     | 1.967 | 1.814 | 0.564    | 1.503 |

**ESM-2 contributes +0.140 Pearson / -0.397 RMSE on CASF-2016.**

## Validation (1,915 complexes, random 10%)

| protein features | Pearson R | RMSE  | SD    |
|------------------|-----------|-------|-------|
| ESM-2 650M       | 0.741     | 1.264 | 1.252 |
| one-hot          | 0.726     | 1.308 | 1.283 |

## The interesting part

On validation the two are nearly tied (0.741 vs 0.726, a 0.015 gap). On CASF-2016
the gap is ~10x larger (0.691 vs 0.551). Validation is a *random* 10% split, so it
shares protein families with train; one-hot residue identity is enough to do well
on pockets whose families were memorized. CASF-2016 is a diverse, curated held-out
set, and there one-hot collapses while ESM-2 holds up.

Conclusion: **random-split validation badly overestimates a one-hot baseline.** The
value of ESM-2 here is generalization to unseen proteins, which only the CASF number
exposes. Any future ablation should be judged on CASF (or a family-aware split), not val.

Supporting detail: on CASF, one-hot is also systematically biased low
(mean pred - true = -0.710 vs +0.024 for ESM-2) and under-disperses
(pred SD 1.491 vs target SD 2.174).

## Training

| run    | epochs | best epoch | stop reason             |
|--------|--------|------------|-------------------------|
| ESM-2  | 41     | 26         | early stop (patience 15)|
| one-hot| 100    | 92         | ran out of epochs       |

Caveat: the comparison is not symmetric in budget. ESM-2 early-stopped at 41 while
one-hot used all 100 epochs and was still improving slowly at the end. If anything
this favours one-hot, so the CASF gap is a conservative estimate of what ESM-2 adds.

## Context

Published models reach ~0.80-0.85 Pearson on CASF-2016. 0.691 is a reasonable Phase 1
floor: the protein encoder still has no geometric features beyond the 10 A distance
mask in co-attention, and no Phase 2 items (RBF distance bias, per-layer FFN,
attention-pooled readout, true residue->atom hierarchy) are implemented.

## Reproduce

```bash
python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed
python scripts/train.py --config configs/default.yaml

python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed_onehot \
    --protein_features onehot
python scripts/train.py --config configs/default.yaml \
    --processed_dir data/processed_onehot --checkpoint_dir checkpoints_onehot
```

Preprocessing: 10m39s with ESM-2 (19,438/19,443 complexes, 5 unparseable ligand SDFs
in the general set, none in CASF), 2m29s with one-hot.
