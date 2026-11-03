> **Superseded.** Every CASF number here was measured through the ligand-hydrogen
> train/test bug fixed in commit `36491d8`; correcting it moves CASF Pearson by about
> +0.08 and changes some conclusions. See `RESULTS.md` for current numbers.

# BindFormer Phase 1 — baseline on real PDBbind v2020 / CASF-2016

Apple M5 Max (MPS), PDBbind v2020. Splits identical across every run below
(train 17,238 / val 1,915 / casf2016 285, zero overlap, verified complex-by-complex).

## Headline: CASF-2016 scoring power

ESM-2 650M protein features, three seeds:

| seed | CASF Pearson R | RMSE  | SD    | val Pearson | epochs | best |
|------|----------------|-------|-------|-------------|--------|------|
| 42   | 0.691          | 1.570 | 1.572 | 0.741       | 41     | 26   |
| 43   | 0.672          | 1.612 | 1.610 | 0.743       | 54     | 39   |
| 44   | 0.648          | 1.683 | 1.657 | 0.750       | 46     | 31   |

**Single-model baseline: Pearson R = 0.670 +/- 0.022 (sd over seeds), RMSE 1.622 +/- 0.057.**

Averaging the three seeds' predictions (a 3-model ensemble) gives
**R = 0.705 [0.637, 0.764]**, RMSE 1.543 — better than any individual seed.

Report the 0.670 +/- 0.022 figure as the architecture's score. A single run
reporting 0.691 is the top of a 0.043-wide seed spread, not a property of the model.

## One-hot ablation (what ESM-2 adds)

| protein features        | CASF Pearson R  | RMSE  | SD    |
|-------------------------|-----------------|-------|-------|
| ESM-2 650M (3-seed mean pred) | 0.705     | 1.543 | 1.542 |
| one-hot (1 seed)        | 0.551           | 1.967 | 1.814 |

Paired bootstrap over the same 285 complexes, 10,000 resamples:

| metric    | delta (onehot - esm2) | 95% CI           | p      | verdict        |
|-----------|-----------------------|------------------|--------|----------------|
| pearson_r | -0.154                | [-0.241, -0.069] | 0.0002 | favours ESM-2  |
| rmse      | +0.424                | [+0.238, +0.611] | 0.0000 | favours ESM-2  |
| sd        | +0.272                | [+0.119, +0.430] | 0.0002 | favours ESM-2  |

The ESM-2 advantage is roughly 6x the seed-to-seed sd, so it survives the fact that
the one-hot arm is a single seed. Caveat: ESM-2 runs early-stopped at 41-54 epochs
while one-hot used all 100 and was still improving, which if anything favours one-hot.

## Validation does not rank models, and barely ranks seeds

| split            | esm2 spread over 3 seeds | onehot | gap   |
|------------------|--------------------------|--------|-------|
| val (random 10%) | 0.741 - 0.750            | 0.726  | 0.015 |
| casf2016         | 0.648 - 0.691            | 0.551  | 0.154 |

Two separate failures of the random-split validation set:

1. **It compresses the ESM-2/one-hot gap tenfold** (0.015 vs 0.154). A random 10%
   split shares protein families with train, so one-hot residue identity suffices
   on familiar pockets. CASF-2016 is diverse and held out, and one-hot collapses there.
2. **Across seeds it is anti-correlated with CASF.** The seed with the best val R
   (44, 0.750) has the worst CASF R (0.648); the worst val (42, 0.741) has the best
   CASF (0.691). Val R varies by 0.009 across seeds while CASF varies by 0.043.

Consequence: judge every change on CASF (or a family-aware split). Do not use val
to pick between configurations, and treat val-based early stopping as a convenience
for terminating a run, not as model selection. Within one run, epoch 26 vs epoch 41
of seed 42 differ by 0.024 CASF Pearson at p=0.17 — i.e. not resolvable, so the
val-selected checkpoint is neither better nor worse than the last one.

## Measurement floor (why the harness exists)

- 95% CI on a single run's CASF Pearson: about +/-0.06 (bootstrap, n=285).
- Seed-to-seed sd: 0.022.
- Smallest delta an UNPAIRED single run can resolve vs R=0.67: about +0.081.
- Paired bootstrap narrows this in proportion to the two models' residual
  correlation: +/-0.035 at corr 0.914 (epoch 26 vs 41 of one run), no gain at
  corr 0.67 (ESM-2 vs one-hot).

Phase 2 items are each expected to move R by 0.01-0.03 — below all of these
thresholds for a single run. Every Phase 2 claim needs >= 3 seeds plus the paired
test, or it is unfalsifiable.

```bash
python scripts/compare.py --a checkpoints checkpoints_esm_s43 checkpoints_esm_s44 \
    --b checkpoints_onehot --split casf2016 --label_a esm2 --label_b onehot
```

## Context

Published models reach ~0.80-0.85 Pearson on CASF-2016; 0.670 +/- 0.022 is a
reasonable Phase 1 floor. The protein encoder still has no geometric features
beyond the binary 10 A co-attention mask, and no Phase 2 item (RBF distance bias,
per-layer FFN, attention-pooled readout, true residue->atom hierarchy) is in yet.

## Reproduce

```bash
python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed
for s in 42 43 44; do
  python scripts/train.py --config configs/default.yaml --seed $s \
      --checkpoint_dir checkpoints_esm_s$s
done

python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed_onehot \
    --protein_features onehot
python scripts/train.py --config configs/default.yaml \
    --processed_dir data/processed_onehot --checkpoint_dir checkpoints_onehot
```

Preprocessing: 10m39s with ESM-2 (19,438/19,443 complexes; 5 unparseable ligand SDFs
in the general set, none in CASF), 2m29s with one-hot. Training: ~70 min per seed.
