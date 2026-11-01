# Phase 2, item 1: RBF distance bias — negative result

Verdict: **no improvement in CASF-2016 scoring power.** Kept in the codebase,
off by default (`rbf_kernels: 0`), for later combination with other Phase 2 items.

## What was added

Gaussian radial-basis expansions of pairwise distances, mixed into one additive
logit per attention head (`bindformer/models/rbf.py`):

* protein self-attention: residue-residue CA distances, 16 kernels over 0-20 A.
  Previously the encoder had no geometry at all — pocket residues were an unordered set.
* residue-atom co-attention: 16 kernels over 0-10 A. Previously only the binary
  "inside the cutoff" mask.

The output projection is zero-initialized, so an untrained RBF model is numerically
identical to the baseline (asserted in `tests/test_rbf.py`). Any difference is learned.

## Result: 3 seeds vs 3 baseline seeds, identical splits

| configuration | per-seed CASF Pearson | seed sd | seed-mean preds   |
|---------------|-----------------------|---------|-------------------|
| baseline      | 0.670                 | 0.022   | 0.705 [0.637, 0.764] |
| + RBF bias    | 0.642                 | 0.002   | 0.693 [0.619, 0.757] |

Per seed — baseline 0.691 / 0.672 / 0.648, RBF 0.640 / 0.645 / 0.642.

Paired bootstrap over the same 285 complexes (10,000 resamples, residual corr 0.937):

| metric    | delta (rbf - baseline) | 95% CI           | p    | verdict         |
|-----------|------------------------|------------------|------|-----------------|
| pearson_r | -0.012                 | [-0.043, +0.019] | 0.45 | not significant |
| rmse      | +0.031                 | [-0.043, +0.103] | 0.40 | not significant |
| sd        | +0.025                 | [-0.040, +0.090] | 0.45 | not significant |

The point estimate is slightly negative on every metric and the interval spans zero.
The honest reading is that this change does not help; it is not established that it hurts.

## It is not inert, so this is a real negative

A no-op would be the trivial explanation, so the learned bias was measured by
sweeping distance through the trained modules and recording the logit span:

| seed | protein self-attn span | co-attention span |
|------|------------------------|-------------------|
| 42   | 0.60                   | 1.41              |
| 43   | 0.69                   | 2.64              |
| 44   | 0.54                   | 2.18              |

Attention logits are `q.k/sqrt(d)`, so a co-attention bias spanning 1.4-2.6 logits
reshapes the softmax substantially. The model learned a strong distance preference
and still did not generalize better. Geometry in this form is not the bottleneck.

## Side effect worth noting: seed variance collapses

Seed sd falls from 0.022 to 0.002, a ~10x reduction — the three RBF runs land within
0.005 of each other while the baseline spans 0.043. Plausibly the distance bias
removes a symmetry the baseline had to break at random. Treat with caution: an sd
from 3 points is itself very uncertain (variance-ratio F-test, df 2/2, p ~ 0.02).
If it holds up, RBF-style runs need fewer seeds to compare.

## Why one seed would have misled

| what you would have run | apparent conclusion       | truth           |
|-------------------------|---------------------------|-----------------|
| seed 42 only            | "RBF costs 0.051 Pearson" | not significant |
| seed-mean only          | "RBF costs 0.012"         | not significant |
| 3 seeds + paired test   | not significant, p=0.45   | --              |

Baseline seed 42 (0.691) was the luckiest of its three and RBF seed 42 (0.640) the
unluckiest comparison point, so the single-seed gap overstates the effect fourfold.

## Validation misled again

RBF runs have the *highest* val Pearson of anything trained so far (0.757 / 0.747 /
0.750 vs baseline 0.741 / 0.743 / 0.750) while scoring *worse* on CASF. That is the
fourth and fifth instance of the val/CASF inversion. Val continues to be unusable
for ranking configurations.

## Reproduce

```bash
for s in 42 43 44; do
  python scripts/train.py --config configs/phase2_rbf.yaml --seed $s \
      --checkpoint_dir checkpoints_rbf_s$s
done
python scripts/compare.py --a checkpoints checkpoints_esm_s43 checkpoints_esm_s44 \
    --b checkpoints_rbf_s42 checkpoints_rbf_s43 checkpoints_rbf_s44 \
    --label_a baseline --label_b rbf
```

~90 min per seed on M5 Max (MPS).
