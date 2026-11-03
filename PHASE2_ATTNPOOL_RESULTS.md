> **Superseded.** Every CASF number here was measured through the ligand-hydrogen
> train/test bug fixed in commit `36491d8`; correcting it moves CASF Pearson by about
> +0.08 and changes some conclusions. See `RESULTS.md` for current numbers.

# Phase 2, item 2: attention-pooled readout — negative result

Verdict after two attempts: **learned pooling does not improve CASF-2016 scoring
power.** The first attempt was inconclusive because the layer never trained; the
second trained it hard and the score did not move. Kept, off by default.

## What was added

`bindformer/models/pooling.py`. The affinity head pooled residues and atoms with a
plain `masked_mean`, so a residue packed against the ligand counted the same as one
at the pocket edge. `AttentionPool` scores each token and takes the softmax-weighted
sum. The scorer's last layer is zero-init, so uniform weights make the layer exactly
`masked_mean` at initialization (asserted to 1e-6, ragged batches included).

## Attempt 1 — inconclusive: the layer stayed at mean pooling

Single linear scorer, temperature 1, model learning rate.

| | per-seed CASF Pearson | seed sd | paired delta vs baseline | p |
|-|-----------------------|---------|--------------------------|---|
| attempt 1 | 0.649 | 0.025 | -0.009 [-0.050, +0.028] | 0.63 |

Diagnostic over the 285 CASF complexes: attention entropy **0.999** of uniform
(1.000 is exactly uniform) and the top-weighted token took only 1.16-1.24x its even
share. The two configurations computed almost the same function, so the null delta
was near-tautological and said nothing about the hypothesis.

## Attempt 2 — the layer trains, and the score still does not move

Three changes, all preserving exact mean-pool equivalence at init: an MLP scorer
(`Linear -> GELU -> Linear`, last layer zero-init), softmax temperature 0.5, and a
separate optimizer group giving the scorer 20x the model learning rate
(`training.pool_lr_mult`). A 5-epoch probe confirmed deviation *before* seeds were
spent: entropy 0.813 / 0.671.

| configuration | per-seed CASF Pearson | seed sd | seed-mean preds      |
|---------------|-----------------------|---------|----------------------|
| baseline      | 0.670                 | 0.022   | 0.705 [0.637, 0.764] |
| + attn pool   | 0.667                 | 0.011   | 0.701 [0.630, 0.763] |

Per seed — baseline 0.691 / 0.672 / 0.648, sharp pool 0.662 / 0.679 / 0.661.

Paired bootstrap, same 285 complexes, 10,000 resamples (residual corr 0.937):

| metric    | delta (sharp - baseline) | 95% CI           | p    | verdict         |
|-----------|--------------------------|------------------|------|-----------------|
| pearson_r | -0.004                   | [-0.036, +0.029] | 0.81 | not significant |
| rmse      | +0.033                   | [-0.045, +0.111] | 0.39 | not significant |
| sd        | +0.008                   | [-0.063, +0.076] | 0.81 | not significant |

Pooling at the selected checkpoints, now far from uniform:

| seed | protein entropy/uniform | protein top x | ligand entropy/uniform | ligand top x |
|------|-------------------------|---------------|------------------------|--------------|
| 42   | 0.513                   | 23.1          | 0.169                  | 30.5         |
| 43   | 0.122                   | 57.8          | 0.093                  | 32.8         |
| 44   | 0.187                   | 49.3          | 0.050                  | 35.4         |

## What this actually shows

The readout is **not** the bottleneck. Mean pooling (entropy 1.000) and
near-argmax pooling (entropy 0.05-0.19, one atom taking 30x its even share) produce
the same CASF Pearson to within 0.004. Discarding almost every token costs nothing,
which means the pooled vector was never where the missing information lived.

The middle of the range was not hand-tuned and skipped: the optimizer passed through
it — the 5-epoch probe sat at entropy 0.67-0.81 — and kept sharpening because that is
what lowered val loss. Performance was flat across the whole path from uniform to
near-argmax, so this is a property of the problem, not of one chosen temperature.

Side note, consistent with the RBF run: seed sd falls from 0.022 to 0.011. Both
Phase 2 additions stabilize across seeds without improving the mean.

## Phase 2 scorecard

| item                  | verdict                        | diagnostic                     |
|-----------------------|--------------------------------|--------------------------------|
| RBF distance bias     | no improvement (p=0.45)        | bias spanned 1.4-2.6 logits    |
| attention-pooled readout | no improvement (p=0.81)     | entropy fell to 0.05-0.51      |

Geometry in the attention bias and the choice of readout are both ruled out as the
limiting factor. Neither moved CASF off 0.670 +/- 0.022. Remaining Phase 2 items are
the per-layer FFN in co-attention and the true residue->atom hierarchy; given these
two results, the more promising direction may be the data and objective (harder
splits, pose sensitivity, the contact auxiliary) rather than more attention capacity.

## Reproduce

```bash
scripts/run_detached.py . attn2chain.log /bin/zsh \
    scripts/train_seeds.sh configs/phase2_attnpool_sharp.yaml checkpoints_attn2_s 42 43 44
python scripts/compare.py --a checkpoints checkpoints_esm_s43 checkpoints_esm_s44 \
    --b checkpoints_attn2_s42 checkpoints_attn2_s43 checkpoints_attn2_s44 \
    --label_a baseline --label_b attnpool_sharp
python scripts/pool_diagnostics.py checkpoints_attn2_s4*/best.pt
```

Runs took 92 / 151 / 130 min; the last two were slowed by unrelated jobs on the machine.
