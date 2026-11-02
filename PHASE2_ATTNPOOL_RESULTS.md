# Phase 2, item 2: attention-pooled readout — inconclusive, not a clean negative

Verdict: **no measurable change, but the experiment did not test the hypothesis.**
The pooling layer stayed at mean pooling throughout training, so this says nothing
about whether a learned readout would help. Needs a rerun before it can be called.

## What was added

`bindformer/models/pooling.py`. The affinity head pooled residues and atoms with a
plain `masked_mean`, so a residue packed against the ligand counted the same as one
at the pocket edge. `AttentionPool` scores each token with `Linear(hidden_dim, 1)`
and takes the softmax-weighted sum.

The scorer is zero-initialized. Uniform attention weights *are* the mean, so at
initialization the layer is exactly `masked_mean` (asserted to 1e-6 in
`tests/test_pooling.py`, ragged batches included).

## Result: 3 seeds vs 3 baseline seeds, identical splits

| configuration | per-seed CASF Pearson | seed sd | seed-mean preds      |
|---------------|-----------------------|---------|----------------------|
| baseline      | 0.670                 | 0.022   | 0.705 [0.637, 0.764] |
| + attn pool   | 0.649                 | 0.025   | 0.696 [0.617, 0.762] |

Per seed — baseline 0.691 / 0.672 / 0.648, attn pool 0.639 / 0.677 / 0.632.

Paired bootstrap, same 285 complexes, 10,000 resamples (residual corr 0.922):

| metric    | delta (attnpool - baseline) | 95% CI           | p    | verdict         |
|-----------|-----------------------------|------------------|------|-----------------|
| pearson_r | -0.009                      | [-0.050, +0.028] | 0.63 | not significant |
| rmse      | +0.028                      | [-0.060, +0.121] | 0.54 | not significant |
| sd        | +0.020                      | [-0.060, +0.104] | 0.63 | not significant |

## Why this is inconclusive rather than a negative

The same diagnostic that validated the RBF negative result undermines this one.
Measured over the 285 CASF complexes with the trained models:

| seed | protein pool entropy / uniform | max weight x n_res | \|W\| protein | \|W\| ligand | ligand entropy / uniform |
|------|-------------------------------|--------------------|---------------|--------------|--------------------------|
| 42   | 0.999                         | 1.16               | 0.095         | 0.214        | 0.997                    |
| 43   | 0.999                         | 1.18               | 0.099         | 0.236        | 0.996                    |
| 44   | 0.999                         | 1.23               | 0.097         | 0.230        | 0.999                    |

An entropy ratio of 1.000 is exactly uniform. All three seeds sit at 0.996-0.999,
and the most-weighted residue gets only 1.16-1.23x its uniform share. **The layer
never left mean pooling**, so the null delta is close to a tautology: the two
configurations compute almost the same function.

Contrast the RBF result, where the learned bias spanned 1.4-2.6 attention logits and
genuinely reshaped the softmax. That null was informative — geometry was used and
did not help. This one is not.

Likely cause: a single zero-init linear layer on LayerNorm'd features gets a weak
gradient, and the fusion trunk downstream can absorb whatever the pooling would have
contributed, so there is little pressure to differentiate tokens.

## What a real test needs

One or more of, then the same 3-seed paired comparison:

* a higher learning rate for the scorer, or non-zero init (gives up exact baseline
  equivalence at init, so keep a seed-matched control)
* a softmax temperature below 1, sharpening weights for the same logit scale
* a deeper scorer (`Linear -> GELU -> Linear`) instead of one linear layer
* an auxiliary signal tying pooling weights to the contact head, which already knows
  which residues touch the ligand

Until then, treat "attention-pooled readout" as untested, not as rejected.

## Reproduce

```bash
scripts/run_detached.py . attnchain.log /bin/zsh \
    scripts/train_seeds.sh configs/phase2_attnpool.yaml checkpoints_attn_s 42 43 44
python scripts/compare.py --a checkpoints checkpoints_esm_s43 checkpoints_esm_s44 \
    --b checkpoints_attn_s42 checkpoints_attn_s43 checkpoints_attn_s44 \
    --label_a baseline --label_b attnpool
```

Runs took 82 / 122 / 78 min on M5 Max (44 / 67 / 46 epochs).
