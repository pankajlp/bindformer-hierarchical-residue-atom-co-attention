# BindFormer results — current

Supersedes `PHASE1_RESULTS.md`, `PHASE2_RBF_RESULTS.md` and
`PHASE2_ATTNPOOL_RESULTS.md`, whose CASF numbers were all measured through a
train/test preprocessing bug (below). Apple M5 Max (MPS), PDBbind v2020,
CASF-2016 scoring power, splits identical across every run
(train 17,238 / val 1,915 / casf2016 285, zero overlap).

## The bug that changed every number

The CASF-2016 coreset SDFs carry explicit hydrogens; PDBbind's general and refined
SDFs carry none. RDKit's `removeHs=True` spares hydrogens it considers meaningful,
so it did not strip them: **36% of cached CASF ligand atoms were hydrogen** while
training ligands were heavy-atom only. Test ligands were about 1.6x the length of
training ligands (median 36 atoms vs 23), and because the pocket crop and the 4 A
contact labels come from ligand coordinates, those were computed against hydrogen
positions too.

Fixed in `load_ligand` with `RemoveAllHs` on every parse path. Training data was
almost unaffected (~10% of complexes had a few stray hydrogens), so the fix only
required re-featurizing and re-scoring the existing checkpoints — no retraining.

Correcting it raised CASF Pearson by about 0.08 across the board and **reversed two
of the project's headline conclusions**.

## Baseline

| | per-seed (3 seeds) | seed sd | 3-seed ensemble |
|-|--------------------|---------|-----------------|
| CASF Pearson | **0.748** | 0.011 | **0.772 [0.705, 0.828]** |
| CASF RMSE | 1.47 | 0.03 | 1.437 |

Per seed 0.751 / 0.758 / 0.736 (was 0.691 / 0.672 / 0.648 through the bug).
Against the ~0.80-0.85 published range this is a much less embarrassing Phase 1
floor than the 0.670 previously reported.

## What a model has to beat

Trivial predictors fitted on train, scored on CASF (`scripts/trivial_baselines.py`):

| predictor                        | CASF Pearson | RMSE  |
|----------------------------------|--------------|-------|
| ligand heavy-atom count alone    | 0.500        | 2.124 |
| pocket residue count alone       | 0.508        | 1.978 |
| ridge on element counts          | 0.526        | 1.910 |
| ridge on mean/sum atom features  | 0.526        | 1.913 |
| **BindFormer (per seed)**        | **0.748**    | 1.47  |

Counting atoms gets 0.5 on CASF, so roughly two thirds of the raw correlation needs
no binding model at all. BindFormer adds ~0.22 over that floor. Worth keeping in
view when quoting any single Pearson number as "scoring power".

## ESM-2 vs one-hot — previous conclusion withdrawn

| protein features | CASF Pearson | val Pearson |
|------------------|--------------|-------------|
| ESM-2 650M (3 seeds) | 0.748    | 0.741-0.751 |
| one-hot (1 seed)     | 0.722    | 0.727       |

Paired bootstrap, seed-mean predictions, 285 complexes:
**delta -0.050 [-0.108, +0.009], p=0.095 — not significant.**

Previously reported as -0.154 at p=0.0002, i.e. "ESM-2 is worth +0.140 Pearson".
That was an artifact: the hydrogen-corrupted ligands hurt the weaker one-hot model
far more (0.551) than the ESM-2 model (0.670), inflating the apparent gap threefold.
**ESM-2 may be worth ~0.05 Pearson here, but on this evidence it is not established.**
The one-hot arm is still a single seed; more seeds would tighten the seed term but
the interval is dominated by complex sampling (residual correlation 0.730), so the
effect sits near the detection limit either way.

The related claim that "validation compresses the ESM-2 gap tenfold" is also
withdrawn: corrected, the gap is 0.019 on val and 0.026 on CASF. For this comparison
val was a fair proxy after all.

What does survive: **val does not rank seeds.** Seed 44 has the best val Pearson
(0.751) and the worst corrected CASF (0.736); seed 43 is middling on val (0.745) and
best on CASF (0.758). Val moves 0.010 across seeds while CASF moves 0.022. With three
seeds this is suggestive, not established, but it is a reason not to pick checkpoints
or configurations on val.

## Phase 2 scorecard (corrected)

Each arm is 3 seeds against the 3 baseline seeds, paired bootstrap over the same 285
complexes, 10,000 resamples.

| item | per-seed CASF | delta vs baseline | p | verdict |
|------|---------------|-------------------|---|---------|
| RBF distance bias | 0.744 | -0.003 [-0.024, +0.018] | 0.77 | no effect |
| attention pool (as first built) | 0.740 | -0.004 [-0.027, +0.018] | 0.70 | no effect, and the layer never trained |
| attention pool (sharpened) | 0.718 | **-0.032 [-0.056, -0.010]** | **0.0076** | **significantly worse** |

The sharpened-pooling verdict changed with the fix: through the bug it read as a null
(p=0.81), corrected it is a real regression, with RMSE agreeing (+0.076, p=0.004).
Learned selective pooling does not merely fail to help, it costs about 0.03 Pearson.
Diagnostics confirm both mechanisms were live rather than inert — the RBF bias spanned
1.4-2.6 attention logits, and the sharpened pool reached attention entropy 0.05-0.51
of uniform with the top token taking 23-58x its even share.

So the readout and the attention-level geometry are both ruled out as the limiting
factor, and one of them is actively harmful when given capacity.

## Where the remaining gap probably is

Against ~0.80-0.85 published, the corrected baseline is 0.748 per seed / 0.772
ensembled. Two attempts at adding attention capacity produced nothing and one hurt,
while a single preprocessing fix was worth +0.08 — more than any architectural change
attempted. That asymmetry argues for spending the next effort on data and objective:

* the trivial floor of 0.5 suggests checking how much of the score survives
  controlling for ligand size
* the pocket crop and attention mask are both derived from the true ligand pose, so
  pose sensitivity is untested and the 10 A crop may leak pocket-size information
* harder splits (LP-PDBBind, protein-family) to see how much of 0.748 is train/test
  similarity
* the contact auxiliary head is trained but never evaluated against PLIP interactions

## Tooling

| script | what it is for |
|--------|----------------|
| `scripts/compare.py` | two configurations over N seeds, paired bootstrap, CIs, p-values |
| `scripts/eval_checkpoints.py` | re-score saved checkpoints on a split without retraining |
| `scripts/pool_diagnostics.py` | how far attention pooling is from uniform |
| `scripts/trivial_baselines.py` | no-binding predictors, the floor to beat |
| `scripts/train_seeds.sh` | seed sweeps, laid out for `compare.py` |
| `scripts/run_detached.py` | survive a closing terminal |

## Reproduce the corrected numbers

```bash
python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed_h
python scripts/eval_checkpoints.py --processed_dir data/processed_h --split casf2016 \
    --suffix _hfix checkpoints checkpoints_esm_s43 checkpoints_esm_s44
python scripts/compare.py --a checkpoints_hfix checkpoints_esm_s43_hfix checkpoints_esm_s44_hfix \
    --b checkpoints_rbf_s42_hfix checkpoints_rbf_s43_hfix checkpoints_rbf_s44_hfix \
    --label_a baseline --label_b rbf
```
