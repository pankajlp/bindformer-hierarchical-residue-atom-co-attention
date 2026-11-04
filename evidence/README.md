# Evidence

Everything needed to verify the paper and `../RESULTS.md` without retraining. Model weights
(`*.pt`, ~557 MB) are omitted as reproducible from the committed code and fixed seeds; every
*number* behind the paper is here, as the raw per-complex predictions it was computed from,
the per-epoch training logs, and the regenerated comparison and diagnostic outputs.

## Layout

```
evidence/
  runs/<run>/          per-run artifacts (no weights):
                         results.json              final CASF + val metrics
                         log.jsonl                 per-epoch train/val curve
                         predictions_casf2016.csv  285 rows: complex_id, pred_pk, true_pk
                         predictions_val.csv        1915 rows (training runs only)
                         train.log                 (baseline, onehot) cleaned stdout
  comparisons/         paired-bootstrap outputs (.txt human-readable, .json machine)
  diagnostics/         trivial baselines, pool diagnostics, pose sweeps, hydrogen evidence
  logs/                seed-chain provenance logs with wall-clock timestamps
```

A run directory named `<config>_s<seed>` is one training seed. A `_hfix` suffix is that
checkpoint re-scored on the hydrogen-corrected CASF cache (eval only, no training log); a
`_hfixval` suffix is the same on the validation split. Runs **without** a `_hfix` suffix
were scored on the original (buggy) cache and are kept for provenance — they are the
pre-correction numbers the paper withdraws.

## How to recompute any number yourself

Every reported Pearson/RMSE/SD is a deterministic function of a `predictions_casf2016.csv`.
To recompute a paired comparison from the committed predictions (no model, no GPU):

```bash
cd bindformer
python scripts/compare.py \
  --a ../evidence/runs/checkpoints_hfix ../evidence/runs/checkpoints_esm_s43_hfix ../evidence/runs/checkpoints_esm_s44_hfix \
  --b ../evidence/runs/checkpoints_rbf_s42_hfix ../evidence/runs/checkpoints_rbf_s43_hfix ../evidence/runs/checkpoints_rbf_s44_hfix \
  --label_a baseline --label_b rbf
```

The `comparisons/*.txt` files are exactly this, pre-run.

## Claim-to-file map

| Paper / RESULTS.md claim | Evidence |
| --- | --- |
| Baseline 0.748 ± 0.011 (3 seeds), ensemble 0.772 | `runs/checkpoints_hfix`, `runs/checkpoints_esm_s43_hfix`, `runs/checkpoints_esm_s44_hfix` (`results.json`) |
| Trivial no-binding floor: 0.50 / 0.51 / 0.526 (Table 1) | `diagnostics/trivial_baselines_corrected.txt` |
| The hydrogen tell: element-count ridge 0.042 < 0.46 | `diagnostics/trivial_baselines_buggy_hydrogen.txt` |
| Hydrogen bug: 36% H in CASF, 0% after fix; median 36→23 atoms | `diagnostics/hydrogen_bug_evidence.txt` |
| ESM-2 vs one-hot, corrected: −0.050, p=0.095 (Table 2) | `comparisons/esm_vs_onehot_corrected.{txt,json}` |
| ESM-2 vs one-hot, **buggy** (withdrawn): −0.154, p=0.0002 | `comparisons/esm_vs_onehot_BUGGY_provenance.{txt,json}` |
| RBF distance bias: −0.003, p=0.77 | `comparisons/rbf_vs_baseline_corrected.{txt,json}` |
| Attention pool, inert build: −0.004, p=0.70; entropy 0.999 | `comparisons/attn_inert_vs_baseline_corr.{txt,json}`, `diagnostics/pool_diagnostics.txt` |
| Attention pool, sharpened: −0.032, p=0.008; entropy 0.05–0.51 | `comparisons/attn_sharp_vs_baseline_corr.{txt,json}`, `diagnostics/pool_diagnostics.txt` |
| Pose sweep (Table 4): rotation −0.02, translation → 0 | `diagnostics/pose_sensitivity_baseline.txt` |
| Rotation penalty, matched baseline: −0.032 | `comparisons/rotation_penalty_base_h.txt` |
| Rotation penalty, pose-trained: −0.030 (no gain) | `comparisons/rotation_penalty_pose.txt` |
| Decoy objective vs matched control: +0.012, p=0.23 | `comparisons/pose_vs_matched_baseline.{txt,json}` |
| Decoy objective vs rescored baseline: +0.007, p=0.48 | `comparisons/pose_vs_rescored_baseline.{txt,json}` |
| Pose-ranking accuracy 83.6/84.6/88.3% | `runs/checkpoints_pose_s4{2,3,4}/log.jsonl` (`train_pose_acc`) |
| Seed-chain timing / provenance | `logs/*.log` |

## Reproduce from scratch (with retraining)

Fixed seeds (42, 43, 44) plus the committed code reproduce every run. See `../RESULTS.md`
for the full command sequence; in outline:

```bash
python scripts/preprocess.py --pdbbind_root data/raw --out_dir data/processed_h
scripts/run_detached.py . chain.log /bin/zsh scripts/train_seeds.sh \
    configs/default.yaml checkpoints_s 42 43 44
python scripts/eval_checkpoints.py --processed_dir data/processed_h --split casf2016 \
    --suffix _hfix checkpoints_s42 checkpoints_s43 checkpoints_s44
```

PDBbind v2020 and CASF-2016 are not redistributed here; obtain them from the respective
sources and extract into `data/raw/` as the preprocessing script expects.
