"""Compare two BindFormer configurations on the same split, with uncertainty.

Each configuration is one or more run directories (one per seed) holding
`predictions_<split>.csv` from scripts/train.py.

    python scripts/compare.py --a checkpoints_esm_s*/ --b checkpoints_onehot/ \
        --split casf2016 --label_a esm2 --label_b onehot

Reports, per metric: each configuration's per-seed spread, a bootstrap CI, and a
paired bootstrap of the difference over the complexes both were scored on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from bindformer.training.stats import (
    METRICS,
    _apply,
    bootstrap_ci,
    min_detectable_delta,
    paired_bootstrap,
    seed_summary,
)


def read_predictions(path: Path) -> dict[str, tuple[float, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {r["complex_id"]: (float(r["pred_pk"]), float(r["true_pk"])) for r in csv.DictReader(handle)}


def load_config_runs(dirs: list[Path], split: str) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Return shared complex ids, one prediction vector per seed, and the targets."""

    runs = []
    for d in dirs:
        path = d if d.is_file() else d / f"predictions_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run train.py for that configuration first")
        runs.append((path, read_predictions(path)))

    shared = set.intersection(*(set(r[1]) for r in runs))
    if not shared:
        raise ValueError("no complexes in common across runs")
    ids = sorted(shared)
    targets = np.array([runs[0][1][c][1] for c in ids])
    for path, table in runs:
        t = np.array([table[c][1] for c in ids])
        if not np.allclose(t, targets):
            raise ValueError(f"{path} has different true_pk values; the runs are not on the same labels")
    preds = [np.array([table[c][0] for c in ids]) for _, table in runs]
    return ids, preds, targets


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare two BindFormer configurations with uncertainty.")
    parser.add_argument("--a", nargs="+", required=True, help="Run dir(s) for configuration A, one per seed")
    parser.add_argument("--b", nargs="+", help="Run dir(s) for configuration B (omit to summarize A alone)")
    parser.add_argument("--split", default="casf2016")
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--metrics", nargs="+", default=["pearson_r", "rmse", "sd"], choices=sorted(METRICS))
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json_out")
    args = parser.parse_args(argv)

    ids_a, preds_a, target = load_config_runs([Path(p) for p in args.a], args.split)
    report: dict = {"split": args.split, "n_complexes": len(ids_a), "n_boot": args.n_boot, "configs": {}}

    groups = [(args.label_a, args.a, preds_a, target)]
    if args.b:
        ids_b, preds_b, target_b = load_config_runs([Path(p) for p in args.b], args.split)
        if ids_a != ids_b:
            shared = sorted(set(ids_a) & set(ids_b))
            keep_a = [ids_a.index(c) for c in shared]
            keep_b = [ids_b.index(c) for c in shared]
            print(f"note: comparing on {len(shared)} shared complexes ({len(ids_a)} vs {len(ids_b)} available)")
            preds_a = [p[keep_a] for p in preds_a]
            preds_b = [p[keep_b] for p in preds_b]
            target = target[keep_a]
            report["n_complexes"] = len(shared)
        else:
            assert np.allclose(target, target_b)
        groups.append((args.label_b, args.b, preds_b, target))

    print(f"\n{args.split}: {report['n_complexes']} complexes, {args.n_boot} bootstrap resamples\n")
    for label, dirs, preds, tgt in groups:
        mean_pred = np.mean(preds, axis=0)
        entry = {"n_seeds": len(preds), "runs": [str(d) for d in dirs], "metrics": {}}
        print(f"{label}  ({len(preds)} seed{'s' if len(preds) != 1 else ''})")
        for metric in args.metrics:
            per_seed = [_apply(metric, p, tgt) for p in preds]
            mean, sd = seed_summary(per_seed)
            ci = bootstrap_ci(mean_pred, tgt, metric, args.n_boot, args.seed)
            entry["metrics"][metric] = {
                "per_seed": per_seed,
                "seed_mean": mean,
                "seed_sd": sd,
                "seed_mean_predictions": {"point": ci.point, "lo": ci.lo, "hi": ci.hi},
            }
            spread = f" +/- {sd:.3f} over seeds" if len(preds) > 1 else ""
            print(f"  {metric:<13} per-seed {mean:.3f}{spread}   seed-mean preds {ci}")
        report["configs"][label] = entry
        print()

    if args.b:
        mean_a, mean_b = np.mean(preds_a, axis=0), np.mean(preds_b, axis=0)
        report["comparison"] = {}
        print(f"paired bootstrap: {args.label_b} - {args.label_a} (seed-mean predictions, same complexes)")
        for metric in args.metrics:
            c = paired_bootstrap(mean_a, mean_b, target, metric, args.n_boot, args.seed)
            verdict = {"a": args.label_a, "b": args.label_b, "neither": "neither"}[c.favours]
            mark = "significant" if c.significant else "NOT significant"
            print(
                f"  {metric:<13} {c.a:.3f} -> {c.b:.3f}  delta {c.delta:+.3f} "
                f"[{c.lo:+.3f}, {c.hi:+.3f}]  p={c.p_value:.4f}  {mark}, favours {verdict}"
            )
            report["comparison"][metric] = {
                "a": c.a, "b": c.b, "delta": c.delta, "lo": c.lo, "hi": c.hi,
                "p_value": c.p_value, "significant": c.significant, "favours": verdict,
            }
        resid_corr = float(np.corrcoef(mean_a - target, mean_b - target)[0, 1])
        report["residual_correlation"] = resid_corr
        base = report["configs"][args.label_a]["metrics"].get("pearson_r")
        if base:
            mdd = min_detectable_delta(base["seed_mean"], report["n_complexes"])
            print(f"\nresidual correlation between the two models: {resid_corr:.3f}")
            print(f"unpaired reference: a single run would need delta pearson_r > {mdd:+.3f} to clear 95%.")
            print(
                "Pairing only tightens the delta in proportion to that correlation, so it buys"
                "\nlittle for models this far apart and a lot for near-identical variants."
            )
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
