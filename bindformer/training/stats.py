"""Uncertainty for scoring-power numbers: bootstrap CIs and paired model comparison.

A single CASF-2016 run reports one Pearson R over 285 complexes, which carries a
95% CI of roughly +/-0.06. Phase 2 changes are expected to move R by 0.01-0.03, so
they are invisible without this. Two noise sources matter and are reported apart:

* test-set sampling - which 285 complexes - removed by pairing the bootstrap, since
  both models are scored on the same complexes.
* initialization - removed by averaging over seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bindformer.training.metrics import casf_sd, pearson_r, rmse, spearman_rho

import torch

METRICS = {
    "pearson_r": pearson_r,
    "spearman_rho": spearman_rho,
    "rmse": rmse,
    "sd": casf_sd,
}
LOWER_IS_BETTER = {"rmse", "sd", "mae"}


def _apply(metric: str, pred: np.ndarray, target: np.ndarray) -> float:
    fn = METRICS[metric]
    return fn(torch.from_numpy(np.ascontiguousarray(pred)), torch.from_numpy(np.ascontiguousarray(target)))


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n_boot: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


@dataclass
class Comparison:
    metric: str
    a: float
    b: float
    delta: float  # b - a
    lo: float
    hi: float
    p_value: float
    n: int
    n_boot: int

    @property
    def significant(self) -> bool:
        return self.lo > 0 or self.hi < 0

    @property
    def favours(self) -> str:
        better_when_negative = self.metric in LOWER_IS_BETTER
        if not self.significant:
            return "neither"
        return "b" if (self.delta < 0) == better_when_negative else "a"


def bootstrap_ci(
    pred: np.ndarray, target: np.ndarray, metric: str = "pearson_r", n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> Interval:
    rng = np.random.default_rng(seed)
    n = len(pred)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = _apply(metric, pred[idx], target[idx])
    lo, hi = np.quantile(draws[np.isfinite(draws)], [alpha / 2, 1 - alpha / 2])
    return Interval(_apply(metric, pred, target), float(lo), float(hi), n_boot)


def paired_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    metric: str = "pearson_r",
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Comparison:
    """Compare two models on the SAME complexes; resample complexes, not models."""

    if not (len(pred_a) == len(pred_b) == len(target)):
        raise ValueError("paired bootstrap needs predictions over the same complexes")
    rng = np.random.default_rng(seed)
    n = len(target)
    a_point, b_point = _apply(metric, pred_a, target), _apply(metric, pred_b, target)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        t = target[idx]
        deltas[i] = _apply(metric, pred_b[idx], t) - _apply(metric, pred_a[idx], t)
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    # Achieved significance level: how much of the bootstrap mass sits on the wrong side of 0.
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return Comparison(metric, a_point, b_point, b_point - a_point, float(lo), float(hi), float(min(1.0, p)), n, len(deltas))


def seed_summary(values: list[float]) -> tuple[float, float]:
    """Mean and standard deviation over seeds (sd is 0.0 for a single run)."""

    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def min_detectable_delta(r: float, n: int, alpha: float = 0.05) -> float:
    """Smallest Pearson gain over `r` that an UNPAIRED single run could resolve.

    The paired bootstrap above is tighter; this is the pessimistic reference that
    motivates pairing in the first place.
    """

    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    crit = 1.96 * se * math.sqrt(2) if alpha == 0.05 else 2.576 * se * math.sqrt(2)
    hi = math.tanh(z + crit)
    return hi - r
