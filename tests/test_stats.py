import numpy as np
import pytest

from bindformer.training.stats import (
    bootstrap_ci,
    min_detectable_delta,
    paired_bootstrap,
    seed_summary,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    target = rng.normal(6.5, 2.0, 400)
    good = target + rng.normal(0, 0.8, 400)
    poor = target + rng.normal(0, 2.4, 400)
    return target, good, poor


def test_bootstrap_ci_brackets_the_point_estimate(toy):
    target, good, _ = toy
    ci = bootstrap_ci(good, target, "pearson_r", n_boot=400, seed=1)
    assert ci.lo < ci.point < ci.hi
    assert 0.0 < ci.lo and ci.hi <= 1.0


def test_paired_bootstrap_detects_a_real_gap(toy):
    target, good, poor = toy
    c = paired_bootstrap(poor, good, target, "pearson_r", n_boot=400, seed=1)
    assert c.delta > 0
    assert c.significant and c.favours == "b"
    assert c.p_value < 0.05


def test_paired_bootstrap_reports_no_gap_for_identical_predictions(toy):
    target, good, _ = toy
    c = paired_bootstrap(good, good, target, "pearson_r", n_boot=400, seed=1)
    assert c.delta == pytest.approx(0.0, abs=1e-12)
    assert not c.significant and c.favours == "neither"


def test_lower_is_better_metrics_flip_the_verdict(toy):
    target, good, poor = toy
    c = paired_bootstrap(poor, good, target, "rmse", n_boot=400, seed=1)
    assert c.delta < 0  # good has the smaller rmse
    assert c.significant and c.favours == "b"


def test_pairing_is_tighter_when_errors_are_correlated(toy):
    target, good, _ = toy
    rng = np.random.default_rng(2)
    nearly = good + rng.normal(0, 0.05, len(good))  # highly correlated with `good`
    tight = paired_bootstrap(good, nearly, target, "pearson_r", n_boot=600, seed=1)
    independent = min_detectable_delta(tight.a, len(target))
    assert (tight.hi - tight.lo) / 2 < independent


def test_mismatched_lengths_are_rejected(toy):
    target, good, _ = toy
    with pytest.raises(ValueError):
        paired_bootstrap(good[:-1], good, target, "pearson_r", n_boot=10)


def test_seed_summary():
    assert seed_summary([0.7]) == (0.7, 0.0)
    mean, sd = seed_summary([0.68, 0.70, 0.72])
    assert mean == pytest.approx(0.70)
    assert sd == pytest.approx(0.02)
