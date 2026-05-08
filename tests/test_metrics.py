"""Tests for evaluation/metrics.py."""

import numpy as np
import pytest

from neural_priors_gym.evaluation.metrics import compute_jsd, compute_jsd_1d


RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# compute_jsd_1d
# ---------------------------------------------------------------------------


def test_jsd_1d_identical_distributions_near_zero() -> None:
    samples = RNG.normal(0, 1, 2000)
    jsd = compute_jsd_1d(samples, samples.copy())
    assert jsd < 0.01


def test_jsd_1d_symmetric() -> None:
    p = RNG.normal(0, 1, 500)
    q = RNG.normal(3, 1, 500)
    assert abs(compute_jsd_1d(p, q) - compute_jsd_1d(q, p)) < 1e-10


def test_jsd_1d_bounded() -> None:
    p = RNG.normal(0, 0.01, 300)
    q = RNG.normal(100, 0.01, 300)
    jsd = compute_jsd_1d(p, q)
    assert 0.0 <= jsd <= 1.0 + 1e-9


def test_jsd_1d_non_overlapping_is_large() -> None:
    p = RNG.normal(0, 0.1, 300)
    q = RNG.normal(100, 0.1, 300)
    assert compute_jsd_1d(p, q) > 0.9


def test_jsd_1d_returns_float() -> None:
    p = RNG.uniform(0, 1, 100)
    q = RNG.uniform(0, 1, 100)
    assert isinstance(compute_jsd_1d(p, q), float)


def test_jsd_1d_finite() -> None:
    p = RNG.exponential(1, 200)
    q = RNG.exponential(2, 200)
    assert np.isfinite(compute_jsd_1d(p, q))


# ---------------------------------------------------------------------------
# compute_jsd (multi-dimensional)
# ---------------------------------------------------------------------------


def test_jsd_multidim_keys() -> None:
    p = RNG.normal(0, 1, (200, 3))
    q = RNG.normal(0, 1, (200, 3))
    result = compute_jsd(p, q)
    for i in range(3):
        assert f"dim_{i}_bits" in result
        assert f"dim_{i}_millibits" in result
    assert "mean_millibits" in result


def test_jsd_multidim_mean_is_average_of_dims() -> None:
    p = RNG.normal(0, 1, (300, 4))
    q = RNG.normal(1, 1, (300, 4))
    result = compute_jsd(p, q)
    per_dim = [result[f"dim_{i}_millibits"] for i in range(4)]
    assert result["mean_millibits"] == pytest.approx(np.mean(per_dim))


def test_jsd_multidim_millibits_equals_bits_times_1000() -> None:
    p = RNG.normal(0, 1, (200, 2))
    q = RNG.normal(2, 1, (200, 2))
    result = compute_jsd(p, q)
    for i in range(2):
        assert result[f"dim_{i}_millibits"] == pytest.approx(
            result[f"dim_{i}_bits"] * 1000.0
        )


def test_jsd_multidim_identical_near_zero() -> None:
    p = RNG.normal(0, 1, (500, 2))
    result = compute_jsd(p, p.copy())
    assert result["mean_millibits"] < 10.0


def test_jsd_multidim_all_values_finite() -> None:
    p = RNG.uniform(1, 2, (200, 4))
    q = RNG.uniform(1, 2, (200, 4))
    result = compute_jsd(p, q)
    assert all(np.isfinite(v) for v in result.values())
