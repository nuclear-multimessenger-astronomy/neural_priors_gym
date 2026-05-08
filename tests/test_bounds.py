"""Tests for out-of-bounds validation."""

import numpy as np
import pytest

from neural_priors_gym.evaluation.bounds import (
    PARAMETER_BOUNDS,
    check_out_of_bounds,
)


def test_parameter_bounds_registry_has_expected_entries():
    assert "mass_ratio" in PARAMETER_BOUNDS
    assert "lambda_1" in PARAMETER_BOUNDS
    assert "lambda_2" in PARAMETER_BOUNDS
    # mass_ratio has an upper bound of 1, no lower bound
    assert PARAMETER_BOUNDS["mass_ratio"] == (None, 1.0)
    # lambdas have a lower bound of 0, no upper bound
    assert PARAMETER_BOUNDS["lambda_1"] == (0.0, None)
    assert PARAMETER_BOUNDS["lambda_2"] == (0.0, None)


def test_no_violations_when_all_in_bounds():
    samples = np.column_stack(
        [
            np.full(100, 0.8),  # mass_ratio: all <= 1
            np.full(100, 500.0),  # lambda_1: all >= 0
            np.full(100, 200.0),  # lambda_2: all >= 0
        ]
    )
    results = check_out_of_bounds(samples, ["mass_ratio", "lambda_1", "lambda_2"])
    for name, stats in results.items():
        assert stats["pct_oob"] == 0.0, f"Expected 0% OOB for {name}"
        assert stats["n_oob"] == 0.0


def test_mass_ratio_upper_bound_violations():
    # 50 samples above 1.0, 50 samples below 1.0
    samples = np.column_stack(
        [
            np.concatenate([np.full(50, 1.5), np.full(50, 0.8)]),
        ]
    )
    results = check_out_of_bounds(samples, ["mass_ratio"])
    assert "mass_ratio" in results
    assert results["mass_ratio"]["pct_oob"] == pytest.approx(50.0)
    assert results["mass_ratio"]["pct_above"] == pytest.approx(50.0)
    assert results["mass_ratio"]["pct_below"] == pytest.approx(0.0)


def test_lambda_lower_bound_violations():
    # 10 negative lambda_1 out of 100 samples
    samples = np.column_stack(
        [
            np.concatenate([np.full(10, -1.0), np.full(90, 200.0)]),
        ]
    )
    results = check_out_of_bounds(samples, ["lambda_1"])
    assert "lambda_1" in results
    assert results["lambda_1"]["pct_oob"] == pytest.approx(10.0)
    assert results["lambda_1"]["pct_below"] == pytest.approx(10.0)
    assert results["lambda_1"]["pct_above"] == pytest.approx(0.0)


def test_unknown_parameters_are_skipped():
    samples = np.column_stack(
        [
            np.full(50, 1.4),  # mass_1_source: no registered bounds
            np.full(50, 1.2),  # mass_2_source: no registered bounds
        ]
    )
    results = check_out_of_bounds(samples, ["mass_1_source", "mass_2_source"])
    assert results == {}


def test_mixed_known_and_unknown_parameters():
    samples = np.column_stack(
        [
            np.full(100, 1.4),  # mass_1_source: unknown
            np.full(100, -50.0),  # lambda_1: violates lower bound
        ]
    )
    results = check_out_of_bounds(samples, ["mass_1_source", "lambda_1"])
    assert "mass_1_source" not in results
    assert "lambda_1" in results
    assert results["lambda_1"]["pct_oob"] == pytest.approx(100.0)


def test_n_samples_field():
    n = 200
    samples = np.ones((n, 1)) * 0.5
    results = check_out_of_bounds(samples, ["mass_ratio"])
    assert results["mass_ratio"]["n_samples"] == n


def test_all_in_bounds_returns_zero_oob_for_all_known_params():
    rng = np.random.default_rng(0)
    samples = np.column_stack(
        [
            rng.uniform(0.1, 1.0, 1000),  # mass_ratio in (0, 1]
            rng.uniform(0.0, 2000.0, 1000),  # lambda_1 >= 0
            rng.uniform(0.0, 2000.0, 1000),  # lambda_2 >= 0
        ]
    )
    results = check_out_of_bounds(samples, ["mass_ratio", "lambda_1", "lambda_2"])
    for name, stats in results.items():
        assert stats["pct_oob"] == pytest.approx(0.0), f"Unexpected OOB for {name}"
