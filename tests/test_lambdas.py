"""Tests for EOSLambdaInterpolator."""

import numpy as np
import pytest
from pathlib import Path

from neural_priors_gym.config.schemas.lambdas import LambdaConfig
from neural_priors_gym.data.lambdas.interpolator import EOSLambdaInterpolator


@pytest.fixture
def lambda_config(small_eos_npz: Path) -> LambdaConfig:
    return LambdaConfig(eos_path=str(small_eos_npz))


def test_interpolator_loads(lambda_config: LambdaConfig) -> None:
    interp = EOSLambdaInterpolator(lambda_config)
    assert interp.masses_EOS.shape[0] > 0
    assert interp.lambdas_EOS.shape == interp.masses_EOS.shape


def test_interpolator_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        config = LambdaConfig(eos_path="/nonexistent/path.npz")
        EOSLambdaInterpolator(config)


def test_interpolate_shape(lambda_config: LambdaConfig) -> None:
    interp = EOSLambdaInterpolator(lambda_config)
    masses = np.column_stack(
        [
            np.full(50, 1.3),
            np.full(50, 1.1),
        ]
    )
    lambdas = interp.interpolate(masses)
    assert lambdas.shape == (50, 2)


def test_interpolate_positive(lambda_config: LambdaConfig) -> None:
    interp = EOSLambdaInterpolator(lambda_config)
    masses = np.column_stack(
        [
            np.random.uniform(1.0, 1.8, 100),
            np.random.uniform(1.0, 1.5, 100),
        ]
    )
    lambdas = interp.interpolate(masses)
    assert np.all(lambdas > 0)


def test_negative_lambdas_filtered(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_eos, n_mass = 5, 20
    masses = np.zeros((n_eos, n_mass))
    lambdas = np.zeros((n_eos, n_mass))
    for i in range(n_eos):
        masses[i] = np.linspace(0.5, 2.2, n_mass)
        lambdas[i] = np.exp(-np.linspace(0, 5, n_mass)) * 1000

    # Inject one bad EOS with negative lambdas
    lambdas[0, 5] = -1.0

    path = tmp_path / "eos.npz"
    np.savez(path, masses_EOS=masses, Lambdas_EOS=lambdas)
    config = LambdaConfig(eos_path=str(path))
    interp = EOSLambdaInterpolator(config)
    assert len(interp.masses_EOS) == n_eos - 1


def test_low_mtov_does_not_filter(tmp_path: Path) -> None:
    """EOS samples with MTOV < 2.0 Msun should be kept (only warned about)."""
    n_eos, n_mass = 3, 20
    masses = np.zeros((n_eos, n_mass))
    lambdas = np.zeros((n_eos, n_mass))
    for i in range(n_eos):
        masses[i] = np.linspace(0.5, 1.5, n_mass)  # MTOV < 2.0
        lambdas[i] = np.exp(-np.linspace(0, 5, n_mass)) * 1000

    path = tmp_path / "eos.npz"
    np.savez(path, masses_EOS=masses, Lambdas_EOS=lambdas)
    config = LambdaConfig(eos_path=str(path))
    interp = EOSLambdaInterpolator(config)
    assert len(interp.masses_EOS) == n_eos


def test_mtov_array_property(lambda_config: LambdaConfig) -> None:
    interp = EOSLambdaInterpolator(lambda_config)
    arr = interp.mtov_array
    assert arr.ndim == 1
    assert len(arr) == len(interp.masses_EOS)
    assert np.all(arr > 0)


def test_interpolate_mass_beyond_eos_grid_is_finite_and_clipped(tmp_path: Path) -> None:
    """Masses above the EOS grid max use flat extrapolation and are clipped to LAMBDA_CLIP_MIN."""
    from neural_priors_gym.data.lambdas.interpolator import LAMBDA_CLIP_MIN

    n_eos, n_mass = 5, 20
    masses = np.zeros((n_eos, n_mass))
    lambdas = np.zeros((n_eos, n_mass))
    for i in range(n_eos):
        masses[i] = np.linspace(0.5, 2.0, n_mass)
        lambdas[i] = np.exp(-np.linspace(0, 5, n_mass)) * 1000

    path = tmp_path / "eos.npz"
    np.savez(path, masses_EOS=masses, Lambdas_EOS=lambdas)
    config = LambdaConfig(eos_path=str(path))
    interp = EOSLambdaInterpolator(config)

    # Request masses well above every EOS grid maximum (2.0 Msun)
    beyond = np.full((10, 2), 5.0)
    result = interp.interpolate(beyond)

    assert result.shape == (10, 2)
    assert np.all(np.isfinite(result))
    assert np.all(result >= LAMBDA_CLIP_MIN)


def test_interpolate_mass_below_eos_grid_is_finite_and_clipped(tmp_path: Path) -> None:
    """Masses below the EOS grid min use flat extrapolation; result must be finite and ≥ clip."""
    from neural_priors_gym.data.lambdas.interpolator import LAMBDA_CLIP_MIN

    n_eos, n_mass = 5, 20
    masses = np.zeros((n_eos, n_mass))
    lambdas = np.zeros((n_eos, n_mass))
    for i in range(n_eos):
        masses[i] = np.linspace(1.0, 2.2, n_mass)
        lambdas[i] = np.exp(-np.linspace(0, 5, n_mass)) * 1000

    path = tmp_path / "eos_low.npz"
    np.savez(path, masses_EOS=masses, Lambdas_EOS=lambdas)
    config = LambdaConfig(eos_path=str(path))
    interp = EOSLambdaInterpolator(config)

    below = np.full((10, 2), 0.1)
    result = interp.interpolate(below)

    assert np.all(np.isfinite(result))
    assert np.all(result >= LAMBDA_CLIP_MIN)
