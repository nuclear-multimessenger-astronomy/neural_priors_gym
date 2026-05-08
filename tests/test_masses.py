"""Tests for mass generator classes."""

import numpy as np
import pytest
from pathlib import Path

from neural_priors_gym.config.schemas.masses import (
    BilbyMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    UniformMassConfig,
)
from neural_priors_gym.data.masses import (
    BilbyPriorMassGenerator,
    DoubleGaussianMassGenerator,
    GaussianMassGenerator,
    UniformMassGenerator,
)


PARAM_NAMES = ["mass_1_source", "mass_2_source"]


@pytest.fixture
def masses_EOS() -> np.ndarray:
    """Small synthetic EOS mass grid."""
    rng = np.random.default_rng(1)
    n_eos, n_pts = 10, 50
    arr = np.zeros((n_eos, n_pts))
    for i in range(n_eos):
        arr[i] = np.linspace(0.5, 2.0 + rng.uniform(0, 0.3), n_pts)
    return arr


@pytest.fixture
def uniform_config() -> UniformMassConfig:
    return UniformMassConfig(parameter_names=PARAM_NAMES, m_min=1.0)


@pytest.fixture
def gaussian_config() -> GaussianMassConfig:
    return GaussianMassConfig(
        parameter_names=PARAM_NAMES, m_min=1.0, mean=1.33, std=0.09
    )


@pytest.fixture
def double_gaussian_config() -> DoubleGaussianMassConfig:
    return DoubleGaussianMassConfig(
        parameter_names=PARAM_NAMES,
        m_min=1.0,
        mean_1=1.34,
        std_1=0.07,
        mean_2=1.80,
        std_2=0.21,
        weight=0.65,
    )


def test_uniform_generator_shape(
    uniform_config: UniformMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = UniformMassGenerator(uniform_config)
    result = gen.generate(100, masses_EOS.max(axis=1))
    assert result.shape == (100, 2)


def test_uniform_generator_m1_ge_m2(
    uniform_config: UniformMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = UniformMassGenerator(uniform_config)
    result = gen.generate(200, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_uniform_generator_within_bounds(
    uniform_config: UniformMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = UniformMassGenerator(uniform_config)
    mtov = masses_EOS.max(axis=1)
    result = gen.generate(200, mtov)
    assert np.all(result[:, 1] >= 1.0)
    assert np.all(result[:, 0] <= mtov.max() + 1e-9)


def test_gaussian_generator_shape(
    gaussian_config: GaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = GaussianMassGenerator(gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)


def test_gaussian_generator_m1_ge_m2(
    gaussian_config: GaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = GaussianMassGenerator(gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_double_gaussian_generator_shape(
    double_gaussian_config: DoubleGaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = DoubleGaussianMassGenerator(double_gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)


def test_double_gaussian_generator_m1_ge_m2(
    double_gaussian_config: DoubleGaussianMassConfig, masses_EOS: np.ndarray
) -> None:
    gen = DoubleGaussianMassGenerator(double_gaussian_config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert np.all(result[:, 0] >= result[:, 1])


def test_bilby_generator(bilby_prior_file: Path, masses_EOS: np.ndarray) -> None:
    config = BilbyMassConfig(
        parameter_names=PARAM_NAMES, prior_file=str(bilby_prior_file)
    )
    gen = BilbyPriorMassGenerator(config)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)
    assert np.all(np.isfinite(result))


def test_bilby_generator_missing_file(masses_EOS: np.ndarray) -> None:
    with pytest.raises(FileNotFoundError):
        config = BilbyMassConfig(
            parameter_names=PARAM_NAMES, prior_file="/nonexistent/path.prior"
        )
        BilbyPriorMassGenerator(config)
