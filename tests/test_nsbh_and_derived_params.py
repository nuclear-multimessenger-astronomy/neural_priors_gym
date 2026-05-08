"""Tests for NSBH source type and derived mass parameter names."""

import numpy as np
import pytest
import yaml
from pathlib import Path

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.config.schemas.lambdas import LambdaConfig
from neural_priors_gym.config.schemas.masses import (
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    UniformMassConfig,
)
from neural_priors_gym.data.generator import (
    TrainingDataGenerator,
    _build_mass_quantities,
)
from neural_priors_gym.data.lambdas.interpolator import EOSLambdaInterpolator
from neural_priors_gym.data.masses.double_gaussian import DoubleGaussianMassGenerator
from neural_priors_gym.data.masses.gaussian import GaussianMassGenerator
from neural_priors_gym.data.masses.nsbh import NSBHMassGenerator
from neural_priors_gym.data.masses.uniform import UniformMassGenerator


PARAM_NAMES = ["mass_1_source", "mass_2_source"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def masses_EOS() -> np.ndarray:
    rng = np.random.default_rng(42)
    n_eos, n_pts = 10, 50
    arr = np.zeros((n_eos, n_pts))
    for i in range(n_eos):
        arr[i] = np.linspace(0.5, 2.0 + rng.uniform(0.1, 0.5), n_pts)
    return arr


@pytest.fixture
def lambda_config(small_eos_npz: Path) -> LambdaConfig:
    return LambdaConfig(eos_path=str(small_eos_npz))


@pytest.fixture
def nsbh_config_yaml(tmp_path: Path, small_eos_npz: Path) -> Path:
    """Minimal NSBH config YAML: 1 lambda (lambda_2 only)."""
    config = {
        "output_dir": str(tmp_path / "outdir"),
        "source_type": "nsbh",
        "m_max_bh": 5.0,
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
        },
        "lambdas": {
            "parameter_names": ["lambda_2"],
            "eos_path": str(small_eos_npz),
        },
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 32,
            "n_samples": 100,
        },
    }
    path = tmp_path / "config_nsbh.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def derived_mass_config_yaml(tmp_path: Path, small_eos_npz: Path) -> Path:
    """Config with double Gaussian masses but output in chirp_mass_source / mass_ratio."""
    config = {
        "output_dir": str(tmp_path / "outdir"),
        "masses": {
            "type": "double_gaussian",
            "parameter_names": ["chirp_mass_source", "mass_ratio"],
            "m_min": 1.0,
            "mean_1": 1.34,
            "std_1": 0.07,
            "mean_2": 1.80,
            "std_2": 0.21,
            "weight": 0.65,
        },
        "lambdas": {
            "parameter_names": ["lambda_1", "lambda_2"],
            "eos_path": str(small_eos_npz),
        },
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "num_epochs": 2,
            "batch_size": 32,
            "n_samples": 100,
        },
    }
    path = tmp_path / "config_derived.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ---------------------------------------------------------------------------
# Unit tests: _build_mass_quantities
# ---------------------------------------------------------------------------


def test_build_mass_quantities_keys() -> None:
    m1 = np.ones(10) * 1.5
    m2 = np.ones(10) * 1.2
    q = _build_mass_quantities(m1, m2)
    assert "mass_1_source" in q
    assert "mass_2_source" in q
    assert "chirp_mass_source" in q
    assert "mass_ratio" in q


def test_mass_ratio_values() -> None:
    m1 = np.array([1.4, 2.0])
    m2 = np.array([1.0, 1.0])
    q = _build_mass_quantities(m1, m2)
    np.testing.assert_allclose(q["mass_ratio"], m2 / m1)


# ---------------------------------------------------------------------------
# Unit tests: NSBHMassGenerator
# ---------------------------------------------------------------------------


def test_nsbh_m1_above_mtov(masses_EOS: np.ndarray) -> None:
    ns_gen = UniformMassGenerator(
        UniformMassConfig(parameter_names=PARAM_NAMES, m_min=1.0)
    )
    gen = NSBHMassGenerator(ns_gen, m_max_bh=5.0)
    mtov = masses_EOS.max(axis=1)
    result = gen.generate(200, mtov)
    # m1 must be above MTOV of its EOS; check it's above the minimum MTOV
    assert np.all(result[:, 0] >= mtov.min())


def test_nsbh_m2_below_mtov(masses_EOS: np.ndarray) -> None:
    ns_gen = UniformMassGenerator(
        UniformMassConfig(parameter_names=PARAM_NAMES, m_min=1.0)
    )
    gen = NSBHMassGenerator(ns_gen, m_max_bh=5.0)
    mtov = masses_EOS.max(axis=1)
    result = gen.generate(200, mtov)
    assert np.all(result[:, 1] <= mtov.max() + 1e-9)


def test_nsbh_m_max_bh_exceeded_raises(masses_EOS: np.ndarray) -> None:
    ns_gen = UniformMassGenerator(
        UniformMassConfig(parameter_names=PARAM_NAMES, m_min=1.0)
    )
    # m_max_bh below every MTOV should raise
    gen = NSBHMassGenerator(ns_gen, m_max_bh=1.0)
    with pytest.raises(ValueError, match="m_max_bh"):
        gen.generate(10, masses_EOS.max(axis=1))


def test_nsbh_generator_shape(masses_EOS: np.ndarray) -> None:
    ns_gen = GaussianMassGenerator(
        GaussianMassConfig(parameter_names=PARAM_NAMES, m_min=1.0, mean=1.33, std=0.09)
    )
    gen = NSBHMassGenerator(ns_gen, m_max_bh=5.0)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)


def test_nsbh_double_gaussian(masses_EOS: np.ndarray) -> None:
    ns_gen = DoubleGaussianMassGenerator(
        DoubleGaussianMassConfig(
            parameter_names=PARAM_NAMES,
            m_min=1.0,
            mean_1=1.34,
            std_1=0.07,
            mean_2=1.80,
            std_2=0.21,
            weight=0.65,
        )
    )
    gen = NSBHMassGenerator(ns_gen, m_max_bh=5.0)
    result = gen.generate(50, masses_EOS.max(axis=1))
    assert result.shape == (50, 2)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Integration tests: NSBH full pipeline via config YAML
# ---------------------------------------------------------------------------


def test_nsbh_config_loads(nsbh_config_yaml: Path) -> None:
    config = load_config(nsbh_config_yaml)
    assert config.source_type == "nsbh"
    assert config.lambdas.parameter_names == ["lambda_2"]


def test_nsbh_generate_only_lambda2(nsbh_config_yaml: Path) -> None:
    config = load_config(nsbh_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert "lambda_2" in data
    assert "lambda_1" not in data


def test_nsbh_generate_length(nsbh_config_yaml: Path) -> None:
    config = load_config(nsbh_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    n = config.training.n_samples
    for arr in data.values():
        assert len(arr) == n


def test_nsbh_lambda2_positive(nsbh_config_yaml: Path) -> None:
    config = load_config(nsbh_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["lambda_2"] > 0)


# ---------------------------------------------------------------------------
# Integration tests: derived mass parameter names
# ---------------------------------------------------------------------------


def test_derived_params_config_loads(derived_mass_config_yaml: Path) -> None:
    config = load_config(derived_mass_config_yaml)
    assert config.masses.parameter_names == ["chirp_mass_source", "mass_ratio"]


def test_derived_params_generate_keys(derived_mass_config_yaml: Path) -> None:
    config = load_config(derived_mass_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert "chirp_mass_source" in data
    assert "mass_ratio" in data
    assert "lambda_1" in data
    assert "lambda_2" in data
    # component masses should NOT appear since they're not in parameter_names
    assert "mass_1_source" not in data
    assert "mass_2_source" not in data


def test_derived_params_mass_ratio_in_range(derived_mass_config_yaml: Path) -> None:
    config = load_config(derived_mass_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    # mass_ratio = m2/m1 with m1 >= m2 so ratio must be in (0, 1]
    assert np.all(data["mass_ratio"] > 0)
    assert np.all(data["mass_ratio"] <= 1.0 + 1e-9)


def test_derived_params_chirp_mass_positive(derived_mass_config_yaml: Path) -> None:
    config = load_config(derived_mass_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["chirp_mass_source"] > 0)


def test_unknown_mass_param_raises(small_eos_npz: Path) -> None:
    """Requesting a name not in the supported set should raise."""
    mass_config = UniformMassConfig(parameter_names=["nonsense_param"], m_min=1.0)
    lambda_interp = EOSLambdaInterpolator(LambdaConfig(eos_path=str(small_eos_npz)))
    gen = TrainingDataGenerator(
        mass_generator=UniformMassGenerator(mass_config),
        lambda_interpolator=lambda_interp,
        n_samples=10,
        mass_parameter_names=["nonsense_param"],
        lambda_parameter_names=["lambda_1", "lambda_2"],
    )
    with pytest.raises(ValueError, match="nonsense_param"):
        gen.generate()


def test_double_gaussian_with_component_masses(small_eos_npz: Path) -> None:
    """Double Gaussian population trained on component masses still works."""
    mass_config = DoubleGaussianMassConfig(
        parameter_names=["mass_1_source", "mass_2_source"],
        m_min=1.0,
        mean_1=1.34,
        std_1=0.07,
        mean_2=1.80,
        std_2=0.21,
        weight=0.65,
    )
    lambda_interp = EOSLambdaInterpolator(LambdaConfig(eos_path=str(small_eos_npz)))
    gen = TrainingDataGenerator(
        mass_generator=DoubleGaussianMassGenerator(mass_config),
        lambda_interpolator=lambda_interp,
        n_samples=50,
        mass_parameter_names=["mass_1_source", "mass_2_source"],
        lambda_parameter_names=["lambda_1", "lambda_2"],
    )
    data = gen.generate()
    assert "mass_1_source" in data
    assert "mass_2_source" in data
    assert np.all(data["mass_1_source"] >= data["mass_2_source"])
