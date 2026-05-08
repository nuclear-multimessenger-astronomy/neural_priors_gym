"""Tests for lambda_tilde/delta_lambda_tilde output names and log_lambda preprocessing."""

import numpy as np
import pytest
import yaml
from pathlib import Path

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.config.schemas.lambdas import LambdaConfig
from neural_priors_gym.config.schemas.masses import UniformMassConfig
from neural_priors_gym.data.generator import (
    TrainingDataGenerator,
    _build_lambda_quantities,
)
from neural_priors_gym.data.lambdas.interpolator import EOSLambdaInterpolator
from neural_priors_gym.data.masses.uniform import UniformMassGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bns_tilde_config_yaml(tmp_path: Path, small_eos_npz: Path) -> Path:
    """BNS config that trains on lambda_tilde / delta_lambda_tilde."""
    config = {
        "output_dir": str(tmp_path / "outdir"),
        "source_type": "bns",
        "masses": {
            "type": "uniform",
            "m_min": 1.0,
        },
        "lambdas": {
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
            "parameter_names": [
                "mass_1_source",
                "mass_2_source",
                "lambda_tilde",
                "delta_lambda_tilde",
            ],
            "num_epochs": 2,
            "batch_size": 32,
            "n_samples": 100,
        },
    }
    path = tmp_path / "config_tilde.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def log_lambda_config_yaml(tmp_path: Path, small_eos_npz: Path) -> Path:
    """BNS config with log_lambda=True."""
    config = {
        "output_dir": str(tmp_path / "outdir"),
        "source_type": "bns",
        "masses": {
            "type": "uniform",
            "m_min": 1.0,
        },
        "lambdas": {
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
            "parameter_names": [
                "mass_1_source",
                "mass_2_source",
                "lambda_1",
                "lambda_2",
            ],
            "log_lambda": True,
            "num_epochs": 2,
            "batch_size": 32,
            "n_samples": 100,
            "scale_input": True,
        },
    }
    path = tmp_path / "config_log.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ---------------------------------------------------------------------------
# Unit tests: _build_lambda_quantities
# ---------------------------------------------------------------------------


def test_build_lambda_quantities_keys() -> None:
    m1 = np.ones(10) * 1.5
    m2 = np.ones(10) * 1.2
    l1 = np.ones(10) * 500.0
    l2 = np.ones(10) * 800.0
    q = _build_lambda_quantities(m1, m2, l1, l2)
    assert set(q.keys()) == {
        "lambda_1",
        "lambda_2",
        "lambda_tilde",
        "delta_lambda_tilde",
    }


def test_lambda_tilde_positive_for_physical_values() -> None:
    """lambda_tilde should be positive for physical BNS systems."""
    rng = np.random.default_rng(0)
    m1 = rng.uniform(1.2, 2.0, 100)
    m2 = rng.uniform(1.0, m1)
    l1 = rng.uniform(100, 1000, 100)
    l2 = rng.uniform(200, 2000, 100)
    q = _build_lambda_quantities(m1, m2, l1, l2)
    assert np.all(q["lambda_tilde"] > 0)


def test_lambda_tilde_agrees_with_bilby() -> None:
    """Cross-check our formula against bilby's implementation."""
    from bilby.gw.conversion import (
        lambda_1_lambda_2_to_lambda_tilde,
        lambda_1_lambda_2_to_delta_lambda_tilde,
    )

    m1 = np.array([1.5, 1.4, 1.8])
    m2 = np.array([1.2, 1.3, 1.1])
    l1 = np.array([400.0, 600.0, 200.0])
    l2 = np.array([700.0, 500.0, 1000.0])

    q = _build_lambda_quantities(m1, m2, l1, l2)

    bilby_tilde = lambda_1_lambda_2_to_lambda_tilde(l1, l2, m1, m2)
    bilby_delta = lambda_1_lambda_2_to_delta_lambda_tilde(l1, l2, m1, m2)

    np.testing.assert_allclose(q["lambda_tilde"], bilby_tilde, rtol=1e-6)
    np.testing.assert_allclose(q["delta_lambda_tilde"], bilby_delta, rtol=1e-6)


def test_equal_masses_symmetry() -> None:
    """For equal masses, lambda_tilde = lambda and delta_lambda_tilde = 0.

    With m1 = m2: eta = 1/4, sqrt(1 - 4*eta) = 0, lambda_1 - lambda_2 = 0.
    lambda_tilde = (8/13) * (1 + 7/4 - 31/16) * 2*lambda = (8/13) * (13/16) * 2*lambda = lambda.
    """
    m = np.array([1.4])
    l = np.array([500.0])
    q = _build_lambda_quantities(m, m, l, l)
    np.testing.assert_allclose(q["lambda_tilde"], l, rtol=1e-6)
    np.testing.assert_allclose(q["delta_lambda_tilde"], np.zeros(1), atol=1e-10)


# ---------------------------------------------------------------------------
# Integration tests: tilde params via config
# ---------------------------------------------------------------------------


def test_tilde_config_loads(bns_tilde_config_yaml: Path) -> None:
    config = load_config(bns_tilde_config_yaml)
    assert "lambda_tilde" in config.training.parameter_names
    assert "delta_lambda_tilde" in config.training.parameter_names


def test_tilde_generate_keys(bns_tilde_config_yaml: Path) -> None:
    config = load_config(bns_tilde_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    # Generator returns all quantities; all lambda types are present
    assert "lambda_tilde" in data
    assert "delta_lambda_tilde" in data
    assert "lambda_1" in data
    assert "lambda_2" in data


def test_tilde_lambda_tilde_positive(bns_tilde_config_yaml: Path) -> None:
    config = load_config(bns_tilde_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["lambda_tilde"] > 0)


def test_tilde_and_component_mixed(small_eos_npz: Path) -> None:
    """Generator returns all lambda quantities including lambda_1 and lambda_tilde."""
    mass_config = UniformMassConfig(m_min=1.0)
    lambda_interp = EOSLambdaInterpolator(LambdaConfig(eos_path=str(small_eos_npz)))
    gen = TrainingDataGenerator(
        mass_generator=UniformMassGenerator(mass_config),
        lambda_interpolator=lambda_interp,
        n_samples=50,
        source_type="bns",
    )
    data = gen.generate()
    assert "lambda_1" in data
    assert "lambda_tilde" in data
    assert len(data["lambda_1"]) == 50


def test_tilde_available_for_nsbh(small_eos_npz: Path) -> None:
    """lambda_tilde is available for NSBH (computed with lambda_1=0)."""
    mass_config = UniformMassConfig(m_min=1.0)
    lambda_interp = EOSLambdaInterpolator(LambdaConfig(eos_path=str(small_eos_npz)))
    gen = TrainingDataGenerator(
        mass_generator=UniformMassGenerator(mass_config),
        lambda_interpolator=lambda_interp,
        n_samples=10,
        source_type="nsbh",
    )
    data = gen.generate()
    assert "lambda_tilde" in data
    assert len(data["lambda_tilde"]) == 10


# ---------------------------------------------------------------------------
# Tests: log_lambda
# ---------------------------------------------------------------------------


def test_log_lambda_config_loads(log_lambda_config_yaml: Path) -> None:
    config = load_config(log_lambda_config_yaml)
    assert config.training.log_lambda is True


def test_log_lambda_npz_stores_raw(
    log_lambda_config_yaml: Path, tmp_path: Path
) -> None:
    """The saved npz must contain raw (non-log) lambdas regardless of log_lambda."""
    config = load_config(log_lambda_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    npz_path = tmp_path / "data.npz"
    gen.save(npz_path)
    loaded = np.load(npz_path)
    # Raw lambdas are always > LAMBDA_CLIP_MIN = 1e-4; log(1e-4) ≈ -9.2
    # so if values are positive we know they're raw, not log-transformed
    assert np.all(loaded["lambda_1"] > 0)
    assert np.all(loaded["lambda_2"] > 0)


def test_log_lambda_applied_before_training(
    log_lambda_config_yaml: Path, tmp_path: Path
) -> None:
    """With log_lambda=True the flow should train without error."""
    from neural_priors_gym.training.trainer import FlowTrainer

    config = load_config(log_lambda_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()

    parameter_names = config.training.parameter_names
    trainer = FlowTrainer(config)
    flow, train_losses, val_losses, scaler = trainer.train(
        data, parameter_names, output_dir=tmp_path
    )
    assert len(train_losses) > 0
    # Scaler should have been fitted on log-lambda values: verify range [0, 1]
    import numpy as np

    x_full = np.column_stack([data[n] for n in parameter_names])
    # log-transform lambdas for comparison (indices 2 and 3 are lambda_1, lambda_2)
    x_full[:, 2] = np.log(x_full[:, 2])
    x_full[:, 3] = np.log(x_full[:, 3])
    if scaler is not None:
        x_scaled = scaler.transform(x_full)
        assert np.all(x_scaled >= -1e-6)
        assert np.all(x_scaled <= 1.0 + 1e-6)


def test_log_lambda_false_by_default(small_eos_npz: Path, tmp_path: Path) -> None:
    """Default training config has log_lambda=False."""
    config_data = {
        "output_dir": str(tmp_path),
        "masses": {
            "type": "uniform",
            "m_min": 1.0,
        },
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {
            "parameter_names": [
                "mass_1_source",
                "mass_2_source",
                "lambda_1",
                "lambda_2",
            ]
        },
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)
    config = load_config(path)
    assert config.training.log_lambda is False
