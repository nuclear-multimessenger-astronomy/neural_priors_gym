"""Tests for YAML config loading and Pydantic validation."""

import pytest
import yaml
from pathlib import Path

from neural_priors_gym.config import load_config
from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.masses import (
    UniformMassConfig,
    GaussianMassConfig,
)


def test_load_uniform_config(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    assert isinstance(config, TrainingConfig)
    assert isinstance(config.masses, UniformMassConfig)
    assert config.masses.m_min == 1.0
    assert config.training.num_epochs == 5
    assert config.training.n_samples == 500


def test_load_gaussian_config(tmp_path: Path, small_eos_npz: Path) -> None:
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {
            "type": "gaussian",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
            "mean": 1.33,
            "std": 0.09,
        },
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = load_config(path)
    assert isinstance(config.masses, GaussianMassConfig)
    assert config.masses.mean == pytest.approx(1.33)


def test_load_double_gaussian_config(tmp_path: Path, small_eos_npz: Path) -> None:
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {
            "type": "double_gaussian",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
            "mean_1": 1.34,
            "std_1": 0.07,
            "mean_2": 1.80,
            "std_2": 0.21,
            "weight": 0.65,
        },
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = load_config(path)
    from neural_priors_gym.config.schemas.masses import DoubleGaussianMassConfig

    assert isinstance(config.masses, DoubleGaussianMassConfig)


def test_invalid_extra_field_raises(tmp_path: Path, small_eos_npz: Path) -> None:
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
            "unknown_field": 42,
        },
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError, match="unknown_field"):
        load_config(path)


def test_missing_eos_path_raises(tmp_path: Path) -> None:
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
        },
        "lambdas": {},
        "flow": {"backend": "glasflow"},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError):
        load_config(path)


def test_relative_eos_path_resolved(tmp_path: Path, small_eos_npz: Path) -> None:
    import shutil

    # Place the EOS file one level above the config directory
    local_eos = tmp_path / "eos_samples.npz"
    shutil.copy(small_eos_npz, local_eos)

    config_dir = tmp_path / "run"
    config_dir.mkdir()

    cfg = {
        "output_dir": "./outdir",
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
        },
        "lambdas": {"eos_path": "../eos_samples.npz"},
        "flow": {"backend": "glasflow"},
        "training": {},
    }
    path = config_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    config = load_config(path)
    assert Path(config.lambdas.eos_path).is_absolute()
    assert Path(config.lambdas.eos_path).exists()
