"""Tests for parameter_names in training config and custom data_path workflow."""

import numpy as np
import pytest
import yaml
from pathlib import Path

from neural_priors_gym.config.parser import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def custom_npz(tmp_path: Path) -> Path:
    """A minimal .npz file with four known columns."""
    rng = np.random.default_rng(0)
    n = 200
    path = tmp_path / "custom_data.npz"
    np.savez(
        path,
        mass_1_source=rng.uniform(1.2, 2.0, n),
        mass_2_source=rng.uniform(1.0, 1.8, n),
        lambda_1=rng.uniform(100, 1000, n),
        lambda_2=rng.uniform(100, 1000, n),
        extra_column=rng.uniform(0, 1, n),
    )
    return path


@pytest.fixture
def custom_data_config_yaml(tmp_path: Path, custom_npz: Path) -> Path:
    """Config that loads training data from a pre-generated npz file."""
    cfg = {
        "output_dir": str(tmp_path / "outdir"),
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "data_path": str(custom_npz),
            "parameter_names": [
                "mass_1_source",
                "mass_2_source",
                "lambda_1",
                "lambda_2",
            ],
            "num_epochs": 2,
            "batch_size": 32,
            "n_samples": 200,
        },
    }
    path = tmp_path / "config_custom.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# Tests: parameter_names in training config
# ---------------------------------------------------------------------------


def test_parameter_names_required(tmp_path: Path, small_eos_npz: Path) -> None:
    """Config without training.parameter_names should fail validation."""
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {"type": "uniform", "m_min": 1.0},
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {},  # missing parameter_names
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError):
        load_config(path)


def test_parameter_names_in_training_config(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    assert config.training.parameter_names == [
        "mass_1_source",
        "mass_2_source",
        "lambda_1",
        "lambda_2",
    ]


def test_parameter_names_chirp_mass(tmp_path: Path, small_eos_npz: Path) -> None:
    """parameter_names can select derived quantities like chirp_mass_source."""
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {"type": "uniform", "m_min": 1.0},
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {
            "parameter_names": [
                "chirp_mass_source",
                "mass_ratio",
                "lambda_1",
                "lambda_2",
            ],
            "n_samples": 50,
            "num_epochs": 1,
            "batch_size": 20,
            "max_patience": 1,
        },
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = load_config(path)
    assert "chirp_mass_source" in config.training.parameter_names


def test_masses_and_lambdas_have_no_parameter_names(
    small_config_yaml: Path,
) -> None:
    """After the refactor, mass and lambda configs no longer carry parameter_names."""
    config = load_config(small_config_yaml)
    assert not hasattr(config.masses, "parameter_names")
    assert config.lambdas is not None
    assert not hasattr(config.lambdas, "parameter_names")


# ---------------------------------------------------------------------------
# Tests: either masses+lambdas or data_path must be provided
# ---------------------------------------------------------------------------


def test_both_masses_lambdas_and_data_path_raises(
    tmp_path: Path, small_eos_npz: Path, custom_npz: Path
) -> None:
    """Providing both generation config and data_path should fail."""
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {"type": "uniform", "m_min": 1.0},
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {"backend": "glasflow"},
        "training": {
            "data_path": str(custom_npz),
            "parameter_names": ["mass_1_source", "lambda_1"],
        },
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError, match="not both"):
        load_config(path)


def test_neither_masses_lambdas_nor_data_path_raises(tmp_path: Path) -> None:
    """Providing neither generation config nor data_path should fail."""
    cfg = {
        "output_dir": str(tmp_path),
        "flow": {"backend": "glasflow"},
        "training": {"parameter_names": ["mass_1_source", "lambda_1"]},
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError):
        load_config(path)


def test_only_masses_without_lambdas_raises(tmp_path: Path) -> None:
    """Providing masses without lambdas should fail."""
    cfg = {
        "output_dir": str(tmp_path),
        "masses": {"type": "uniform", "m_min": 1.0},
        "flow": {"backend": "glasflow"},
        "training": {"parameter_names": ["mass_1_source", "lambda_1"]},
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError, match="both"):
        load_config(path)


# ---------------------------------------------------------------------------
# Tests: custom data_path workflow
# ---------------------------------------------------------------------------


def test_custom_data_config_loads(custom_data_config_yaml: Path) -> None:
    config = load_config(custom_data_config_yaml)
    assert config.training.data_path is not None
    assert Path(config.training.data_path).exists()
    assert config.masses is None
    assert config.lambdas is None


def test_custom_data_path_resolved_to_absolute(
    tmp_path: Path, custom_npz: Path
) -> None:
    """Relative data_path in YAML is resolved to absolute."""
    # Put the npz one level above the config
    import shutil

    npz_copy = tmp_path / "data.npz"
    shutil.copy(custom_npz, npz_copy)

    config_dir = tmp_path / "run"
    config_dir.mkdir()
    cfg = {
        "output_dir": "./outdir",
        "flow": {"backend": "glasflow"},
        "training": {
            "data_path": "../data.npz",
            "parameter_names": ["mass_1_source", "lambda_1"],
        },
    }
    path = config_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = load_config(path)
    assert config.training.data_path is not None
    assert Path(config.training.data_path).is_absolute()


def test_custom_data_parameter_names_validated(
    tmp_path: Path, custom_npz: Path
) -> None:
    """Requesting a key not in the npz file raises a clear error."""
    from neural_priors_gym.cli import main

    cfg = {
        "output_dir": str(tmp_path / "outdir"),
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "data_path": str(custom_npz),
            "parameter_names": ["mass_1_source", "nonexistent_param"],
            "num_epochs": 1,
            "batch_size": 32,
            "n_samples": 200,
        },
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = load_config(path)
    with pytest.raises(ValueError, match="nonexistent_param"):
        main(config)


def test_custom_data_trains_flow(custom_data_config_yaml: Path, tmp_path: Path) -> None:
    """End-to-end: load custom npz, train flow, save outputs."""
    from neural_priors_gym.cli import main

    config = load_config(custom_data_config_yaml)
    main(config)

    outdir = Path(config.output_dir)
    assert (outdir / "model" / "scaler.gz").exists()


def test_custom_data_subset_of_columns(tmp_path: Path, custom_npz: Path) -> None:
    """User can select a subset of columns from a larger npz file."""
    from neural_priors_gym.cli import main

    cfg = {
        "output_dir": str(tmp_path / "outdir"),
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "data_path": str(custom_npz),
            # Only use 2 of the 5 columns in the npz
            "parameter_names": ["mass_1_source", "lambda_1"],
            "num_epochs": 2,
            "batch_size": 32,
            "n_samples": 200,
        },
    }
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    config = load_config(path)
    main(config)
    outdir = Path(config.output_dir)
    assert (outdir / "model" / "scaler.gz").exists()
