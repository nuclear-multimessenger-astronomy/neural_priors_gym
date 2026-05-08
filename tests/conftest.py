"""Shared fixtures for neural_priors_gym tests."""

from pathlib import Path

import pytest
import yaml


FIXTURES_DIR = Path(__file__).parent / "fixtures"
EOS_TEST_NPZ = FIXTURES_DIR / "eos_samples_test.npz"


@pytest.fixture
def eos_npz_path() -> Path:
    """Path to the lightweight test EOS npz (10 curves, MTOV > 2 Msun)."""
    return EOS_TEST_NPZ


@pytest.fixture
def small_eos_npz() -> Path:
    """Same lightweight test EOS npz, used wherever a small EOS file is needed."""
    return EOS_TEST_NPZ


@pytest.fixture
def small_config_yaml(tmp_path: Path, small_eos_npz: Path) -> Path:
    """Minimal training config YAML for fast tests (5 epochs, 500 samples)."""
    config = {
        "output_dir": str(tmp_path / "outdir"),
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
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
            "num_epochs": 5,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "max_patience": 5,
            "validation_split": 0.2,
            "scale_input": True,
            "n_samples": 500,
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def bilby_prior_file() -> Path:
    """Path to the synthetic bilby prior fixture file."""
    return FIXTURES_DIR / "bns_test.prior"
