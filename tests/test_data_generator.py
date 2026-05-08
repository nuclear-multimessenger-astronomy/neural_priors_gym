"""Tests for TrainingDataGenerator."""

import numpy as np
from pathlib import Path

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.data.generator import TrainingDataGenerator


def test_generate_returns_correct_keys(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert "mass_1_source" in data
    assert "mass_2_source" in data
    assert "lambda_1" in data
    assert "lambda_2" in data


def test_generate_correct_length(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    n = config.training.n_samples
    for arr in data.values():
        assert len(arr) == n


def test_generate_lambdas_positive(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["lambda_1"] > 0)
    assert np.all(data["lambda_2"] > 0)


def test_generate_m1_ge_m2(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    data = gen.generate()
    assert np.all(data["mass_1_source"] >= data["mass_2_source"])


def test_save_creates_npz(small_config_yaml: Path, tmp_path: Path) -> None:
    config = load_config(small_config_yaml)
    gen = TrainingDataGenerator.from_config(config)
    output_path = tmp_path / "training_data.npz"
    gen.save(output_path)
    assert output_path.exists()
    loaded = np.load(output_path)
    assert "mass_1_source" in loaded
