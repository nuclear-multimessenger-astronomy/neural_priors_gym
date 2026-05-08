"""Integration test for the CLI training pipeline."""

import pytest
from pathlib import Path

from neural_priors_gym.cli import main
from neural_priors_gym.config.parser import load_config


@pytest.mark.slow
def test_cli_full_pipeline(small_config_yaml: Path) -> None:
    """Run the full training pipeline end-to-end with a small config.

    Checks that all expected output files are created.
    """
    config = load_config(small_config_yaml)
    output_dir = Path(config.output_dir)

    main(config)

    assert (output_dir / "training_data.npz").exists(), "training_data.npz not created"
    assert (output_dir / "model" / "model.pt").exists(), "model.pt not created"
    assert (
        output_dir / "model" / "model_kwargs.json"
    ).exists(), "model_kwargs.json not created"
    assert (output_dir / "training_loss.pdf").exists(), "training_loss.pdf not created"
    assert (output_dir / "corner_plot.pdf").exists(), "corner_plot.pdf not created"


@pytest.mark.slow
def test_cli_training_data_correct(small_config_yaml: Path) -> None:
    """Check that training data has the expected keys and is finite."""
    import numpy as np

    config = load_config(small_config_yaml)
    main(config)

    output_dir = Path(config.output_dir)
    data = np.load(output_dir / "training_data.npz")
    assert "mass_1_source" in data
    assert "mass_2_source" in data
    assert "lambda_1" in data
    assert "lambda_2" in data

    for key in data.files:
        assert np.all(np.isfinite(data[key])), f"Non-finite values in {key}"
