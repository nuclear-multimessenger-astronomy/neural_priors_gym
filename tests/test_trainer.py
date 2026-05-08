"""Tests for the FlowTrainer."""

import numpy as np
from pathlib import Path

from neural_priors_gym.config.parser import load_config
from neural_priors_gym.training.trainer import FlowTrainer


def test_trainer_runs(small_config_yaml: Path) -> None:
    config = load_config(small_config_yaml)
    trainer = FlowTrainer(config)

    rng = np.random.default_rng(42)
    n = config.training.n_samples
    data = {
        "mass_1_source": rng.uniform(1.0, 2.0, n),
        "mass_2_source": rng.uniform(1.0, 1.8, n),
        "lambda_1": rng.uniform(10, 1000, n),
        "lambda_2": rng.uniform(10, 1000, n),
    }
    parameter_names = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]

    flow, train_losses, val_losses, scaler = trainer.train(data, parameter_names)

    assert len(train_losses) > 0
    assert len(val_losses) == len(train_losses)
    assert scaler is not None
    samples = flow.sample(20)
    assert samples.shape[1] == 4


def test_trainer_saves_scaler(small_config_yaml: Path, tmp_path: Path) -> None:
    config = load_config(small_config_yaml)
    trainer = FlowTrainer(config)

    n = config.training.n_samples
    rng = np.random.default_rng(0)
    data = {
        "mass_1_source": rng.uniform(1.0, 2.0, n),
        "mass_2_source": rng.uniform(1.0, 1.8, n),
        "lambda_1": rng.uniform(10, 1000, n),
        "lambda_2": rng.uniform(10, 1000, n),
    }

    trainer.train(data, list(data.keys()), output_dir=tmp_path)
    assert (tmp_path / "scaler.gz").exists()


def test_trainer_no_scaler_when_disabled(
    small_config_yaml: Path, tmp_path: Path
) -> None:
    import yaml

    with open(small_config_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["scale_input"] = False
    no_scale_path = tmp_path / "config_no_scale.yaml"
    with open(no_scale_path, "w") as f:
        yaml.dump(cfg, f)

    config = load_config(no_scale_path)
    trainer = FlowTrainer(config)
    n = config.training.n_samples
    rng = np.random.default_rng(0)
    data = {
        "mass_1_source": rng.uniform(1.0, 2.0, n),
        "mass_2_source": rng.uniform(1.0, 1.8, n),
        "lambda_1": rng.uniform(10, 1000, n),
        "lambda_2": rng.uniform(10, 1000, n),
    }
    _, _, _, scaler = trainer.train(data, list(data.keys()), output_dir=tmp_path)
    assert scaler is None


def test_early_stopping_fires(tmp_path: Path, small_eos_npz: Path) -> None:
    """With max_patience=1 and many epochs, training must stop before num_epochs."""
    import yaml

    config_data = {
        "output_dir": str(tmp_path / "outdir"),
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
        },
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "num_epochs": 50,
            "batch_size": 64,
            "n_samples": 500,
            "max_patience": 1,
            "scale_input": False,
        },
    }
    path = tmp_path / "config_patience.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(path)
    trainer = FlowTrainer(config)
    rng = np.random.default_rng(7)
    n = config.training.n_samples
    data = {
        "mass_1_source": rng.uniform(1.0, 2.0, n),
        "mass_2_source": rng.uniform(1.0, 1.8, n),
        "lambda_1": rng.uniform(10, 1000, n),
        "lambda_2": rng.uniform(10, 1000, n),
    }
    _, train_losses, _, _ = trainer.train(data, list(data.keys()), output_dir=tmp_path)
    # patience=1 means at most 2 epochs before stopping (one improvement + one failure)
    assert len(train_losses) < config.training.num_epochs


def test_early_stopping_restores_best_checkpoint(
    tmp_path: Path, small_eos_npz: Path
) -> None:
    """The flow returned must correspond to the best val-loss checkpoint, not the last epoch."""
    import yaml

    config_data = {
        "output_dir": str(tmp_path / "outdir"),
        "masses": {
            "type": "uniform",
            "parameter_names": ["mass_1_source", "mass_2_source"],
            "m_min": 1.0,
        },
        "lambdas": {"eos_path": str(small_eos_npz)},
        "flow": {
            "backend": "glasflow",
            "n_transforms": 2,
            "n_neurons": 32,
            "n_blocks_per_transform": 1,
            "num_bins": 4,
        },
        "training": {
            "num_epochs": 10,
            "batch_size": 64,
            "n_samples": 500,
            "max_patience": 2,
            "scale_input": False,
        },
    }
    path = tmp_path / "config_ckpt.yaml"
    with open(path, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(path)
    trainer = FlowTrainer(config)
    rng = np.random.default_rng(0)
    n = config.training.n_samples
    data = {
        "mass_1_source": rng.uniform(1.0, 2.0, n),
        "mass_2_source": rng.uniform(1.0, 1.8, n),
        "lambda_1": rng.uniform(10, 1000, n),
        "lambda_2": rng.uniform(10, 1000, n),
    }
    flow, _, val_losses, _ = trainer.train(data, list(data.keys()), output_dir=tmp_path)

    # The best val loss seen during training must be ≤ any individual epoch's val loss.
    best_seen = min(val_losses)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x = np.column_stack([data[k] for k in data])
    tensor = torch.tensor(x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=64)
    final_eval = flow.eval_epoch(loader)
    # The restored checkpoint should score at or near the best seen validation loss.
    assert final_eval <= best_seen + 0.5
