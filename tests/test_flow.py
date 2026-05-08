"""Tests for flow backends and FlowBase utilities."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from neural_priors_gym.config.schemas.flow import GlasflowConfig, ZukoMAFConfig
from neural_priors_gym.flows.base import FlowBase
from neural_priors_gym.flows.glasflow import GlasflowNSF
from neural_priors_gym.flows.zuko_maf import ZukoMAF


@pytest.fixture
def small_flow() -> GlasflowNSF:
    config = GlasflowConfig(
        n_transforms=2, n_neurons=16, n_blocks_per_transform=1, num_bins=4
    )
    return GlasflowNSF.from_config(config, n_inputs=4)


def test_flow_creation(small_flow: GlasflowNSF) -> None:
    assert small_flow.n_inputs == 4


def test_flow_sample_shape(small_flow: GlasflowNSF) -> None:
    samples = small_flow.sample(100)
    assert samples.shape == (100, 4)


def test_flow_log_prob_shape(small_flow: GlasflowNSF) -> None:
    x = np.random.randn(50, 4).astype(np.float32)
    lp = small_flow.log_prob(x)
    assert lp.shape == (50,)


def test_flow_save_and_load(small_flow: GlasflowNSF, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    small_flow.save(model_dir)

    assert (model_dir / "model.pt").exists()
    assert (model_dir / "model_kwargs.json").exists()

    loaded = GlasflowNSF.load(model_dir)
    assert loaded.n_inputs == small_flow.n_inputs

    samples = loaded.sample(50)
    assert samples.shape == (50, 4)


def test_flow_log_prob_finite(small_flow: GlasflowNSF) -> None:
    x = np.random.randn(20, 4).astype(np.float32)
    lp = small_flow.log_prob(x)
    assert np.all(np.isfinite(lp))


def test_flow_train_epoch_reduces_loss(tmp_path: Path) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    config = GlasflowConfig(
        n_transforms=2, n_neurons=32, n_blocks_per_transform=1, num_bins=4
    )
    flow = GlasflowNSF.from_config(config, n_inputs=2)
    flow.set_optimizer(learning_rate=1e-3)

    x = torch.randn(200, 2)
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)

    loss_before = flow.eval_epoch(loader)
    for _ in range(3):
        flow.train_epoch(loader)
    loss_after = flow.eval_epoch(loader)

    # After a few updates the loss should not have exploded
    assert np.isfinite(loss_after)


# ---------------------------------------------------------------------------
# FlowBase.load_from_dir
# ---------------------------------------------------------------------------


def test_load_from_dir_glasflow(tmp_path: Path) -> None:
    config = GlasflowConfig(
        n_transforms=2, n_neurons=16, n_blocks_per_transform=1, num_bins=4
    )
    flow = GlasflowNSF.from_config(config, n_inputs=3)
    model_dir = tmp_path / "model"
    flow.save(model_dir)

    loaded = FlowBase.load_from_dir(model_dir)
    assert isinstance(loaded, GlasflowNSF)
    assert loaded.n_inputs == 3
    assert loaded.sample(10).shape == (10, 3)


def test_load_from_dir_zuko(tmp_path: Path) -> None:
    config = ZukoMAFConfig(transforms=2, hidden_features=[16, 16])
    flow = ZukoMAF.from_config(config, n_inputs=3)
    model_dir = tmp_path / "model"
    flow.save(model_dir)

    loaded = FlowBase.load_from_dir(model_dir)
    assert isinstance(loaded, ZukoMAF)
    assert loaded.n_inputs == 3
    assert loaded.sample(10).shape == (10, 3)


def test_load_from_dir_missing_kwargs(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="model_kwargs.json"):
        FlowBase.load_from_dir(model_dir)


def test_load_from_dir_missing_backend_key(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_kwargs.json").write_text(json.dumps({"n_inputs": 2}))
    with pytest.raises(ValueError, match="'backend' key missing"):
        FlowBase.load_from_dir(model_dir)


def test_load_from_dir_unknown_backend(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_kwargs.json").write_text(
        json.dumps({"backend": "nonexistent_backend", "n_inputs": 2})
    )
    with pytest.raises(NotImplementedError, match="nonexistent_backend"):
        FlowBase.load_from_dir(model_dir)


# ---------------------------------------------------------------------------
# FlowBase.compute_log_jacobian_correction
# ---------------------------------------------------------------------------


def test_compute_log_jacobian_correction_known_values() -> None:
    scaler = MagicMock()
    scaler.data_max_ = np.array([2.0, 10.0])
    scaler.data_min_ = np.array([0.0, 0.0])
    # ranges = [2.0, 10.0], correction = -(log(2) + log(10))
    expected = -(np.log(2.0) + np.log(10.0))
    result = FlowBase.compute_log_jacobian_correction(scaler)
    assert np.isclose(result, expected)


def test_compute_log_jacobian_correction_unit_range() -> None:
    scaler = MagicMock()
    scaler.data_max_ = np.array([1.0, 1.0, 1.0])
    scaler.data_min_ = np.array([0.0, 0.0, 0.0])
    # All ranges = 1, so correction = 0
    assert FlowBase.compute_log_jacobian_correction(scaler) == pytest.approx(0.0)
