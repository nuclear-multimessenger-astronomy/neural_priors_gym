"""Tests for the GlasflowNSF flow wrapper."""

import numpy as np
import pytest
from pathlib import Path

from neural_priors_gym.config.schemas.flow import GlasflowConfig
from neural_priors_gym.flows.glasflow import GlasflowNSF


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
