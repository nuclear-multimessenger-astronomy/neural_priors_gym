"""Tests for the ZukoMAF flow wrapper."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_priors_gym.config.schemas.flow import ZukoMAFConfig
from neural_priors_gym.flows.zuko_maf import ZukoMAF


@pytest.fixture
def small_flow() -> ZukoMAF:
    config = ZukoMAFConfig(transforms=2, hidden_features=[16, 16])
    return ZukoMAF.from_config(config, n_inputs=4)


def test_flow_creation(small_flow: ZukoMAF) -> None:
    assert small_flow.n_inputs == 4


def test_flow_sample_shape(small_flow: ZukoMAF) -> None:
    samples = small_flow.sample(100)
    assert samples.shape == (100, 4)


def test_flow_log_prob_shape(small_flow: ZukoMAF) -> None:
    x = np.random.randn(50, 4).astype(np.float32)
    lp = small_flow.log_prob(x)
    assert lp.shape == (50,)


def test_flow_log_prob_finite(small_flow: ZukoMAF) -> None:
    x = np.random.randn(20, 4).astype(np.float32)
    lp = small_flow.log_prob(x)
    assert np.all(np.isfinite(lp))


def test_flow_save_and_load(small_flow: ZukoMAF, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    small_flow.save(model_dir)

    assert (model_dir / "model.pt").exists()
    assert (model_dir / "model_kwargs.json").exists()

    loaded = ZukoMAF.load(model_dir)
    assert loaded.n_inputs == small_flow.n_inputs

    x = np.random.randn(50, 4).astype(np.float32)
    lp_orig = small_flow.log_prob(x)
    lp_loaded = loaded.log_prob(x)
    np.testing.assert_allclose(lp_orig, lp_loaded, rtol=1e-5)


def test_flow_train_epoch_finite() -> None:
    config = ZukoMAFConfig(transforms=2, hidden_features=[16, 16])
    flow = ZukoMAF.from_config(config, n_inputs=2)
    flow.set_optimizer(learning_rate=1e-3)

    x = torch.randn(200, 2)
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)

    loss_before = flow.eval_epoch(loader)
    for _ in range(3):
        flow.train_epoch(loader)
    loss_after = flow.eval_epoch(loader)

    assert np.isfinite(loss_before)
    assert np.isfinite(loss_after)


def test_flow_eval_epoch_shape() -> None:
    config = ZukoMAFConfig(transforms=2, hidden_features=[16, 16])
    flow = ZukoMAF.from_config(config, n_inputs=3)
    flow.set_optimizer(learning_rate=1e-3)

    x = torch.randn(100, 3)
    loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=False)
    loss = flow.eval_epoch(loader)

    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_flow_config_randperm() -> None:
    config = ZukoMAFConfig(transforms=3, randperm=True, hidden_features=[16, 16])
    flow = ZukoMAF.from_config(config, n_inputs=4)
    assert flow.config.randperm is True
    samples = flow.sample(10)
    assert samples.shape == (10, 4)
