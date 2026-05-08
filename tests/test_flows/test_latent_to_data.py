"""Tests for FlowBase.latent_to_data — the generative direction (latent → data).

Both GlasflowNSF and ZukoMAF must implement this method so that nested
samplers (dynesty) can use _rescale without backend-specific hacks.

Round-trip correctness test
----------------------------
For an invertible normalising flow f:
    x = f^{-1}(z)  <=>  z = f(x)
We verify this by:
1. Sampling x from the flow.
2. Encoding x → z via the backend's own forward transform.
3. Decoding z → x' via ``latent_to_data``.
4. Asserting x' ≈ x to within floating-point tolerance.
"""

import numpy as np
import pytest
import torch

from neural_priors_gym.config.schemas.flow import GlasflowConfig, ZukoMAFConfig
from neural_priors_gym.flows.glasflow import GlasflowNSF
from neural_priors_gym.flows.zuko_maf import ZukoMAF

N_INPUTS = 4
N_SAMPLES = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def glasflow() -> GlasflowNSF:
    config = GlasflowConfig(
        n_transforms=2,
        n_neurons=16,
        n_blocks_per_transform=1,
        num_bins=4,
    )
    return GlasflowNSF.from_config(config, n_inputs=N_INPUTS)


@pytest.fixture
def zuko_maf() -> ZukoMAF:
    config = ZukoMAFConfig(transforms=2, hidden_features=[16, 16])
    return ZukoMAF.from_config(config, n_inputs=N_INPUTS)


# ---------------------------------------------------------------------------
# GlasflowNSF tests
# ---------------------------------------------------------------------------


def test_glasflow_latent_to_data_shape(glasflow: GlasflowNSF) -> None:
    z = np.random.randn(N_SAMPLES, N_INPUTS).astype(np.float32)
    x = glasflow.latent_to_data(z)
    assert x.shape == (N_SAMPLES, N_INPUTS)


def test_glasflow_latent_to_data_returns_numpy(glasflow: GlasflowNSF) -> None:
    z = np.random.randn(10, N_INPUTS).astype(np.float32)
    x = glasflow.latent_to_data(z)
    assert isinstance(x, np.ndarray)


def test_glasflow_latent_to_data_finite(glasflow: GlasflowNSF) -> None:
    z = np.random.randn(N_SAMPLES, N_INPUTS).astype(np.float32)
    x = glasflow.latent_to_data(z)
    assert np.all(np.isfinite(x))


def test_glasflow_latent_to_data_roundtrip(glasflow: GlasflowNSF) -> None:
    """Encoding x with the forward transform then decoding should recover x."""
    glasflow._flow.eval()
    x_np = glasflow.sample(N_SAMPLES).astype(np.float32)
    x_t = torch.tensor(x_np)

    # Forward: x → z  (glasflow convention: forward returns (z, log_jacobian))
    with torch.inference_mode():
        z_t, _ = glasflow._flow.forward(x_t)

    x_recovered = glasflow.latent_to_data(z_t.numpy())
    np.testing.assert_allclose(x_recovered, x_np, atol=1e-5)


# ---------------------------------------------------------------------------
# ZukoMAF tests
# ---------------------------------------------------------------------------


def test_zuko_latent_to_data_shape(zuko_maf: ZukoMAF) -> None:
    z = np.random.randn(N_SAMPLES, N_INPUTS).astype(np.float32)
    x = zuko_maf.latent_to_data(z)
    assert x.shape == (N_SAMPLES, N_INPUTS)


def test_zuko_latent_to_data_returns_numpy(zuko_maf: ZukoMAF) -> None:
    z = np.random.randn(10, N_INPUTS).astype(np.float32)
    x = zuko_maf.latent_to_data(z)
    assert isinstance(x, np.ndarray)


def test_zuko_latent_to_data_finite(zuko_maf: ZukoMAF) -> None:
    z = np.random.randn(N_SAMPLES, N_INPUTS).astype(np.float32)
    x = zuko_maf.latent_to_data(z)
    assert np.all(np.isfinite(x))


def test_zuko_latent_to_data_roundtrip(zuko_maf: ZukoMAF) -> None:
    """Encoding x with the forward transform then decoding should recover x."""
    zuko_maf._flow.eval()
    x_np = zuko_maf.sample(N_SAMPLES).astype(np.float32)
    x_t = torch.tensor(x_np)

    # Forward: x → z  (zuko convention: dist.transform maps x to latent z)
    with torch.inference_mode():
        dist = zuko_maf._flow()
        z_t = dist.transform(x_t)

    x_recovered = zuko_maf.latent_to_data(z_t.numpy())
    np.testing.assert_allclose(x_recovered, x_np, atol=1e-5)
