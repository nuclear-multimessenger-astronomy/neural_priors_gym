"""Top-level training configuration schema."""

from typing import Literal

from .schemas._base import NeuralPriorBaseModel
from .schemas.flow import FlowConfig, GlasflowConfig
from .schemas.lambdas import LambdaConfig
from .schemas.masses import (
    BilbyMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    MassConfig,
    UniformMassConfig,
)
from .schemas.training import TrainingHyperparamsConfig


class TrainingConfig(NeuralPriorBaseModel):
    """Top-level configuration for training a neural prior.

    Parameters
    ----------
    output_dir:
        Directory where all outputs (model, plots, training data) are saved.
    generate_only:
        If True, generate and save training data and its corner plot, then exit
        without training the flow. Useful for inspecting samples before training.
    source_type:
        "bns" for binary neutron star (both masses below MTOV, lambda_1 and
        lambda_2 interpolated) or "nsbh" for neutron star-black hole (m1
        sampled uniformly above MTOV, only lambda_2 interpolated for the NS).
    m_max_bh:
        Upper bound on the black-hole mass for NSBH sources (solar masses).
        Only used when source_type is "nsbh".
    masses:
        Configuration for the mass distribution generator.
    lambdas:
        Configuration for the tidal deformability generator.
    flow:
        Configuration for the normalizing flow backend and architecture.
    training:
        Hyperparameters for the training loop.
    """

    output_dir: str = "./outdir"
    source_type: Literal["bns", "nsbh"] = "bns"
    m_max_bh: float = 5.0
    generate_only: bool = False
    masses: MassConfig
    lambdas: LambdaConfig
    flow: FlowConfig
    training: TrainingHyperparamsConfig = TrainingHyperparamsConfig()


__all__ = [
    "TrainingConfig",
    "MassConfig",
    "UniformMassConfig",
    "GaussianMassConfig",
    "DoubleGaussianMassConfig",
    "BilbyMassConfig",
    "LambdaConfig",
    "FlowConfig",
    "GlasflowConfig",
    "TrainingHyperparamsConfig",
    "NeuralPriorBaseModel",
]
