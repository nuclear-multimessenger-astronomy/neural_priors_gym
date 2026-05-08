"""Configuration schemas for mass generator types."""

from typing import Annotated, Literal, Union

from pydantic import Discriminator

from ._base import NeuralPriorBaseModel


class UniformMassConfig(NeuralPriorBaseModel):
    """Uniform mass distribution between m_min and MTOV (per EOS sample)."""

    type: Literal["uniform"] = "uniform"
    m_min: float


class GaussianMassConfig(NeuralPriorBaseModel):
    """Single Gaussian mass distribution truncated at m_min and MTOV."""

    type: Literal["gaussian"] = "gaussian"
    m_min: float
    mean: float
    std: float
    max_attempts: int = 1_000_000


class DoubleGaussianMassConfig(NeuralPriorBaseModel):
    """Mixture of two Gaussians for the mass distribution."""

    type: Literal["double_gaussian"] = "double_gaussian"
    m_min: float
    mean_1: float
    std_1: float
    mean_2: float
    std_2: float
    weight: float
    max_attempts: int = 1_000_000


class BilbyMassConfig(NeuralPriorBaseModel):
    """Mass distribution sampled from a bilby prior file.

    The prior file must define parameters from which bilby's
    generate_all_bns_parameters can derive mass_1_source and mass_2_source
    (e.g. chirp_mass, mass_ratio, luminosity_distance).
    """

    type: Literal["bilby"] = "bilby"
    prior_file: str


MassConfig = Annotated[
    Union[
        UniformMassConfig,
        GaussianMassConfig,
        DoubleGaussianMassConfig,
        BilbyMassConfig,
    ],
    Discriminator("type"),
]
