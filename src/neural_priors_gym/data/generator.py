"""Orchestrates mass and Lambda generation to produce training data."""

from pathlib import Path
from typing import Optional

import numpy as np
from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    component_masses_to_mass_ratio,
    lambda_1_lambda_2_to_delta_lambda_tilde,
    lambda_1_lambda_2_to_lambda_tilde,
)

from neural_priors_gym.config.schema import TrainingConfig
from neural_priors_gym.config.schemas.masses import (
    BilbyMassConfig,
    DoubleGaussianMassConfig,
    GaussianMassConfig,
    UniformMassConfig,
)
from neural_priors_gym.data.lambdas.interpolator import EOSLambdaInterpolator
from neural_priors_gym.data.masses.base import MassGenerator
from neural_priors_gym.data.masses.bilby_prior import BilbyPriorMassGenerator
from neural_priors_gym.data.masses.double_gaussian import DoubleGaussianMassGenerator
from neural_priors_gym.data.masses.gaussian import GaussianMassGenerator
from neural_priors_gym.data.masses.nsbh import NSBHMassGenerator
from neural_priors_gym.data.masses.uniform import UniformMassGenerator
from neural_priors_gym.logging_config import get_logger

logger = get_logger("neural_priors_gym.data")


def _build_mass_quantities(
    mass_1_source: np.ndarray, mass_2_source: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute all supported mass quantities from component masses."""
    return {
        "mass_1_source": mass_1_source,
        "mass_2_source": mass_2_source,
        "chirp_mass_source": component_masses_to_chirp_mass(
            mass_1_source, mass_2_source
        ),
        "mass_ratio": component_masses_to_mass_ratio(mass_1_source, mass_2_source),
    }


def _build_lambda_quantities(
    mass_1_source: np.ndarray,
    mass_2_source: np.ndarray,
    lambda_1: np.ndarray,
    lambda_2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all supported lambda quantities for BNS systems."""
    return {
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "lambda_tilde": lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1_source, mass_2_source
        ),
        "delta_lambda_tilde": lambda_1_lambda_2_to_delta_lambda_tilde(
            lambda_1, lambda_2, mass_1_source, mass_2_source
        ),
    }


def build_mass_generator(config: TrainingConfig) -> MassGenerator:
    """Construct the appropriate MassGenerator from the training config.

    For NSBH sources, the base NS mass generator is wrapped in
    NSBHMassGenerator so that m1 (BH) is sampled above MTOV and m2 (NS)
    is sampled from the chosen NS population model.
    """
    mass_config = config.masses
    if isinstance(mass_config, UniformMassConfig):
        ns_gen: MassGenerator = UniformMassGenerator(mass_config)
    elif isinstance(mass_config, GaussianMassConfig):
        ns_gen = GaussianMassGenerator(mass_config)
    elif isinstance(mass_config, DoubleGaussianMassConfig):
        ns_gen = DoubleGaussianMassGenerator(mass_config)
    elif isinstance(mass_config, BilbyMassConfig):
        ns_gen = BilbyPriorMassGenerator(mass_config)
    else:
        raise ValueError(f"Unknown mass config type: {type(mass_config)}")

    if config.source_type == "nsbh":
        return NSBHMassGenerator(ns_gen, config.m_max_bh)

    return ns_gen


class TrainingDataGenerator:
    """Generates training data by pairing a mass generator with an EOS interpolator.

    Parameters
    ----------
    mass_generator:
        Produces (m1, m2) pairs according to the chosen population model.
    lambda_interpolator:
        Produces Lambda values from EOS samples.
    n_samples:
        Number of training samples to generate.
    mass_parameter_names:
        Names for the mass columns in the output. Supported names are
        "mass_1_source" / "m_1", "mass_2_source" / "m_2",
        "chirp_mass_source", and "mass_ratio".
    lambda_parameter_names:
        Names for the lambda columns (e.g. ["lambda_1", "lambda_2"] for BNS
        or ["lambda_2"] for NSBH).
    source_type:
        "bns" or "nsbh". For NSBH, only m2 (the NS) is passed to the lambda
        interpolator; m1 (the BH) has no tidal deformability.
    """

    def __init__(
        self,
        mass_generator: MassGenerator,
        lambda_interpolator: EOSLambdaInterpolator,
        n_samples: int,
        mass_parameter_names: Optional[list[str]] = None,
        lambda_parameter_names: Optional[list[str]] = None,
        source_type: str = "bns",
    ) -> None:
        self.mass_generator = mass_generator
        self.lambda_interpolator = lambda_interpolator
        self.n_samples = n_samples
        self.mass_parameter_names = (
            mass_parameter_names or mass_generator.parameter_names
        )
        self.lambda_parameter_names = (
            lambda_parameter_names or lambda_interpolator.parameter_names
        )
        self.source_type = source_type

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "TrainingDataGenerator":
        """Build a TrainingDataGenerator from a TrainingConfig."""
        mass_gen = build_mass_generator(config)
        lambda_interp = EOSLambdaInterpolator(config.lambdas)
        return cls(
            mass_generator=mass_gen,
            lambda_interpolator=lambda_interp,
            n_samples=config.training.n_samples,
            mass_parameter_names=config.masses.parameter_names,
            lambda_parameter_names=config.lambdas.parameter_names,
            source_type=config.source_type,
        )

    def generate(self) -> dict[str, np.ndarray]:
        """Generate the full training dataset.

        Component masses (mass_1_source, mass_2_source) are always generated
        internally. Output names in mass_parameter_names are then looked up
        from a computed dict that includes derived quantities such as
        chirp_mass_source and mass_ratio.

        For NSBH sources, only mass_2_source (the NS) is passed to the lambda
        interpolator. mass_1_source (BH) gets lambda_1=0, so lambda_tilde and
        delta_lambda_tilde are still well-defined and available as outputs.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to 1-D arrays of length
            n_samples.
        """
        logger.info(f"Generating {self.n_samples} training samples ...")

        masses = self.mass_generator.generate(
            self.n_samples, self.lambda_interpolator.mtov_array
        )
        mass_1_source, mass_2_source = masses[:, 0], masses[:, 1]

        # Compute all supported mass quantities so any name can be looked up.
        mass_quantities = _build_mass_quantities(mass_1_source, mass_2_source)

        # For NSBH, only pass mass_2_source (NS) to the interpolator.
        if self.source_type == "nsbh":
            lambda_masses = mass_2_source[:, np.newaxis]
        else:
            lambda_masses = masses

        lambdas = self.lambda_interpolator.interpolate(lambda_masses)

        # Build a registry of all computable lambda quantities.
        if self.source_type == "bns":
            lambda_quantities: dict[str, np.ndarray] = _build_lambda_quantities(
                mass_1_source, mass_2_source, lambdas[:, 0], lambdas[:, 1]
            )
        else:
            # NSBH: BH has lambda_1=0; NS lambda_2 drives all tidal combinations.
            lambda_1_nsbh = np.zeros_like(lambdas[:, 0])
            lambda_2_nsbh = lambdas[:, 0]
            lambda_quantities = _build_lambda_quantities(
                mass_1_source, mass_2_source, lambda_1_nsbh, lambda_2_nsbh
            )

        data: dict[str, np.ndarray] = {}

        for name in self.mass_parameter_names:
            if name not in mass_quantities:
                raise ValueError(
                    f"Unknown mass parameter name '{name}'. Supported names: "
                    f"{sorted(mass_quantities.keys())}"
                )
            data[name] = mass_quantities[name]

        for name in self.lambda_parameter_names:
            if name not in lambda_quantities:
                raise ValueError(
                    f"Lambda parameter '{name}' is not available for "
                    f"source_type='{self.source_type}'. "
                    f"Available: {sorted(lambda_quantities.keys())}"
                )
            data[name] = lambda_quantities[name]

        logger.info("Training data generation complete.")
        return data

    def save(self, path: Path) -> None:
        """Generate and save training data to an npz file."""
        data = self.generate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, allow_pickle=True, **data)
        logger.info(f"Training data saved to {path}")
