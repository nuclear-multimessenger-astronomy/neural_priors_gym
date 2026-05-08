"""Abstract base class for mass generators."""

from abc import ABC, abstractmethod

import numpy as np


class MassGenerator(ABC):
    """Base class for all mass distribution generators.

    Subclasses implement generate() to produce pairs of neutron star masses
    consistent with the chosen population model and the available EOS samples.
    """

    parameter_names: list[str]

    @abstractmethod
    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        """Generate mass pairs.

        Parameters
        ----------
        n_samples:
            Number of (mass_1_source, mass_2_source) pairs to generate.
        mtov_array:
            Array of shape (n_eos,) with the maximum TOV mass for each EOS.
            A random EOS is drawn per sample and its MTOV used as the upper
            bound on the NS mass.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) where each row is
            (mass_1_source, mass_2_source) with mass_1_source >= mass_2_source.
        """
        ...

    @abstractmethod
    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        """Generate single NS masses for use as the secondary in NSBH systems.

        Parameters
        ----------
        n_samples:
            Number of NS masses to generate.
        mtov_array:
            Array of shape (n_eos,) with MTOV per EOS, used as the upper bound.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing NS masses below MTOV.
        """
        ...
