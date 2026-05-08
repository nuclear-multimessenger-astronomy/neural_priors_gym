"""Gaussian mass generator."""

import numpy as np
from scipy.stats import truncnorm

from neural_priors_gym.config.schemas.masses import GaussianMassConfig

from .base import MassGenerator


class GaussianMassGenerator(MassGenerator):
    """Samples masses from a Gaussian distribution truncated to [m_min, MTOV].

    Default parameters (mean=1.33, std=0.09) are taken from the galactic
    NS population study in arXiv:2407.16669.
    """

    def __init__(self, config: GaussianMassConfig) -> None:
        self.parameter_names = config.parameter_names
        self.m_min = config.m_min
        self.mean = config.mean
        self.std = config.std
        self.max_attempts = config.max_attempts

    def _sample_truncated(self, mtov: float) -> float:
        a = (self.m_min - self.mean) / self.std
        b = (mtov - self.mean) / self.std
        m = float(truncnorm.rvs(a, b, loc=self.mean, scale=self.std))
        if not (self.m_min <= m <= mtov):
            raise RuntimeError(
                f"truncnorm sample {m:.4f} out of [{self.m_min}, {mtov:.4f}] "
                f"after {self.max_attempts} max_attempts (numerical edge case)."
            )
        return m

    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]

        masses = np.empty((n_samples, 2))
        for idx in range(n_samples):
            mtov = mtov_per_sample[idx]
            masses[idx, 0] = self._sample_truncated(mtov)
            masses[idx, 1] = self._sample_truncated(mtov)

        m1 = np.maximum(masses[:, 0], masses[:, 1])
        m2 = np.minimum(masses[:, 0], masses[:, 1])
        return np.column_stack([m1, m2])

    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]
        return np.array(
            [self._sample_truncated(mtov_per_sample[i]) for i in range(n_samples)]
        )
