"""Double Gaussian mass generator."""

import numpy as np
from scipy.stats import truncnorm

from neural_priors_gym.config.schemas.masses import DoubleGaussianMassConfig

from .base import MassGenerator


class DoubleGaussianMassGenerator(MassGenerator):
    """Samples masses from a mixture of two Gaussians truncated to [m_min, MTOV].

    Default parameters are from the galactic NS population study in
    arXiv:2407.16669 (mu1=1.34, sigma1=0.07, mu2=1.80, sigma2=0.21, w=0.65).
    """

    def __init__(self, config: DoubleGaussianMassConfig) -> None:
        self.m_min = config.m_min
        self.mean_1 = config.mean_1
        self.std_1 = config.std_1
        self.mean_2 = config.mean_2
        self.std_2 = config.std_2
        self.weight = config.weight
        self.max_attempts = config.max_attempts

    def _sample_one(self, mtov: float) -> float:
        if np.random.rand() < self.weight:
            mean, std = self.mean_1, self.std_1
        else:
            mean, std = self.mean_2, self.std_2
        a = (self.m_min - mean) / std
        b = (mtov - mean) / std
        m = float(truncnorm.rvs(a, b, loc=mean, scale=std))
        if not (self.m_min <= m <= mtov):
            raise RuntimeError(
                f"truncnorm sample {m:.4f} out of [{self.m_min}, {mtov:.4f}] "
                f"after {self.max_attempts} max_attempts (numerical edge case)."
            )
        return m

    def generate(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]

        masses = np.array(
            [
                [
                    self._sample_one(mtov_per_sample[i]),
                    self._sample_one(mtov_per_sample[i]),
                ]
                for i in range(n_samples)
            ]
        )

        m1 = np.maximum(masses[:, 0], masses[:, 1])
        m2 = np.minimum(masses[:, 0], masses[:, 1])
        return np.column_stack([m1, m2])

    def generate_ns_mass(self, n_samples: int, mtov_array: np.ndarray) -> np.ndarray:
        eos_indices = np.random.randint(0, len(mtov_array), size=n_samples)
        mtov_per_sample = mtov_array[eos_indices]
        return np.array(
            [self._sample_one(mtov_per_sample[i]) for i in range(n_samples)]
        )
