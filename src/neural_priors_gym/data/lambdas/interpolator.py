"""EOS-based tidal deformability interpolator."""

from pathlib import Path

import numpy as np

from neural_priors_gym.config.schemas.lambdas import LambdaConfig
from neural_priors_gym.logging_config import get_logger

logger = get_logger("neural_priors_gym.data.lambdas")

LAMBDA_CLIP_MIN = 1e-4


class EOSLambdaInterpolator:
    """Interpolates tidal deformabilities from a set of EOS samples.

    For each requested pair of masses, an EOS is drawn at random from the
    loaded set and the corresponding Lambda values are obtained by linear
    interpolation of that EOS's Lambda(M) curve.

    Parameters
    ----------
    config:
        Lambda configuration containing the path to the EOS npz file.
    """

    def __init__(self, config: LambdaConfig) -> None:
        self.parameter_names = config.parameter_names
        eos_path = Path(config.eos_path)
        if not eos_path.exists():
            raise FileNotFoundError(f"EOS file not found: {eos_path}")

        logger.info(f"Loading EOS samples from {eos_path}")
        data = np.load(eos_path)
        masses_raw: np.ndarray = data["masses_EOS"]
        lambdas_raw: np.ndarray = data["Lambdas_EOS"]

        n_total = len(masses_raw)
        keep = np.ones(n_total, dtype=bool)

        for i in range(n_total):
            if np.any(lambdas_raw[i] < 0.0):
                keep[i] = False

        n_removed = n_total - keep.sum()
        if n_removed > 0:
            logger.info(
                f"Removed {n_removed}/{n_total} EOS samples with negative lambdas."
            )

        self.masses_EOS: np.ndarray = masses_raw[keep]
        self.lambdas_EOS: np.ndarray = lambdas_raw[keep]

        # Warn about EOS samples with MTOV < 2.0 Msun
        n_low_mtov = sum(m.max() < 2.0 for m in self.masses_EOS)
        if n_low_mtov > 0:
            logger.warning(
                f"{n_low_mtov}/{len(self.masses_EOS)} EOS samples have MTOV < 2.0 Msun. "
                "These are kept but may produce unphysical mass samples."
            )

        logger.info(f"Loaded {len(self.masses_EOS)} valid EOS samples.")

    @property
    def mtov_array(self) -> np.ndarray:
        """Maximum TOV mass per EOS, shape (n_eos,)."""
        return np.array([m.max() for m in self.masses_EOS])

    def interpolate(self, masses: np.ndarray) -> np.ndarray:
        """Compute Lambda values for an array of masses.

        For each row in masses, a random EOS is selected and Lambda is
        obtained by linear interpolation. Values are clipped to a minimum
        of LAMBDA_CLIP_MIN for numerical stability.

        Parameters
        ----------
        masses:
            Array of shape (n_samples, k) where k is the number of neutron
            star masses to interpolate (k=2 for BNS, k=1 for NSBH where only
            the NS secondary is passed).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, k) containing tidal deformabilities.
        """
        n_samples, n_stars = masses.shape
        n_eos = len(self.masses_EOS)
        eos_indices = np.random.randint(0, n_eos, size=n_samples)

        lambdas = np.empty((n_samples, n_stars))
        for i in range(n_samples):
            idx = eos_indices[i]
            m_grid = self.masses_EOS[idx]
            l_grid = self.lambdas_EOS[idx]
            for j in range(n_stars):
                lambdas[i, j] = np.interp(masses[i, j], m_grid, l_grid)

        return np.clip(lambdas, a_min=LAMBDA_CLIP_MIN, a_max=None)
