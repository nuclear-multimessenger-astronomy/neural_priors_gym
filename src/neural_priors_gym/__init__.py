"""neural_priors_gym - Train normalizing flow priors for neutron star inference."""

__version__ = "0.0.1"
__author__ = "ThibeauWouters"

from neural_priors_gym.config import load_config
from neural_priors_gym.config.schema import TrainingConfig

__all__ = ["load_config", "TrainingConfig"]
