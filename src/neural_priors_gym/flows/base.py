"""Abstract base class for normalizing flow backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class FlowBase(ABC):
    """Common interface for all normalizing flow backends.

    Each backend (glasflow, zuko, etc.) provides a concrete subclass that
    wraps the backend-specific flow object and implements this API.
    """

    @abstractmethod
    def set_optimizer(self, learning_rate: float) -> None:
        """Initialise the optimizer (called by FlowTrainer before training)."""
        ...

    @abstractmethod
    def train_epoch(self, data_loader: "DataLoader") -> float:
        """Run one training epoch and return the mean negative log-likelihood."""
        ...

    @abstractmethod
    def eval_epoch(self, data_loader: "DataLoader") -> float:
        """Evaluate the flow on a data loader and return the mean NLL."""
        ...

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the flow. Returns array of shape (n, n_inputs)."""
        ...

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute log probabilities for x. Returns array of shape (len(x),)."""
        ...

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Save the flow weights and metadata to directory."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "FlowBase":
        """Load a saved flow from directory."""
        ...
