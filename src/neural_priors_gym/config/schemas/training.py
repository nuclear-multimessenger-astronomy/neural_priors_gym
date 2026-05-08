"""Configuration schema for training hyperparameters."""

from ._base import NeuralPriorBaseModel


class TrainingHyperparamsConfig(NeuralPriorBaseModel):
    """Hyperparameters for the normalizing flow training loop."""

    num_epochs: int = 2000
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_patience: int = 250
    validation_split: float = 0.2
    scale_input: bool = True
    n_samples: int = 20_000
    log_every_n_epochs: int = 100
