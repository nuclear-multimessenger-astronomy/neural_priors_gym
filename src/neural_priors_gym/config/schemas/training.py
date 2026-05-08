"""Configuration schema for training hyperparameters."""

from ._base import NeuralPriorBaseModel


class TrainingHyperparamsConfig(NeuralPriorBaseModel):
    """Hyperparameters for the normalizing flow training loop.

    Parameters
    ----------
    parameter_names:
        Ordered list of parameter names to train the normalizing flow on.
        These are the columns extracted from the training data (whether
        generated or loaded from ``data_path``). All names must exist as
        keys in the training data.
    data_path:
        Path to a pre-generated ``.npz`` file to use as training data.
        When set, the ``masses`` and ``lambdas`` generation steps are
        skipped entirely. The file must contain all keys listed in
        ``parameter_names``.
    log_lambda:
        If ``True``, apply a natural-log transform to all parameters whose
        names appear in the known lambda parameter set (``lambda_1``,
        ``lambda_2``, ``lambda_tilde``, ``delta_lambda_tilde``) before
        the flow sees the data. The saved ``training_data.npz`` always
        retains raw values.
    """

    parameter_names: list[str]
    data_path: str | None = None
    log_lambda: bool = False
    num_epochs: int = 2000
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_patience: int = 250
    validation_split: float = 0.2
    scale_input: bool = True
    n_samples: int = 20_000
    log_every_n_epochs: int = 100
