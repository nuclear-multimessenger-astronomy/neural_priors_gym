``neural_priors_gym.training`` module
======================================

.. currentmodule:: neural_priors_gym.training

The training module provides the :class:`~neural_priors_gym.training.trainer.FlowTrainer`
class, which handles the full training loop: data splitting, input scaling,
mini-batch optimization, early stopping, and checkpoint saving. The trainer is
driven entirely by the :class:`~neural_priors_gym.config.schema.TrainingConfig`
so no manual setup is required when using the command-line interface.

Trainer
-------

.. currentmodule:: neural_priors_gym.training.trainer

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   FlowTrainer
