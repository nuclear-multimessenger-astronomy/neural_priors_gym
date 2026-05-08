``neural_priors_gym.evaluation`` module
========================================

.. currentmodule:: neural_priors_gym.evaluation

The evaluation module provides metrics for assessing training quality. The primary
diagnostic is the Jensen-Shannon divergence (JSD), which is computed between the
generated training samples and samples drawn from the trained flow. JSD values
close to zero indicate that the flow has accurately learned the target distribution.
Results are reported in millibits (thousandths of a bit) for easy interpretation:
a well-trained flow on a smooth prior typically achieves a mean JSD of a few
millibits per dimension.

Metrics
-------

.. currentmodule:: neural_priors_gym.evaluation.metrics

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   compute_jsd
   compute_jsd_1d
