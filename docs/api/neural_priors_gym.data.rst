``neural_priors_gym.data`` module
==================================

.. currentmodule:: neural_priors_gym.data

The data module handles the generation of training data by combining a mass
generator with an EOS-based tidal deformability interpolator.
:class:`~neural_priors_gym.data.generator.TrainingDataGenerator` is the main
orchestrator; it is typically constructed via
:meth:`~neural_priors_gym.data.generator.TrainingDataGenerator.from_config` from
a validated :class:`~neural_priors_gym.config.schema.TrainingConfig`.
For a description of all mass distribution models and their configuration
options, see the :doc:`/yaml_reference`.

Data generator
--------------

.. currentmodule:: neural_priors_gym.data.generator

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   TrainingDataGenerator

Mass generators
---------------

.. currentmodule:: neural_priors_gym.data.masses.base

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   MassGenerator

.. currentmodule:: neural_priors_gym.data.masses.uniform

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   UniformMassGenerator

.. currentmodule:: neural_priors_gym.data.masses.gaussian

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   GaussianMassGenerator

.. currentmodule:: neural_priors_gym.data.masses.double_gaussian

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   DoubleGaussianMassGenerator

.. currentmodule:: neural_priors_gym.data.masses.bilby_prior

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   BilbyPriorMassGenerator

.. currentmodule:: neural_priors_gym.data.masses.nsbh

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   NSBHMassGenerator

Lambda interpolator
-------------------

.. currentmodule:: neural_priors_gym.data.lambdas.interpolator

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   EOSLambdaInterpolator
