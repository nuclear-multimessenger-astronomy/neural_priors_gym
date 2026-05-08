neural_priors_gym documentation
================================

*Train normalizing flow priors for neutron star gravitational-wave inference*

``neural_priors_gym`` is a package for training normalizing flows as neural prior
distributions for Bayesian gravitational-wave parameter estimation.
Given a set of equation-of-state (EOS) samples and a chosen compact-object mass
distribution, it generates training data, fits a normalizing flow, and saves the
trained model so that it can be plugged directly into Bayesian inference pipelines
such as ``bilby``.

What's in ``neural_priors_gym``?
==================================

``neural_priors_gym`` combines flexible mass-distribution models with a
configurable normalizing-flow backend to produce EOS-informed priors for BNS
and NSBH gravitational-wave analyses.

.. grid:: 2
    :class-container: component-grid

    .. grid-item:: ⚖️ Mass distributions

       Population-level models for compact-object masses

       - Uniform
       - Gaussian
       - Double Gaussian
       - Bilby prior

    .. grid-item:: 🌊 Flow backends

       Train a normalizing flow on the generated prior samples

       - ``glasflow`` CouplingNSF (Neural Spline Flow)

    .. grid-item:: 🌟 Source types

       Supported compact binary configurations

       - Binary neutron star (BNS)
       - Neutron star – black hole (NSBH)

    .. grid-item:: 📊 Evaluation and outputs

       Inspect and verify the trained prior

       - Jensen-Shannon divergence (millibits)
       - Corner plots: training data vs flow samples
       - Loss curves across training epochs


Getting started
===============

* New to ``neural_priors_gym``? Start with the :doc:`quickstart` guide to train your first neural prior in a few minutes.
* For a complete reference of every configuration option, see the :doc:`yaml_reference`.
* Dive into the code in the :doc:`API reference <api/neural_priors_gym>`.


Installation
=============

Install the latest version by cloning the repository::

    git clone https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym

We recommend using ``uv`` for managing the Python environment. Once ``uv`` is
installed, create a virtual environment and activate it::

    uv venv --python=3.12
    source .venv/bin/activate

Install the package in editable mode::

    cd neural_priors_gym
    uv pip install -e .            # Basic install
    uv pip install -e ".[dev]"     # For developers (tests, docs)

Or using ``uv sync``::

    uv sync
    uv sync --extra dev            # For developers


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   yaml_reference

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/neural_priors_gym

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   developer_guide/contributing
   developer_guide/adding_new_flow
   developer_guide/adding_new_mass_model

.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
