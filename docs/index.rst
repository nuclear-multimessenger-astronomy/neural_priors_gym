neural_priors_gym documentation
==========================

*A place to train your neural priors*

``neural_priors_gym`` makes it easy to train a normalizing flow prior for multimessenger astronomy. It provides a unified interface that is compatible with various codebases and samplers, so you can focus on the science instead of the plumbing.


Getting started
===============

* Will be added soon


Installation
=============

Install the latest version by cloning the repository::

    git clone https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym

We recommend using ``uv`` for managing the Python environment::

   uv venv --python=3.12
   source .venv/bin/activate

The package can then be installed directly::

    cd neural_priors_gym
    uv pip install -e .             # Basic install
    uv pip install -e ".[dev]"      # For developers (tests, docs)

Or using ``uv sync``::

    uv sync
    uv sync --extra dev      # For developers


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/neural_priors_gym

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   developer_guide/contributing

.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
