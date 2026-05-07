[![CI](https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://nuclear-multimessenger-astronomy.github.io/neural_priors_gym/)

# neural_priors_gym

*A place to train your neural priors*

`neural_priors_gym` makes it easy to train a normalizing flow prior for multimessenger astronomy. It provides a unified interface that is compatible with various codebases and samplers, so you can focus on the science instead of the plumbing.

> [!TIP]
> **The documentation is the best place to get started.**
> It covers installation, examples, and the API reference.
>
> **[Read the full documentation →](https://nuclear-multimessenger-astronomy.github.io/neural_priors_gym/)**

## Installation

Install the latest version by cloning the repository:

```bash
git clone https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym
cd neural_priors_gym
uv sync
```

Extra dependencies can be installed as follows:
```bash
uv sync --extra dev    # For developers (run tests, build docs)
```

## For developers

All development guidelines — including how to run tests, contribute code, and write documentation — are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

If you use `neural_priors_gym` in your work, please cite the relevant paper(s).

See [`CITATION.cff`](CITATION.cff) or the [citing page in the documentation](https://nuclear-multimessenger-astronomy.github.io/neural_priors_gym/citing.html) for the full list of references.
