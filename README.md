[![CI](https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/neural_priors_gym/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://nuclear-multimessenger-astronomy.github.io/neural_priors_gym/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.22987-b31b1b.svg)](https://arxiv.org/abs/2511.22987)

# neural_priors_gym

*A place to train your neural priors*

`neural_priors_gym` makes it easy to train a normalizing flow prior for multimessenger astronomy. It provides a unified interface that is compatible with various codebases and samplers, so you can focus on the science instead of the plumbing.

> [!TIP]
> **The documentation is the best place to get started.**
> It covers installation, examples, and the API reference.
>
> **[Read the full documentation →](https://nuclear-multimessenger-astronomy.github.io/neural_priors_gym/)**

## Minimal example

Write a `config.yaml`:

```yaml
output_dir: ./outdir
source_type: bns

masses:
  type: uniform
  m_min: 1.0

lambdas:
  eos_path: /path/to/eos_samples.npz  # must contain `masses_EOS` and `Lambdas_EOS`

flow:
  backend: zuko
  flow_type: maf
  transforms: 4
  hidden_features: [64, 64]

training:
  parameter_names: [mass_1_source, mass_2_source, lambda_1, lambda_2]
  n_samples: 20000
  num_epochs: 500
  learning_rate: 0.0001
  batch_size: 1024
  max_patience: 100
  validation_split: 0.2
  scale_input: true
```

Then train:

```bash
train_neural_prior config.yaml
```

The trained flow is saved to `outdir/model/` and ready to use as a prior in bilby or any other sampler.
More config settings and examples are detailed in the documentation.

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
