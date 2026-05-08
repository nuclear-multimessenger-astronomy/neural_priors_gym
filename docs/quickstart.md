(quickstart)=
# Quick start

**Train your first neural prior in a few minutes**

This guide explains how to train a normalizing flow prior with ``neural_priors_gym``
using a simple configuration file. As an example, we will train a prior over BNS
source-frame component masses and tidal deformabilities, using a uniform mass
distribution and a set of EOS samples to interpolate $\Lambda(M)$.


## What you need

To train a neural prior, two inputs are required:

1. **A ``config.yaml`` file** — specifies all settings for data generation, the flow architecture, and the training loop.
2. **An EOS sample file** (`.npz`) — a NumPy archive containing the arrays `masses_EOS` and `Lambdas_EOS`, each of shape `(n_eos, n_mass_points)`. These are used to interpolate the tidal deformability $\Lambda$ for each sampled mass.

The config file is all that is needed to run the command-line interface; the path to the EOS file is specified inside it.

## Running your first training

### Write a ``config.yaml`` file

Save the following to a file named ``config.yaml``:

```yaml
output_dir: ./outdir
source_type: bns

masses:
  type: uniform
  parameter_names:
    - mass_1_source
    - mass_2_source
  m_min: 1.0

lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  eos_path: /path/to/your/eos_samples.npz

flow:
  backend: glasflow
  n_transforms: 4
  n_neurons: 256
  n_blocks_per_transform: 3
  num_bins: 8
  tail_bound: 5.0
  activation: relu

training:
  n_samples: 20000
  num_epochs: 500
  learning_rate: 0.0001
  batch_size: 1024
  max_patience: 100
  validation_split: 0.2
  scale_input: true
```

```{note}
**Further reading:**

- Full explanation of every ``config.yaml`` field: {ref}`yaml-reference`
- Mass distribution options: see the {doc}`yaml_reference` (mass distribution section)
```

### Launch the training

From the directory containing ``config.yaml``, run:

```bash
neural_priors_gym_train config.yaml
```

```{note}
This command is only available after activating the environment where
``neural_priors_gym`` was installed. If the command is not found, try
``uv run neural_priors_gym_train config.yaml`` to use the package's virtual
environment directly.
```

``neural_priors_gym`` will then:

1. Load and validate the configuration.
2. Generate ``n_samples`` training samples by pairing masses drawn from the chosen distribution with EOS-interpolated tidal deformabilities.
3. Save the training data to ``outdir/training_data.npz`` and produce a corner plot of the raw samples (``outdir/training_data_corner.pdf``).
4. Split the data into training and validation sets, fit a ``MinMaxScaler`` if ``scale_input: true``, and train the normalizing flow with early stopping.
5. Save the trained flow to ``outdir/model/`` and the scaler to ``outdir/scaler.gz``.
6. Draw flow samples, compute the per-dimension Jensen-Shannon divergence (JSD) in millibits between the training data and flow samples, and save a comparison corner plot (``outdir/corner.pdf``) and loss curves (``outdir/losses.pdf``).

## Inspecting the training data first

Before committing to a full training run, it is often useful to inspect the
generated training data to verify the prior looks as expected. Setting
``generate_only: true`` in the config causes ``neural_priors_gym_train`` to
generate and save the training data and its corner plot, then exit without
training the flow:

```yaml
output_dir: ./outdir
generate_only: true
source_type: bns

masses:
  type: uniform
  parameter_names:
    - mass_1_source
    - mass_2_source
  m_min: 1.0

lambdas:
  parameter_names:
    - lambda_1
    - lambda_2
  eos_path: /path/to/your/eos_samples.npz

flow:
  backend: glasflow

training:
  n_samples: 20000
```

Open ``outdir/training_data_corner.pdf`` to inspect the prior. When you are
satisfied, set ``generate_only: false`` (or remove the field) and rerun to
start the training.

## Output files

After a successful run, the following files are saved to ``output_dir``:

| File | Description |
|------|-------------|
| ``training_data.npz`` | Raw training samples (masses and tidal deformabilities) |
| ``training_data_corner.pdf`` | Corner plot of the training data |
| ``model/`` | Directory containing the trained flow and scaler |
| ``scaler.gz`` | Fitted ``MinMaxScaler`` (copy also in ``model/``) |
| ``losses.pdf`` | Training and validation NLL per epoch |
| ``corner.pdf`` | Corner plot comparing training data to flow samples, annotated with JSD |

## Evaluating training quality

The main diagnostic is the Jensen-Shannon divergence (JSD) between the training
data and flow samples, reported in millibits. The value is printed to the log at
the end of training and shown in the corner plot title. A well-trained flow on a
smooth prior typically reaches a mean JSD below a few millibits per dimension.

## Next steps

1. **Try other mass distributions** — the {doc}`yaml_reference` documents all
   available mass models (uniform, Gaussian, double Gaussian, bilby prior).
2. **Train on NSBH systems** — set ``source_type: nsbh`` and adjust
   ``lambdas.parameter_names`` to ``[lambda_2]`` (only the NS contributes tidal
   deformability).
3. **Tune the flow architecture** — see the {ref}`yaml-reference-flow` section
   for all ``glasflow`` hyperparameters.
4. **Use the trained prior in bilby** — the ``model/`` directory is self-contained
   and can be loaded to sample from or evaluate the learned prior density.

---

**Quick Start Version**: 0.0.1
**Last Updated**: May 2026
