# Adding a new flow backend

This guide walks through the steps required to add a new normalizing flow backend
to ``neural_priors_gym``. The package is designed so that flow backends are
interchangeable: each backend is a thin wrapper around a specific library that
implements the common {class}`~neural_priors_gym.flows.base.FlowBase` interface.

## Overview

Adding a new backend requires changes in three places:

1. **A new config schema** in `config/schemas/flow.py` — defines the
   hyperparameters specific to the new backend.
2. **A new flow class** in `flows/<backend>.py` — implements the
   `FlowBase` abstract methods.
3. **A dispatch entry** in `training/trainer.py` — wires the new config
   type to the new flow class.

## Step 1 — Add a config schema

Open `src/neural_priors_gym/config/schemas/flow.py` and add a new Pydantic model
for your backend. The `backend` field must be a `Literal` type with a unique
string value that users will specify in their YAML file.

```python
class MyBackendConfig(NeuralPriorBaseModel):
    """Configuration for the my_backend flow."""

    backend: Literal["my_backend"] = "my_backend"
    n_transforms: int = 4
    # ... add hyperparameters specific to your backend
```

Then extend the `FlowConfig` discriminated union to include the new class:

```python
FlowConfig = Annotated[
    Union[GlasflowConfig, MyBackendConfig],
    Discriminator("backend"),
]
```

Finally, export the new class from `config/schema.py` so it is accessible from
the top-level module.

## Step 2 — Implement the flow class

Create a new file `src/neural_priors_gym/flows/my_backend.py`. Your class must
subclass {class}`~neural_priors_gym.flows.base.FlowBase` and implement all six
abstract methods:

```python
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from neural_priors_gym.flows.base import FlowBase

class MyBackendFlow(FlowBase):

    def train_epoch(self, data_loader: DataLoader) -> float:
        """One training epoch; return mean negative log-likelihood."""
        ...

    def eval_epoch(self, data_loader: DataLoader) -> float:
        """Evaluate on data_loader; return mean NLL."""
        ...

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples; return array of shape (n, n_inputs)."""
        ...

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Log probability for x; return array of shape (len(x),)."""
        ...

    def save(self, directory: Path) -> None:
        """Save weights and metadata to directory."""
        ...

    @classmethod
    def load(cls, directory: Path) -> "MyBackendFlow":
        """Load a saved flow from directory."""
        ...
```

Follow `flows/glasflow.py` as a reference for how to interact with the
underlying library and handle the optimizer and device placement.

## Step 3 — Add a dispatch entry in the trainer

Open `src/neural_priors_gym/training/trainer.py` and import your new config and
flow classes at the top of the file. Then add an `elif` branch in the
`FlowTrainer.build_flow` method:

```python
from neural_priors_gym.config.schemas.flow import GlasflowConfig, MyBackendConfig
from neural_priors_gym.flows.glasflow import GlasflowNSF
from neural_priors_gym.flows.my_backend import MyBackendFlow

class FlowTrainer:
    ...
    def build_flow(self, n_inputs: int) -> FlowBase:
        flow_config = self.config.flow
        if isinstance(flow_config, GlasflowConfig):
            return GlasflowNSF.from_config(flow_config, n_inputs)
        if isinstance(flow_config, MyBackendConfig):
            return MyBackendFlow.from_config(flow_config, n_inputs)
        raise ValueError(f"Unknown flow config type: {type(flow_config)}")
```

## Step 4 — Add tests

Add a test file `tests/test_flows/test_my_backend.py` that verifies at minimum:

- The flow can be constructed from a config.
- `train_epoch` and `eval_epoch` return finite scalars.
- `sample` returns an array of the correct shape.
- `log_prob` returns an array of the correct shape.
- The flow can be saved and loaded, and round-trips samples correctly.

Run the tests with:

```bash
uv run pytest -v tests/test_flows/test_my_backend.py
```

## Step 5 — Update the documentation

Add your new config class to the API reference in
`docs/api/neural_priors_gym.flows.rst` and document the new config options in
`docs/yaml_reference.md` following the existing dropdown pattern for the glasflow
backend. Rebuild the docs to verify no warnings are introduced:

```bash
uv run sphinx-build -W --keep-going docs docs/_build/html
```
