"""Configuration schemas for normalizing flow backends."""

from typing import Annotated, Literal, Union

from pydantic import Discriminator

from ._base import NeuralPriorBaseModel


class GlasflowConfig(NeuralPriorBaseModel):
    """Configuration for the glasflow CouplingNSF backend."""

    backend: Literal["glasflow"] = "glasflow"
    flow_type: Literal["CouplingNSF"] = (
        "CouplingNSF"  # TODO: extend with more flow types
    )
    n_transforms: int = 4
    n_neurons: int = 256
    n_blocks_per_transform: int = 3
    num_bins: int = 8
    tail_bound: float = 5.0
    activation: Literal["relu", "gelu", "silu", "tanh"] = "relu"
    linear_transform: Literal["permutation", "lu", "svd", "none"] = "none"
    dropout_probability: float = 0.0
    batch_norm_within_blocks: bool = False
    batch_norm_between_transforms: bool = False


FlowConfig = Annotated[
    Union[GlasflowConfig,],
    Discriminator("backend"),
]
