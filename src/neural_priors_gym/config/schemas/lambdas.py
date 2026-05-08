"""Configuration schema for the tidal deformability (Lambda) generator."""

from ._base import NeuralPriorBaseModel


class LambdaConfig(NeuralPriorBaseModel):
    """Configuration for generating tidal deformabilities from EOS samples.

    The npz file at eos_path must contain arrays named 'masses_EOS' and
    'Lambdas_EOS', each of shape (n_eos, n_mass_points).

    Supported parameter_names (BNS): lambda_1, lambda_2, lambda_tilde,
    delta_lambda_tilde (or any combination).
    For NSBH, only lambda_2 is supported.

    If log_lambda is True, a log transform is applied to all lambda
    parameters before the flow is trained. The saved training_data.npz
    always contains raw (untransformed) values.
    """

    parameter_names: list[str] = ["lambda_1", "lambda_2"]
    eos_path: str
    log_lambda: bool = False
