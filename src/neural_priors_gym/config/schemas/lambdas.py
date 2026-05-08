"""Configuration schema for the tidal deformability (Lambda) generator."""

from pydantic import model_validator

from ._base import NeuralPriorBaseModel


class LambdaConfig(NeuralPriorBaseModel):
    r"""Configuration for generating tidal deformabilities from EOS samples.

    Exactly one of ``eos_path`` or ``jester_path`` must be provided.

    - ``eos_path``: path to a NumPy ``.npz`` archive containing arrays
      ``masses_EOS`` and ``Lambdas_EOS``, each of shape
      ``(n_eos, n_mass_points)``.
    - ``jester_path``: path to a jester inference result ``.h5`` file.
      The posterior must contain ``masses_EOS`` and ``Lambdas_EOS`` arrays.

    Supported ``parameter_names`` (BNS): ``lambda_1``, ``lambda_2``,
    ``lambda_tilde``, ``delta_lambda_tilde`` (or any combination).
    For NSBH, only ``lambda_2`` is supported.

    If ``log_lambda`` is ``True``, a natural-log transform is applied to all
    :math:`\Lambda` columns before the flow sees the data. The saved
    ``training_data.npz`` always contains raw (untransformed) values.
    """

    parameter_names: list[str] = ["lambda_1", "lambda_2"]
    eos_path: str | None = None
    jester_path: str | None = None
    log_lambda: bool = False

    @model_validator(mode="after")
    def _check_exactly_one_path(self) -> "LambdaConfig":
        has_eos = self.eos_path is not None
        has_jester = self.jester_path is not None
        if has_eos == has_jester:
            raise ValueError(
                "Exactly one of 'eos_path' or 'jester_path' must be set in the "
                "lambdas config, but "
                + ("both were provided." if has_eos else "neither was provided.")
            )
        return self
