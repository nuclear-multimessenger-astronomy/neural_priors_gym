"""
Convert a jester inference result HDF5 file to the npz format expected by neural_priors_gym.
This makes the jester result more lightweight for consumption by neural_priors_gym's EOSLambdaInterpolator, which only needs the masses_EOS and Lambdas_EOS arrays, not the full `jester` output.

Usage
-----
    python scripts/jester_hdf5_to_npz.py results.h5 eos_samples.npz

The output npz will contain two arrays:
  - ``masses_EOS``   : shape (n_eos, n_mass_points)
  - ``Lambdas_EOS``  : shape (n_eos, n_mass_points)

These are the keys expected by :class:`neural_priors_gym.data.lambdas.interpolator.EOSLambdaInterpolator`.
"""

import sys
from jesterTOV.inference.result import InferenceResult

import argparse
from pathlib import Path

import numpy as np


def convert(input_path: Path, output_path: Path) -> None:
    print(f"Loading jester result from {input_path} ...")
    result = InferenceResult.load(str(input_path))

    posterior = result.posterior
    masses_EOS: np.ndarray = posterior["masses_EOS"]
    Lambdas_EOS: np.ndarray = posterior["Lambdas_EOS"]

    print(f"Found {len(masses_EOS)} EOS samples.")
    np.savez(output_path, masses_EOS=masses_EOS, Lambdas_EOS=Lambdas_EOS)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a jester HDF5 inference result to an EOS npz file."
    )
    parser.add_argument("input", type=Path, help="Path to the jester results.h5 file.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("eos_samples.npz"),
        help="Output npz path (default: eos_samples.npz).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
