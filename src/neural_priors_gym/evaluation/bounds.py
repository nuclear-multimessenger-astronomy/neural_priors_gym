"""Out-of-bounds validation for flow samples."""

from __future__ import annotations

import numpy as np

# Central registry of known physical parameter bounds.
# Each entry maps a parameter name to (lower_bound, upper_bound).
# Use None for an unconstrained side.
# Add new parameters here to extend validation without touching other code.
PARAMETER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "mass_ratio": (None, 1.0),  # q = m2/m1 <= 1 by convention
    "lambda_1": (0.0, None),
    "lambda_2": (0.0, None),
}


def check_out_of_bounds(
    samples: np.ndarray,
    parameter_names: list[str],
) -> dict[str, dict[str, float]]:
    """Check flow samples against known physical bounds.

    Only parameters that appear in both *parameter_names* and
    :data:`PARAMETER_BOUNDS` are checked. Parameters with no registered
    bounds are silently skipped.

    Parameters
    ----------
    samples:
        Array of shape ``(n_samples, n_params)``.
    parameter_names:
        Ordered list of parameter names corresponding to columns of *samples*.

    Returns
    -------
    dict
        Keyed by parameter name (only those with registered bounds that are
        present in *parameter_names*). Each value is a dict with:

        - ``pct_oob``: percentage of samples that violate any bound (0–100).
        - ``pct_below``: percentage below the lower bound (0 if no lower bound).
        - ``pct_above``: percentage above the upper bound (0 if no upper bound).
        - ``n_oob``: absolute count of out-of-bounds samples.
        - ``n_samples``: total number of samples checked.
    """
    n_samples = len(samples)
    results: dict[str, dict[str, float]] = {}

    for col_idx, name in enumerate(parameter_names):
        if name not in PARAMETER_BOUNDS:
            continue

        lo, hi = PARAMETER_BOUNDS[name]
        col = samples[:, col_idx]

        n_below = int(np.sum(col < lo)) if lo is not None else 0
        n_above = int(np.sum(col > hi)) if hi is not None else 0
        n_oob = n_below + n_above

        results[name] = {
            "pct_oob": 100.0 * n_oob / n_samples,
            "pct_below": 100.0 * n_below / n_samples,
            "pct_above": 100.0 * n_above / n_samples,
            "n_oob": float(n_oob),
            "n_samples": float(n_samples),
        }

    return results
