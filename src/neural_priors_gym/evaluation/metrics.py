"""Quality metrics for comparing training data and flow samples."""

import numpy as np
from scipy.special import kl_div


def compute_jsd_1d(
    p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = 50
) -> float:
    """Compute Jensen-Shannon divergence between two 1-D sample sets.

    Uses histogram density estimation with a common bin range. Returns the
    JSD in bits.

    Parameters
    ----------
    p_samples:
        Samples from distribution P.
    q_samples:
        Samples from distribution Q.
    n_bins:
        Number of histogram bins.

    Returns
    -------
    float
        JSD in bits.
    """
    min_val = min(p_samples.min(), q_samples.min())
    max_val = max(p_samples.max(), q_samples.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    p_prob = p_hist / (p_hist.sum() + 1e-300)
    q_prob = q_hist / (q_hist.sum() + 1e-300)

    eps = 1e-10
    p_prob = np.maximum(p_prob, eps)
    q_prob = np.maximum(q_prob, eps)

    m_prob = 0.5 * (p_prob + q_prob)

    # kl_div from scipy computes x*log(x/y) element-wise; sum gives KL(P||M)
    kl_pm = np.sum(kl_div(p_prob, m_prob))
    kl_qm = np.sum(kl_div(q_prob, m_prob))
    jsd_nats = 0.5 * kl_pm + 0.5 * kl_qm

    return float(jsd_nats / np.log(2))


def compute_jsd(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    n_bins: int = 50,
) -> dict[str, float]:
    """Compute per-dimension JSD and a mean JSD in millibits.

    Parameters
    ----------
    p_samples:
        Array of shape (n, d) from distribution P (e.g. training data).
    q_samples:
        Array of shape (m, d) from distribution Q (e.g. flow samples).
    n_bins:
        Number of histogram bins per dimension.

    Returns
    -------
    dict
        Dictionary with keys 'dim_{i}_bits', 'dim_{i}_millibits' for each
        dimension i, plus 'mean_millibits'.
    """
    n_dims = p_samples.shape[1]
    results: dict[str, float] = {}
    millibits = []

    for i in range(n_dims):
        jsd_bits = compute_jsd_1d(p_samples[:, i], q_samples[:, i], n_bins=n_bins)
        jsd_mb = jsd_bits * 1000.0
        results[f"dim_{i}_bits"] = jsd_bits
        results[f"dim_{i}_millibits"] = jsd_mb
        millibits.append(jsd_mb)

    results["mean_millibits"] = float(np.mean(millibits))
    return results
