"""
Per-head diagnostic metrics.

These numbers let you *rank* or *filter* heads before visualising them —
solving the scale problem described in the GitHub issue.  Metrics are
intentionally simple so they can be computed from the sparse top-k data.
"""

import numpy as np


def _entropy(mat: np.ndarray) -> float:
    """Shannon entropy of the flattened (non-zero) distribution."""
    flat = mat.ravel()
    total = flat.sum()
    if total == 0:
        return 0.0
    p = flat[flat > 0] / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def _sparsity(mat: np.ndarray) -> float:
    """Fraction of matrix entries that are zero."""
    return float((mat == 0).sum() / mat.size)


def _diagonal_concentration(mat: np.ndarray, band: int = 5) -> float:
    """
    Fraction of total weight on the main diagonal ± *band* residues.
    High values → local / sequential attention; low values → long-range.
    """
    n = mat.shape[0]
    mask = np.zeros_like(mat, dtype=bool)
    for d in range(-band, band + 1):
        idx = np.arange(max(0, -d), min(n, n - d))
        mask[idx, idx + d] = True
    total = mat.sum()
    if total == 0:
        return 0.0
    return float(mat[mask].sum() / total)


def _max_weight(mat: np.ndarray) -> float:
    return float(mat.max())


def _mean_attention_distance(mat: np.ndarray) -> float:
    """
    Weighted mean |query - key| distance.  Long-range heads have high values.
    """
    total_w = mat.sum()
    if total_w == 0:
        return 0.0
    n = mat.shape[0]
    qs, ks = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dist = np.abs(qs - ks).astype(np.float32)
    return float((mat * dist).sum() / total_w)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_head_metrics(matrices: dict[int, np.ndarray]) -> dict[int, dict]:
    """
    Compute a set of diagnostic metrics for each head.

    Returns
    -------
    dict[head_idx -> dict]
        Keys: entropy, sparsity, diagonal_concentration, max_weight,
              mean_attention_distance
    """
    results = {}
    for head_idx, mat in matrices.items():
        results[head_idx] = {
            "entropy": round(_entropy(mat), 4),
            "sparsity": round(_sparsity(mat), 4),
            "diagonal_concentration": round(_diagonal_concentration(mat), 4),
            "max_weight": round(_max_weight(mat), 4),
            "mean_attention_distance": round(_mean_attention_distance(mat), 4),
        }
    return results
