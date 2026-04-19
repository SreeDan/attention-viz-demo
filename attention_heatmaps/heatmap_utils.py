"""
Small-multiples heatmap visualization for all attention heads in a single layer.

The input format is the existing top-k text files produced by OpenFold's
save_attention_topk() in openfold/model/evoformer.py.  Each file covers one
layer and contains blocks like:

    Layer 47, Head 0
    18 45 0.952341
    ...
    Layer 47, Head 1
    ...

Because only top-k pairs are stored (not the full matrix), we reconstruct a
dense N×N matrix by placing each (res1, res2, weight) triple in its cell and
leaving everything else at 0.  For display purposes this is fine: the zeros
are visually distinct from the sparse non-zero entries, and users can toggle
normalization / thresholding via the web interface.
"""

import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe for servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_attention_matrix(file_path: str, seq_len: int | None = None, top_k: int | None = None):
    """
    Parse a top-k attention text file and return a dict mapping
      head_idx -> (N, N) float32 ndarray.

    Parameters
    ----------
    file_path : str
        Path to the layer attention file (e.g. msa_row_attn_layer47.txt).
    seq_len : int | None
        Expected sequence length.  If None it is inferred from the max
        residue index seen in the file (plus 1).
    top_k : int | None
        If given, keep only the top-k edges per head (by weight).

    Returns
    -------
    dict[int, np.ndarray]  head_idx -> (N, N) matrix
    int                    inferred or provided sequence length
    """
    raw: dict[int, list[tuple[int, int, float]]] = {}
    current_head = None

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("layer"):
                parts = line.replace(",", "").split()
                current_head = int(parts[-1])
                raw.setdefault(current_head, [])
            else:
                r1, r2, w = line.split()
                raw[current_head].append((int(r1), int(r2), float(w)))

    # Infer sequence length if not given
    if seq_len is None:
        max_idx = 0
        for triples in raw.values():
            for r1, r2, _ in triples:
                max_idx = max(max_idx, r1, r2)
        seq_len = max_idx + 1

    matrices: dict[int, np.ndarray] = {}
    for head_idx, triples in raw.items():
        triples.sort(key=lambda x: x[2], reverse=True)
        if top_k is not None:
            triples = triples[:top_k]
        mat = np.zeros((seq_len, seq_len), dtype=np.float32)
        for r1, r2, w in triples:
            if r1 < seq_len and r2 < seq_len:
                mat[r1, r2] = w
        matrices[head_idx] = mat

    return matrices, seq_len


# ---------------------------------------------------------------------------
# Normalization / thresholding helpers
# ---------------------------------------------------------------------------

def _normalize(mat: np.ndarray, mode: str) -> np.ndarray:
    """Apply normalization to a 2-D attention matrix."""
    if mode == "none":
        return mat
    if mode == "global_max":
        mx = mat.max()
        return mat / mx if mx > 0 else mat
    if mode == "row_softmax":
        # softmax row-wise over non-zero rows so sparse structure is kept
        out = np.zeros_like(mat)
        for i, row in enumerate(mat):
            if row.max() > 0:
                e = np.exp(row - row.max())
                out[i] = e / e.sum()
        return out
    raise ValueError(f"Unknown normalization mode: {mode!r}")


def _threshold(mat: np.ndarray, percentile: float) -> np.ndarray:
    """Zero out values below *percentile* of the non-zero entries."""
    if percentile <= 0:
        return mat
    nonzero = mat[mat > 0]
    if nonzero.size == 0:
        return mat
    cutoff = np.percentile(nonzero, percentile)
    out = mat.copy()
    out[out < cutoff] = 0.0
    return out


# ---------------------------------------------------------------------------
# Small-multiples heatmap grid
# ---------------------------------------------------------------------------

def plot_head_heatmaps(
    matrices: dict[int, np.ndarray],
    output_path: str,
    protein: str = "",
    layer_idx: int = 0,
    attention_type: str = "msa_row",
    normalization: str = "global_max",    # "none" | "global_max" | "row_softmax"
    threshold_pct: float = 0.0,           # 0–100; zero = no thresholding
    cmap: str = "viridis",
    cols: int = 4,
    cell_size: float = 3.0,
    dpi: int = 150,
    residue_sequence: str | None = None,
    highlight_residue: int | None = None,
):
    """
    Render all attention heads for one layer as a small-multiples grid of
    heatmaps and save to *output_path*.

    Parameters
    ----------
    matrices        : head_idx -> (N, N) array (from load_attention_matrix)
    output_path     : where to write the PNG
    cols            : number of columns in the grid
    cell_size       : inches per subplot cell (width = height)
    normalization   : per-head normalization scheme
    threshold_pct   : percentile threshold applied before plotting
    highlight_residue : draw a red cross-hair on this residue index
    """
    head_indices = sorted(matrices.keys())
    n_heads = len(head_indices)
    rows = math.ceil(n_heads / cols)

    fig_w = cols * cell_size
    fig_h = rows * cell_size + 1.0   # +1 for suptitle

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    # Build a shared color normalizer so all panels share the same scale
    all_vals = np.concatenate([m.ravel() for m in matrices.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for idx, head_idx in enumerate(head_indices):
        row_i, col_i = divmod(idx, cols)
        ax = axes[row_i][col_i]

        mat = _threshold(_normalize(matrices[head_idx], normalization), threshold_pct)

        im = ax.imshow(mat, cmap=cmap, norm=norm_obj, interpolation="nearest", aspect="auto")

        ax.set_title(f"Head {head_idx}", fontsize=8, pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if highlight_residue is not None and 0 <= highlight_residue < mat.shape[0]:
            ax.axhline(highlight_residue, color="red", linewidth=0.8, alpha=0.7)
            ax.axvline(highlight_residue, color="red", linewidth=0.8, alpha=0.7)

    # Hide unused axes
    for idx in range(n_heads, rows * cols):
        row_i, col_i = divmod(idx, cols)
        axes[row_i][col_i].set_visible(False)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.4, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_obj, cmap=cmap), cax=cbar_ax, label="Attention weight")

    title_parts = [f"{protein} · {attention_type.replace('_', ' ').title()} · Layer {layer_idx}"]
    if normalization != "none":
        title_parts.append(f"norm={normalization}")
    if threshold_pct > 0:
        title_parts.append(f"threshold={threshold_pct:.0f}th pct")
    fig.suptitle("  |  ".join(title_parts), fontsize=11, y=1.01)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Aggregated / summary heatmap
# ---------------------------------------------------------------------------

def plot_aggregated_heatmap(
    matrices: dict[int, np.ndarray],
    output_path: str,
    protein: str = "",
    layer_idx: int = 0,
    attention_type: str = "msa_row",
    agg: str = "mean",          # "mean" | "max" | "std"
    cmap: str = "magma",
    dpi: int = 150,
    highlight_residue: int | None = None,
):
    """
    Collapse all heads into a single matrix using *agg* and plot it.

    'std' is particularly useful: high-std positions differ a lot across
    heads, suggesting head specialisation; low-std positions are attended
    to uniformly (or ignored uniformly).
    """
    stack = np.stack(list(matrices.values()), axis=0)   # (n_heads, N, N)

    if agg == "mean":
        agg_mat = stack.mean(axis=0)
        label = "Mean attention across heads"
    elif agg == "max":
        agg_mat = stack.max(axis=0)
        label = "Max attention across heads"
    elif agg == "std":
        agg_mat = stack.std(axis=0)
        label = "Std-dev of attention across heads\n(high = head specialisation)"
    else:
        raise ValueError(f"Unknown agg: {agg!r}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(agg_mat, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.set_xlabel("Residue (key)")
    ax.set_ylabel("Residue (query)")
    ax.set_title(
        f"{protein} · {attention_type.replace('_', ' ').title()} · Layer {layer_idx}\n{label}",
        fontsize=10,
    )

    if highlight_residue is not None and 0 <= highlight_residue < agg_mat.shape[0]:
        ax.axhline(highlight_residue, color="red", linewidth=1.0, alpha=0.8, label=f"Residue {highlight_residue}")
        ax.axvline(highlight_residue, color="red", linewidth=1.0, alpha=0.8)
        ax.legend(fontsize=8, loc="upper right")

    fig.colorbar(im, ax=ax, label=label.split("\n")[0])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")
    return output_path
