"""
CLI script: generate small-multiples heatmaps and aggregated views for all
available layers and attention types in an attention directory.

Example
-------
    python attention_heatmaps/generate_heatmaps.py \
        --attn_dir /path/to/attn_maps \
        --output_dir outputs/heatmaps_6KWC \
        --protein 6KWC \
        --layer 47 \
        --normalization global_max \
        --threshold_pct 50 \
        --cols 4
"""

import argparse
import os
import glob as glob_mod

from heatmap_utils import load_attention_matrix, plot_head_heatmaps, plot_aggregated_heatmap
from head_metrics import compute_head_metrics


def main():
    parser = argparse.ArgumentParser(description="Batch heatmap generator for attention heads")
    parser.add_argument("--attn_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protein", default="protein")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer index. Default: all layers found.")
    parser.add_argument("--normalization", default="global_max",
                        choices=["none", "global_max", "row_softmax"])
    parser.add_argument("--threshold_pct", type=float, default=0.0,
                        help="Percentile threshold for non-zero attention values (0–95).")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--agg", default="mean", choices=["mean", "max", "std"])
    parser.add_argument("--highlight_residue", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- MSA Row ----
    msa_files = glob_mod.glob(os.path.join(args.attn_dir, "msa_row_attn_layer*.txt"))
    for fpath in sorted(msa_files):
        layer_idx = int(os.path.basename(fpath).split("layer")[1].split(".txt")[0])
        if args.layer is not None and layer_idx != args.layer:
            continue

        print(f"\n[MSA Row] Layer {layer_idx}")
        matrices, seq_len = load_attention_matrix(fpath)
        print(f"  Sequence length: {seq_len}, Heads: {list(matrices.keys())}")

        # Small-multiples
        grid_path = os.path.join(
            args.output_dir,
            f"msa_row_layer{layer_idx}_{args.protein}_grid.png"
        )
        plot_head_heatmaps(
            matrices,
            output_path=grid_path,
            protein=args.protein,
            layer_idx=layer_idx,
            attention_type="msa_row",
            normalization=args.normalization,
            threshold_pct=args.threshold_pct,
            cols=args.cols,
            cmap=args.cmap,
            dpi=args.dpi,
            highlight_residue=args.highlight_residue,
        )

        # Aggregated
        agg_path = os.path.join(
            args.output_dir,
            f"msa_row_layer{layer_idx}_{args.protein}_agg_{args.agg}.png"
        )
        plot_aggregated_heatmap(
            matrices,
            output_path=agg_path,
            protein=args.protein,
            layer_idx=layer_idx,
            attention_type="msa_row",
            agg=args.agg,
            cmap="magma",
            dpi=args.dpi,
            highlight_residue=args.highlight_residue,
        )

        # Print metrics
        metrics = compute_head_metrics(matrices)
        print(f"  {'Head':>4}  {'Entropy':>8}  {'Sparsity':>8}  {'DiagConc':>8}  {'MeanDist':>8}  {'MaxW':>6}")
        for head_idx in sorted(metrics):
            m = metrics[head_idx]
            print(f"  {head_idx:>4}  {m['entropy']:>8.4f}  {m['sparsity']:>8.4f}  "
                  f"{m['diagonal_concentration']:>8.4f}  {m['mean_attention_distance']:>8.2f}  "
                  f"{m['max_weight']:>6.4f}")

    # ---- Triangle Start ----
    tri_files = glob_mod.glob(os.path.join(args.attn_dir, "triangle_start_attn_layer*.txt"))
    for fpath in sorted(tri_files):
        basename = os.path.basename(fpath)
        layer_idx = int(basename.split("layer")[1].split("_residue")[0])
        residue_idx = int(basename.split("residue_idx_")[1].split(".txt")[0])

        if args.layer is not None and layer_idx != args.layer:
            continue

        print(f"\n[Triangle Start] Layer {layer_idx}, Residue {residue_idx}")
        matrices, seq_len = load_attention_matrix(fpath)
        print(f"  Sequence length: {seq_len}, Heads: {list(matrices.keys())}")

        grid_path = os.path.join(
            args.output_dir,
            f"tri_start_layer{layer_idx}_res{residue_idx}_{args.protein}_grid.png"
        )
        plot_head_heatmaps(
            matrices,
            output_path=grid_path,
            protein=args.protein,
            layer_idx=layer_idx,
            attention_type=f"triangle_start (res {residue_idx})",
            normalization=args.normalization,
            threshold_pct=args.threshold_pct,
            cols=args.cols,
            cmap=args.cmap,
            dpi=args.dpi,
            highlight_residue=residue_idx,
        )

        agg_path = os.path.join(
            args.output_dir,
            f"tri_start_layer{layer_idx}_res{residue_idx}_{args.protein}_agg_{args.agg}.png"
        )
        plot_aggregated_heatmap(
            matrices,
            output_path=agg_path,
            protein=args.protein,
            layer_idx=layer_idx,
            attention_type=f"triangle_start (res {residue_idx})",
            agg=args.agg,
            cmap="magma",
            dpi=args.dpi,
            highlight_residue=residue_idx,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
