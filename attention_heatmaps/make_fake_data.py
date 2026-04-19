"""
Generate synthetic attention text files for local testing.
No GPU, no OpenFold, no model weights needed.

Usage:
    python attention_heatmaps/make_fake_data.py --out_dir /tmp/fake_attn
"""

import argparse
import os
import random

import numpy as np


def write_layer_file(path: str, n_heads: int, seq_len: int, top_k: int, layer_idx: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for head in range(n_heads):
            f.write(f"Layer {layer_idx}, Head {head}\n")
            # Give each head a slightly different character
            if head % 3 == 0:
                # local / diagonal head
                pairs = [(i, i + rng.integers(1, 6), float(rng.random())) for i in range(seq_len - 6)]
            elif head % 3 == 1:
                # long-range head
                pairs = [
                    (rng.integers(0, seq_len // 2), rng.integers(seq_len // 2, seq_len), float(rng.random()))
                    for _ in range(top_k * 2)
                ]
            else:
                # diffuse / random head
                pairs = [
                    (int(rng.integers(0, seq_len)), int(rng.integers(0, seq_len)), float(rng.random()))
                    for _ in range(top_k * 2)
                ]
            pairs.sort(key=lambda x: x[2], reverse=True)
            for r1, r2, w in pairs[:top_k]:
                f.write(f"{r1} {r2} {w:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/tmp/fake_attn")
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--layers", nargs="+", type=int, default=[45, 46, 47])
    parser.add_argument("--residues", nargs="+", type=int, default=[10, 18, 30])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for layer in args.layers:
        # MSA row
        path = os.path.join(args.out_dir, f"msa_row_attn_layer{layer}.txt")
        write_layer_file(path, args.n_heads, args.seq_len, args.top_k, layer, seed=layer)
        print(f"[wrote] {path}")

        # Triangle start (one file per residue)
        for res in args.residues:
            path = os.path.join(args.out_dir, f"triangle_start_attn_layer{layer}_residue_idx_{res}.txt")
            write_layer_file(path, args.n_heads, args.seq_len, args.top_k, layer, seed=layer + res)
            print(f"[wrote] {path}")

    print(f"\nDone. {len(args.layers) * (1 + len(args.residues))} files in {args.out_dir}")
    print("\nNow run:")
    print(f"  cd attention_heatmaps/web && python app.py --attn_dir {args.out_dir} --protein TEST")


if __name__ == "__main__":
    main()
