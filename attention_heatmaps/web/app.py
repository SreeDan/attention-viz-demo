"""
Lightweight Flask app for interactive attention-head exploration.

Usage
-----
    cd attention_heatmaps/web
    python app.py --attn_dir /path/to/attn_maps --protein 6KWC

Then open http://localhost:5050 in a browser.

The app reads the same top-k text files produced by OpenFold's
save_attention_topk() function and lets you:
  - Pick attention type (msa_row / triangle_start) and layer
  - Choose normalization and threshold
  - View a small-multiples heatmap for all heads
  - View an aggregated heatmap (mean / max / std)
  - Inspect per-head metrics sorted by any column
  - Download any generated PNG
"""

import argparse
import io
import json
import os
import sys
import tempfile
import glob as glob_mod

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    abort,
)

# Allow running from the web/ subdirectory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from attention_heatmaps.heatmap_utils import (
    load_attention_matrix,
    plot_head_heatmaps,
    plot_aggregated_heatmap,
)
from attention_heatmaps.head_metrics import compute_head_metrics

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global config (set from CLI args at startup)
# ---------------------------------------------------------------------------
CONFIG: dict = {}


def _discover_layers(attn_dir: str) -> dict[str, list[int]]:
    """Return {attention_type: [layer_indices]} from files present in attn_dir."""
    result: dict[str, list[int]] = {}

    msa_files = glob_mod.glob(os.path.join(attn_dir, "msa_row_attn_layer*.txt"))
    if msa_files:
        layers = sorted({
            int(os.path.basename(f).split("layer")[1].split(".txt")[0])
            for f in msa_files
        })
        result["msa_row"] = layers

    tri_files = glob_mod.glob(os.path.join(attn_dir, "triangle_start_attn_layer*.txt"))
    if tri_files:
        # triangle files encode both layer and residue; group by layer
        layers = sorted({
            int(os.path.basename(f).split("layer")[1].split("_residue")[0])
            for f in tri_files
        })
        result["triangle_start"] = layers

    return result


def _get_file_path(attn_type: str, layer: int, residue: int | None = None) -> str:
    attn_dir = CONFIG["attn_dir"]
    if attn_type == "msa_row":
        return os.path.join(attn_dir, f"msa_row_attn_layer{layer}.txt")
    if attn_type == "triangle_start":
        res = residue if residue is not None else CONFIG.get("default_residue", 0)
        return os.path.join(attn_dir, f"triangle_start_attn_layer{layer}_residue_idx_{res}.txt")
    abort(400, f"Unknown attention type: {attn_type}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    layers = _discover_layers(CONFIG["attn_dir"])
    return render_template(
        "index.html",
        protein=CONFIG["protein"],
        layers_json=json.dumps(layers),
        default_layer=CONFIG.get("default_layer", 47),
    )


@app.route("/api/layers")
def api_layers():
    return jsonify(_discover_layers(CONFIG["attn_dir"]))


@app.route("/api/metrics")
def api_metrics():
    attn_type = request.args.get("attn_type", "msa_row")
    layer = int(request.args.get("layer", CONFIG.get("default_layer", 47)))
    residue = request.args.get("residue", None)
    residue = int(residue) if residue is not None else None

    file_path = _get_file_path(attn_type, layer, residue)
    if not os.path.exists(file_path):
        abort(404, f"File not found: {file_path}")

    matrices, _ = load_attention_matrix(file_path)
    metrics = compute_head_metrics(matrices)

    # Flatten for easy table display
    rows = [{"head": k, **v} for k, v in sorted(metrics.items())]
    return jsonify(rows)


@app.route("/api/heatmap")
def api_heatmap():
    """
    Returns a PNG for the small-multiples or aggregated heatmap.
    Query params:
        attn_type, layer, residue, mode (grid|agg), agg (mean|max|std),
        norm, threshold_pct, cols, highlight_residue, cmap
    """
    attn_type = request.args.get("attn_type", "msa_row")
    layer = int(request.args.get("layer", CONFIG.get("default_layer", 47)))
    residue = request.args.get("residue", None)
    residue = int(residue) if residue is not None else None
    mode = request.args.get("mode", "grid")          # "grid" or "agg"
    agg = request.args.get("agg", "mean")
    norm = request.args.get("norm", "global_max")
    threshold_pct = float(request.args.get("threshold_pct", 0))
    cols = int(request.args.get("cols", 4))
    highlight_residue = request.args.get("highlight_residue", None)
    highlight_residue = int(highlight_residue) if highlight_residue else None
    cmap = request.args.get("cmap", "viridis")

    file_path = _get_file_path(attn_type, layer, residue)
    if not os.path.exists(file_path):
        abort(404, f"Attention file not found: {file_path}")

    matrices, _ = load_attention_matrix(file_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if mode == "agg":
            plot_aggregated_heatmap(
                matrices,
                output_path=tmp_path,
                protein=CONFIG["protein"],
                layer_idx=layer,
                attention_type=attn_type,
                agg=agg,
                cmap=cmap,
                highlight_residue=highlight_residue,
            )
        else:
            plot_head_heatmaps(
                matrices,
                output_path=tmp_path,
                protein=CONFIG["protein"],
                layer_idx=layer,
                attention_type=attn_type,
                normalization=norm,
                threshold_pct=threshold_pct,
                cols=cols,
                cmap=cmap,
                highlight_residue=highlight_residue,
            )

        with open(tmp_path, "rb") as f:
            img_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return send_file(
        io.BytesIO(img_bytes),
        mimetype="image/png",
        download_name=f"{CONFIG['protein']}_{attn_type}_layer{layer}_{mode}.png",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attention head explorer web app")
    parser.add_argument("--attn_dir", required=True, help="Directory with top-k attention text files")
    parser.add_argument("--protein", default="protein", help="Protein name / PDB ID (for labels)")
    parser.add_argument("--default_layer", type=int, default=47)
    parser.add_argument("--default_residue", type=int, default=0)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    CONFIG.update(vars(args))
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
