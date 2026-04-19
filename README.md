# Vizfold Foundations

This repository has two main components:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## Visualizations

Three complementary approaches are available:

| Approach | Location | Best for |
|---|---|---|
| Arc diagrams | `visualize_attention_arc_diagram_demo_utils.py` | Single head, sequence-space view |
| PyMOL 3D overlays | `visualize_attention_3d_demo_utils.py` | Structure-space, one head at a time |
| **Heatmaps (new)** | `attention_heatmaps/` | **Comparing all heads in one layer** |

### Heatmap module (`attention_heatmaps/`)

Addresses the scale problem: arc diagrams and PyMOL overlays show one head at a time, making cross-head comparison hard. The new module provides:

- **Small-multiples grid** — every head in one layer rendered as an N×N heatmap on a shared color scale. Normalization (global-max, row-softmax) and percentile thresholding are configurable.
- **Aggregated heatmap** — collapse all heads via mean / max / std. The std view highlights *head specialisation*: positions with high variance are attended to differently across heads.
- **Per-head metrics** — entropy (diffuse vs. focused), sparsity, diagonal concentration (local vs. long-range), mean attention distance, and max weight. Use these to rank heads before plotting.
- **Interactive web app** — Flask server lets you switch layer/attention-type, adjust normalization and threshold live, sort heads by metric, and download PNGs.

#### Quick start (CLI)

```bash
pip install -r attention_heatmaps/requirements.txt

# Batch-generate all layers found in an attention directory
python attention_heatmaps/generate_heatmaps.py \
    --attn_dir /path/to/attn_maps \
    --output_dir outputs/heatmaps_6KWC \
    --protein 6KWC \
    --layer 47 \
    --normalization global_max \
    --threshold_pct 50 \
    --cols 4
```

#### Interactive web app

```bash
cd attention_heatmaps/web
python app.py \
    --attn_dir /path/to/attn_maps \
    --protein 6KWC \
    --default_layer 47
# Open http://localhost:5050
```

#### Notebook demo

See `notebooks/heatmap_demo.ipynb` for a step-by-step walkthrough.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
