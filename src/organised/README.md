# Organised Tracking Framework

This folder contains a notebook-friendly, modular crown tracking framework.

Design constraints:
- **No interactive UI assumptions** (Flask not used).
- **Notebook-first**: key textual summaries are returned/printed; figures are saved to disk.
- **Modular**: alignment, matching, graph building, chains, augmentation, consensus, diagnostics, and visualization writers are separate modules.

## Mathematical objects

- Orthomosaics (OMs): indexed by integer $t \in \{1,\dots,T\}$.
- Crown polygon set for OM $t$: $\{g_{t,i}\}_{i=1}^{N_t}$.
- Graph nodes: $v=(t,i)$.
- Edges: directed $((t,i)\to(t+1,j))$ with attributes including IoU, overlap fractions, centroid distance, and a similarity score.

### Overlap quantities

For polygons $g_p$ (previous) and $g_c$ (current):

- Intersection-over-union (IoU):

$$
\mathrm{IoU}(g_p,g_c) = \frac{|g_p \cap g_c|}{|g_p \cup g_c|}
$$

- Overlap fractions:

$$
\mathrm{ov}_p = \frac{|g_p \cap g_c|}{|g_p|},\quad
\mathrm{ov}_c = \frac{|g_p \cap g_c|}{|g_c|}
$$

### Consensus crown

A consensus crown is a single polygon $g^*$ per chain. The default method is a **medoid**: choose one observed polygon minimizing a weighted distance to other chain polygons.

## Where to start

Run the SIT pipeline from the notebook:
- [organised.ipynb](organised.ipynb)

Outputs are written under the configured `output_dir`:
- `reports/` (quality report + metrics JSON)
- `figures/` (diagnostic plots + chain panels + consensus panels)
- `artifacts/` (consensus crowns GeoJSON/GPKG)
