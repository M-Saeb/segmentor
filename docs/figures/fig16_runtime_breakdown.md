# Figure 16 — Runtime Breakdown

**Referenced in**: Section V-C  
**Type**: Table (replace with bar chart in final paper)  
**Status**: ⚠️ Replace `—` cells with measured wall-clock times.

## Recommended IEEE Format

**Diagram type**: Horizontal grouped bar chart (pgfplots) — preferred over table for visual impact

Once the timing numbers are available, render a `pgfplots` `xbar` chart with one bar group per pipeline stage (Pre-processing, Segmentation, VLM Clustering, Exporting) and individual bars per sub-operation, with time (seconds) on the x-axis. Use a legend for sub-operation colors. If timing variance is the primary message (e.g., VLM inference dominates), a horizontal bar chart makes the bottleneck immediately obvious at a glance — more impactful than a table for a results section.

If the table format is preferred (e.g., for precise numbers), use an IEEE `\begin{table}` with `booktabs` rules and a double horizontal rule before the **Stage total** rows to visually group sub-operations. Include hardware and dataset specs in a table footnote.

**LaTeX approach**: `\usepackage{pgfplots}` with `\begin{axis}[xbar, xlabel={Time (s)}, ytick=data, yticklabels={...}]`. For the table variant: `\begin{table*}` (two-column span) with `\cmidrule` between stage groups.

*Hardware: [TODO — CPU model, RAM, GPU if any]*  
*Dataset: [TODO — scanner, total point count, scene type]*

| Stage | Sub-operation | Key parameters | Time (s) |
|---|---|---|:---:|
| **① Pre-processing** | Noise removal | nb_neighbors=20, std_ratio=2.0 | — |
| | Voxel downsample + attr transfer | voxel_size=0.2 m | — |
| | KD-tree reorder + write LAS | target=25,000 pts/tile | — |
| | **Stage total** | | — |
| **② Segmentation** | Ground RANSAC (preview) | distance_threshold=0.1, 10 iter | — |
| | Ceiling RANSAC (preview) | facing_up_threshold=0.1, ≤100 iter | — |
| | Wall RANSAC (preview) | rotation_threshold=0.9, ≤100 iter | — |
| | Full-file chunk processing | chunk_size=25,000 pts | — |
| | **Stage total** | | — |
| **③ VLM Clustering** | Merge all chunks | — | — |
| | DBSCAN | eps=0.45, min_points=15 | — |
| | Original resolution restore | single-pass over raw LAS | — |
| | Image generation (per cluster) | 4 views, 640×480 | — |
| | VLM inference (per cluster) | gemma3:4b, localhost | — |
| | **Stage total** | | — |
| **④ Exporting** | Read full 3_post_process.las | — | — |
| | Ground mesh (2 passes) | radii=[1, 3] m | — |
| | Ceiling mesh (2 passes) | radii=[1, 3] m | — |
| | Wall DBSCAN + meshes (1 pass/cluster) | ε=1 m, radius=1 m | — |
| | Car meshes (1 pass/cluster) | radius=1 m | — |
| | **Stage total** | | — |
| **Pipeline total** | | | — |
