# Figure 10 — RANSAC Floor Segmentation Example

**Referenced in**: Section IV-A.2  
**Type**: Side-by-side point cloud renders  
**Status**: ⚠️ Requires actual screenshots from the Open3D visualizer.

## Recommended IEEE Format

**Diagram type**: Dual `\subfloat` figure

Place two equally-sized images side by side using `\subfloat[(a) Input RGB point cloud]{\includegraphics[width=0.48\linewidth]{fig10a.png}}` and `\hfill` before `\subfloat[(b) After ground RANSAC]{\includegraphics[width=0.48\linewidth]{fig10b.png}}`. Both renders should use the same camera viewpoint and zoom level to make the classification change visually unambiguous. Add the key parameter values (`distance\_threshold=0.1`, `num\_iterations=10`) in the overall figure caption rather than annotating the image itself.

**LaTeX approach**: `\usepackage{subfig}` (or `subcaption`). Crop screenshots tightly to the point cloud bounding box before including — white or black background matches IEEE print better than Open3D's default gray. Fits in one column (each subfigure ~3.5 cm wide).

## Layout

```
┌──────────────────────────────┬──────────────────────────────┐
│                              │                              │
│   (a) Input point cloud      │   (b) After ground RANSAC    │
│       (RGB colors)           │       (classification view)  │
│                              │                              │
│   All points colored by      │   ■ GREEN  — Ground points   │
│   original RGB from LAS      │   □ GRAY   — Unclassified    │
│                              │                              │
│                              │   Params used:               │
│                              │   distance_threshold = 0.1   │
│                              │   num_iterations     = 10    │
│                              │   N floor planes found: ?    │
└──────────────────────────────┴──────────────────────────────┘
```

## How to produce this screenshot

1. Run the **Segmentation** stage GUI: `python main.py pipeline config.yml`
2. In the viewer, click **Apply RGB Colors** → screenshot → save as `fig10a.png`
3. Click **Run Ground Segmentation**, then **Apply Classification Colors** → screenshot → save as `fig10b.png`
4. Composite side-by-side (e.g., using ImageMagick: `convert +append fig10a.png fig10b.png fig10.png`)
