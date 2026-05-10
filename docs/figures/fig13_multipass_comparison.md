# Figure 13 — Multi-pass Mesh Refinement Comparison

**Referenced in**: Section IV-D.1  
**Type**: Side-by-side mesh renders  
**Status**: ⚠️ Requires actual screenshots from the Exporting stage visualizer.

## Recommended IEEE Format

**Diagram type**: Dual `\subfloat` figure

Same layout as Figure 10: two equally-sized mesh render screenshots side by side. **(a)** shows the wireframe mesh after pass 1 only (radius = 1 m) with visible holes in low-density regions. **(b)** shows the mesh after pass 2 (radius = 3 m) with holes filled and smoother coverage. Use the same camera angle for both panels so the difference is immediately visible. Annotate with the triangle count for each panel in the caption (e.g., *"Pass 1: ~N₁ triangles; after Pass 2: ~N₂ triangles"*).

**LaTeX approach**: `\subfloat[(a) Pass 1 — radius 1 m]{\includegraphics[width=0.48\linewidth]{fig13a.png}} \hfill \subfloat[(b) After Pass 2 — radius 3 m]{\includegraphics[width=0.48\linewidth]{fig13b.png}}`. Render with a white or black background for clean print reproduction.

## Layout

```
┌──────────────────────────────┬──────────────────────────────┐
│                              │                              │
│   (a) Pass 1 only            │   (b) After Pass 2           │
│   ball_radius = 1 m          │   ball_radius = 3 m          │
│   min_neighbours = 10        │   min_neighbours = 7         │
│                              │                              │
│   Fine detail captured.      │   Gaps filled.               │
│   Holes visible in           │   Surface smoother.          │
│   low-density regions.       │   Better coverage.           │
│                              │                              │
│   Triangles: ~N₁             │   Triangles: ~N₂             │
└──────────────────────────────┴──────────────────────────────┘
```

## How to produce this screenshot

Run `scripts/citygml_export.py` twice on the same wall cluster, controlling the `--wall-iterations` flag:

```bash
# Pass 1 only
python scripts/citygml_export.py --config config.yml \
    --wall-iterations 1 --output out_pass1.city.json --display true

# Two passes
python scripts/citygml_export.py --config config.yml \
    --wall-iterations 2 --output out_pass2.city.json --display true
```

Screenshot the Open3D visualizer window for each run, then composite side-by-side.
