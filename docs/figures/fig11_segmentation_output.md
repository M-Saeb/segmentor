# Figure 11 — Full Segmentation Stage Output

**Referenced in**: Section IV-C  
**Type**: Single point cloud render  
**Status**: ⚠️ Requires actual screenshot from the Open3D visualizer.

## Recommended IEEE Format

**Diagram type**: Full-width `\includegraphics` with inset legend

Use a single full-column-width image showing the classified point cloud with clearly distinct colors for Ground (green), Wall (blue), Ceiling (orange), and INIT remainder (gray). Overlay a small legend in the bottom-left corner — either composited into the PNG before import, or added as a TikZ overlay on top of the `\includegraphics`. Do not use the Open3D default GUI chrome (toolbar buttons etc.); screenshot only the 3D viewport.

**LaTeX approach**: `\begin{figure}[t] \includegraphics[width=\linewidth]{fig11.png} \end{figure}`. For the legend overlay, wrap in a `tikzpicture` with `\node[inner sep=0] (img) {\includegraphics[...]}` then place `\node[draw, fill=white, font=\scriptsize, anchor=south west] at (img.south west) {...legend...}`.

## Layout

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│              Segmentation Stage — Full Output            │
│                                                          │
│   All three RANSAC sub-stages complete:                  │
│   ground · ceiling · wall                                │
│                                                          │
│   ■ GREEN   — Ground / floor (code 2)                   │
│   ■ BLUE    — Walls          (code 3)                   │
│   ■ ORANGE  — Ceiling        (code 4)                   │
│   □ GRAY    — Unclassified INIT remaining (code 1)       │
│                                                          │
│   ┌── Legend ─────────────────────────────────────┐     │
│   │  ■ Ground  ■ Wall  ■ Ceiling  □ INIT          │     │
│   └────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

## How to produce this screenshot

1. Run the **Segmentation** stage GUI: `python main.py pipeline config.yml`
2. Confirm all three stages (ground → ceiling → wall)
3. Click **Apply Classification Colors**
4. Screenshot the 3D viewport → save as `fig11.png`
