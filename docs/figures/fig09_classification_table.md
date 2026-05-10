# Figure 9 — Classification Code Table with Color Legend

**Referenced in**: Section III-I  
**Type**: Table with color legend

## Recommended IEEE Format

**Diagram type**: IEEE `\begin{table}` with color swatches

Replace the emoji color indicators with small `\colorbox{<color>}{\phantom{XX}}` rectangles in the Color column. Use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`) and a horizontal rule between codes 4 and 5 to visually separate the *deterministic* codes (1–4) from the *VLM-assigned* codes (5–8). Align the Code column center, Color column center, and the remaining columns left. Add a footnote below the table for the three notes.

**LaTeX approach**: `\begin{table}[t]` with `\usepackage{booktabs, xcolor, colortbl}`. Define named colors: `\definecolor{clsGround}{RGB}{34,139,34}` etc. Use `\rowcolor{gray!8}` for alternating row shading if desired. Fits in one column.

---

| Code | Color | Constant | Semantic label | Stage assigned |
|:---:|:---:|---|---|---|
| 1 | ⬜ Gray | `PCB_CLASSIFICATION_INIT` | Unclassified (initial state) | Pre-processing reset |
| 2 | 🟩 Green | `PCB_CLASSIFICATION_GROUND` | Ground / floor surface | Segmentation |
| 3 | 🟦 Blue | `PCB_CLASSIFICATION_WALL` | Vertical wall surface | Segmentation |
| 4 | 🟧 Orange | `PCB_CLASSIFICATION_CEILING` | Ceiling / overhead surface | Segmentation |
| 5 | 🟫 Brown | `PCB_CLASSIFICATION_STRUCTURE` | Non-object structure (pillar, curb, bollard …) | VLM Clustering |
| 6 | 🟨 Yellow | `PCB_CLASSIFICATION_OBJECTS` | Object cluster *(transient — pre-VLM)* | VLM Clustering |
| 7 | 🩵 Cyan | `PCB_CLASSIFICATION_CAR` | Vehicle identified by VLM | VLM Clustering |
| 8 | 🟪 Purple | `PCB_CLASSIFICATION_OTHER_OBJECTS` | Non-vehicle object (chair, desk, bin …) | VLM Clustering |

**Notes:**

- Codes **1–4** are assigned by deterministic geometric RANSAC algorithms and do not depend on AI inference.
- Code **6** is a transient intermediate state applied by DBSCAN before VLM inference runs. After inference, every code-6 point is reclassified to 5, 7, or 8.
- The **Exporting stage** consumes codes **2** (ground), **3** (wall), **4** (ceiling), and **7** (car) to reconstruct triangle meshes. Codes 5, 8 are preserved in the LAS but not meshed by default.
