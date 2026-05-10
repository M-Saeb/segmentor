# Figure 2 — End-to-End Data Flow Diagram

**Referenced in**: Section III-C  
**Type**: Pipeline / data-flow diagram

## Recommended IEEE Format

**Diagram type**: Linear pipeline block diagram (TikZ)

Draw four rounded-rectangle stage boxes arranged left-to-right, connected by thick horizontal arrows. Label each arrow with the intermediate artifact filename (e.g., `1_pre_processing.las`). Below each stage box, add a small italicized sub-list of the two or three key operations. The raw input file sits left of stage ① and the final `citygml.city.json` sits right of stage ④.

**LaTeX approach**: TikZ `\node[draw, rounded corners, minimum width=2.2cm]` chain with `\draw[->, thick]` edges; artifact labels as `\node[font=\scriptsize\ttfamily]` above each arrow. Spans the full text width (two-column `figure*` environment).

---

```mermaid
flowchart LR
    RAW["📂 Raw LAS\n(original scan)"]

    subgraph P1["① Pre-processing"]
        direction TB
        p1a["noise removal"]
        p1b["voxel downsample"]
        p1c["KD-tree spatial reorder"]
        p1a --> p1b --> p1c
    end

    subgraph P2["② Segmentation"]
        direction TB
        p2a["RANSAC ground"]
        p2b["RANSAC ceiling"]
        p2c["RANSAC walls"]
        p2a --> p2b --> p2c
    end

    subgraph P3["③ VLM Clustering"]
        direction TB
        p3a["merge all chunks"]
        p3b["DBSCAN on INIT pts"]
        p3c["multi-view render"]
        p3d["VLM inference"]
        p3a --> p3b --> p3c --> p3d
    end

    subgraph P4["④ Exporting"]
        direction TB
        p4a["split by class"]
        p4b["iterative mesh\nreconstruction"]
        p4c["CityJSON write"]
        p4a --> p4b --> p4c
    end

    F1["1_pre_processing.las\n1_pre_processing_index.json"]
    F2["2_pipeline.las\n2_pipeline.yml"]
    F3["3_post_process.las\n3_objects.yml"]
    F4["citygml.city.json"]

    RAW --> P1 --> F1 --> P2 --> F2 --> P3 --> F3 --> P4 --> F4
```
