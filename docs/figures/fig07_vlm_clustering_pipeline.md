# Figure 7 — VLM Clustering Stage Pipeline Diagram

**Referenced in**: Section III-G  
**Type**: Flowchart

## Recommended IEEE Format

**Diagram type**: Flowchart with decision nodes and loop-back arrow (TikZ)

Draw a top-to-bottom flowchart. Use two diamond decision nodes: one for *"INIT points exist?"* (branch to early exit) and one for *"cluster ≥ 50 pts?"* (branch to discard). After the VLM classification step, draw a loop-back arrow from *"Update classification"* back up to the size-filter diamond labeled *"next cluster"*, and a downward arrow out of the loop labeled *"all clusters done"*. Use parallelograms for the two file-output nodes at the bottom.

**LaTeX approach**: TikZ with `\node[diamond, aspect=2, draw]` for decisions, `\node[rectangle, draw]` for processes, `\node[trapezium, draw]` for file outputs. Route the loop-back arrow to the left side of the diagram with `\draw[->, rounded corners] ... -- ++(-2,0) -- ...`. Fits in one column with a narrow width.

---

```mermaid
flowchart TD
    A["📂 2_pipeline.las\n(Segmentation output)"]
    B["merge_chunks()\nLoad all chunks into RAM\nxyz array + attrs dict"]
    C{"INIT points\nexist?"}
    SKIP["write 3_post_process.las\n(unchanged)\n3_objects.yml empty"]
    D["DBSCAN\neps=0.45  min_points=15\nopen3d cluster_dbscan\non INIT subset only"]
    E["ClustersManager\nassign globally unique cluster IDs\n(thread-safe singleton)"]
    F{"cluster size\n≥ 50 pts?"}
    DISC["discard cluster\n(too small)"]
    G["build_original_pcds_per_cluster()\nSingle streaming pass over raw LAS\n→ full-resolution RGB PCD per cluster"]
    H["take_rich_media_screenshots()\nOffscreenRenderer  640×480\n4 isometric views @ 35° elevation"]
    I["identify_object_from_views()\nbase64-encode PNGs\nPOST to Ollama VLM\n(up to 3 retries)"]
    J["Parse JSON response\nlabel → classification code\n5=Structure  7=Car  8=Other"]
    K["Update classification array\nadd entry to object registry"]
    L["📄 write_identified_objects_yaml()\n3_objects.yml"]
    M["📄 write_las_merged()\n3_post_process.las"]

    A --> B --> C
    C -- "none" --> SKIP
    C -- "found" --> D --> E --> F
    F -- "no" --> DISC
    F -- "yes" --> G --> H --> I --> J --> K
    K -- "next cluster" --> F
    K --> L
    K --> M
```
