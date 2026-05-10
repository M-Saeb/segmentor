# Figure 1 — High-level MVC Architecture Diagram

**Referenced in**: Section III-B  
**Type**: Block/layer diagram

## Recommended IEEE Format

**Diagram type**: Three-layer block diagram (TikZ)

Draw three horizontal bands labeled **View**, **Controller**, and **Model** from top to bottom. Inside each band, place four equally-spaced labeled rectangles — one per pipeline stage (Pre-processing, Segmentation, VLM Clustering, Exporting). Connect vertically aligned boxes across layers with bidirectional arrows labeled *"user events"* (View→Controller) and *"calls / results"* (Controller↔Model). Use a vertical stage label on the left margin (①–④) to group columns.

**LaTeX approach**: TikZ with `\tikzstyle{layer}` rectangles and `\draw[->]` connectors. Keep to black-and-white; distinguish bands with light gray fills (`fill=gray!10`). Fits in one column (~8 cm wide).

---

```mermaid
graph TB
    subgraph VIEWER["VIEWER  (modules/viewers/)"]
        direction TB
        V1["PreProcessingViewer\npre_process/main.py"]
        V2["PipelineViewer\npipeline/main.py"]
        V3["PostProcessingViewer\npost_process/main.py"]
        V4["(headless — no viewer)"]
    end

    subgraph CONTROLLER["CONTROLLER  (modules/controllers/)"]
        direction TB
        C1["PreProcessingController\npre_process/main.py"]
        C2["PipelineController\npipeline/main.py"]
        C3["PostProcessingController\npost_process/main.py"]
        C4["citygml_export.py\n(script)"]
    end

    subgraph MODEL["MODEL  (modules/models/)"]
        direction TB
        M1["pcd_tools · las_tools\npre_process_tools"]
        M2["pcd_tools · las_tools\nLasPipelineWrapper"]
        M3["post_processing_tools\nvlm_tools · cluster_tools"]
        M4["mesh_tools · cityjson_export\noutput_tools"]
    end

    subgraph STAGES[" "]
        S1["① Pre-processing"]
        S2["② Segmentation"]
        S3["③ VLM Clustering"]
        S4["④ Exporting"]
    end

    S1 --- V1 & C1 & M1
    S2 --- V2 & C2 & M2
    S3 --- V3 & C3 & M3
    S4 --- V4 & C4 & M4

    V1 -- "user events" --> C1
    C1 -- "algorithm calls" --> M1
    M1 -- "results" --> C1
    C1 -- "display update" --> V1

    V2 -- "user events" --> C2
    C2 -- "algorithm calls" --> M2
    M2 -- "results" --> C2
    C2 -- "display update" --> V2

    V3 -- "user events" --> C3
    C3 -- "algorithm calls" --> M3
    M3 -- "results" --> C3
    C3 -- "display update" --> V3

    C4 -- "calls" --> M4
```
