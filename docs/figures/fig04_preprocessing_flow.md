# Figure 4 — Pre-processing Stage Flow Diagram

**Referenced in**: Section III-E  
**Type**: Flowchart

## Recommended IEEE Format

**Diagram type**: Standard ANSI/ISO flowchart (TikZ)

Use the classic flowchart symbol set: parallelograms for file I/O (raw LAS input, `.las` output, `.json` output), rectangles for processing steps (noise removal, downsample, grid build, KD-tree assign, sort). The flow is strictly linear (top-to-bottom, no branches), so a single-column layout is clean and compact. Annotate each process box with its key parameter values in a smaller font.

**LaTeX approach**: TikZ `shapes.geometric` library — `\node[trapezium]` for I/O, `\node[rectangle,draw]` for processes, all connected by `\draw[->]`. Place parameter annotations as `\node[font=\scriptsize, right=0.1cm of ...]`. Fits comfortably in one column.

---

```mermaid
flowchart TD
    A["📂 Raw LAS\ninput_file"]
    B["Read chunks\n(laspy chunk_iterator)\nchunk_size = 10 M pts"]
    C["Merge all chunks\ninto single PCD + attrs dict"]
    D["Statistical Outlier Removal\nnb_neighbors=20  std_ratio=2.0\nopen3d remove_statistical_outlier"]
    E["Voxel Downsampling\nvoxel_size=0.2 m\n+ KD-tree nearest-attr transfer"]
    F["Build 3D Seed Grid\n⌈N / target_chunk_size⌉ seeds\nuniform 3D grid"]
    G["Assign each point\nto nearest seed\nopen3d KDTreeFlann"]
    H["Sort points by\n(bin_id, Z)"]
    I["📄 Write reordered LAS\n1_pre_processing.las"]
    J["📄 Write spatial index\n1_pre_processing_index.json\nfields: start · end · bin_id · point_count"]

    A --> B --> C --> D --> E --> F --> G --> H
    H --> I
    H --> J
```
