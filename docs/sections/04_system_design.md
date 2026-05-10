# III. System Design and Architecture

## A. Design Goals

Segmentor was designed around four requirements that collectively drove every architectural decision:

1. **Scalability**: LAS files from terrestrial scanners routinely exceed 100 million points and do not fit in GPU or even CPU memory at full resolution. The system must process files in streaming chunks without loading the entire dataset.

2. **Interactivity**: A practitioner must be able to preview the effect of each algorithm on a representative subset before committing to a multi-hour full-file run. Parameter changes must be previewed instantly without re-reading the file.

3. **Separation of concerns**: The processing logic must be cleanly separable from the GUI so that the same algorithms can be invoked from shell scripts, unit tests, or external orchestration systems.

4. **Extensibility**: New segmentation stages, output formats, and AI backends should be addable without restructuring the existing codebase.

## B. Overall Architecture

[FIGURE PLACEHOLDER — Figure 1: High-level MVC architecture diagram]

*The figure should show three vertical lanes (Model, Controller, Viewer) with the four processing stages (Pre-processing, Segmentation, VLM Clustering, Exporting) as horizontal groupings. Arrows should show data flow from LAS files through each layer. The Exporting stage has no GUI viewer (headless-only) and should be annotated accordingly.*

Segmentor follows a strict **Model–View–Controller (MVC)** pattern. The three layers and their responsibilities are:

| Layer | Location | Responsibility |
|---|---|---|
| **Model** | `modules/models/` | LAS I/O, point cloud algorithms, AI inference, output generation |
| **Controller** | `modules/controllers/` | Orchestration, pipeline state, parameter binding |
| **Viewer** | `modules/viewers/` | Open3D GUI windows, user interaction, display |

The system has four processing stages. The first three have a dedicated Controller–Viewer GUI pair sharing the same Model layer. The fourth (Exporting) is headless-only, invoked directly as a CLI script:

| Stage | GUI entry point | GUI Viewer class | Controller class | Headless script |
|---|---|---|---|---|
| Pre-processing | `main.py pre_process` | `PreProcessingViewer` | `PreProcessingController` | `scripts/pre_processing.py` |
| Segmentation | `main.py pipeline` | `PipelineViewer` | `PipelineController` | `scripts/run_pipeline_from_file.py` |
| VLM Clustering | `main.py post_process` | `PostProcessingViewer` | `PostProcessingController` | `scripts/post_processing.py` |
| Exporting | *(none)* | *(none)* | *(inline in script)* | `scripts/citygml_export.py` |

## C. End-to-End Data Flow

[FIGURE PLACEHOLDER — Figure 2: End-to-end data flow diagram]

*The figure should show a horizontal pipeline with four stage boxes connected by arrows. Each arrow is annotated with the file(s) it represents. File names match exactly what the system writes to disk.*

The system produces a versioned artifact hierarchy inside a single output folder:

```
<output_folder>/
├── 1_pre_processing.las        ← Stage 1: noise-filtered, downsampled, spatially reordered cloud
├── 1_pre_processing_index.json ← Stage 1: KD-tree chunk boundary index (bin start/end indices)
├── 2_pipeline.las              ← Stage 2: classified points (ground / wall / ceiling labels)
├── 2_pipeline.yml              ← Stage 2: serialized process pipeline (reproducibility)
├── 3_post_process.las          ← Stage 3: object-classified points (car / other-object labels)
├── 3_objects.yml               ← Stage 3: VLM object registry (cluster ID, label, bounds)
├── vlm_object_traces/          ← Stage 3: isometric PNGs generated per cluster for VLM input
│   ├── 01.png
│   └── ...
└── citygml.city.json           ← Stage 4: CityJSON v2 mesh export
```

Each stage reads the output of the previous stage, so stages can be run independently (e.g., re-running only the Exporting stage after adjusting mesh parameters) without repeating earlier steps.

## D. Configuration System

All system parameters are controlled through a single YAML configuration file (`config.yml`). At startup, the file is parsed by the `SegmentorConfig` singleton class (`modules/utils/config.py`). The class enforces a process-wide singleton constraint: once initialized with a file path, subsequent accesses from anywhere in the codebase return the same validated instance. Class-level attributes serve as defaults that are overridden only by values present in the YAML, making explicit every parameter that departs from the standard behaviour.

[FIGURE PLACEHOLDER — Figure 3: SegmentorConfig singleton class diagram with key attributes]

Validation is performed at construction time, rejecting invalid combinations (e.g., a `facing_up_threshold` outside (0, 1] for ceiling detection) with descriptive error messages before any processing begins.

Key configurable parameters include:

| Parameter group | Key fields | Stage used |
|---|---|---|
| `downsample` | `voxel_size` | Pre-processing |
| `filter_noise` | `nb_neighbors`, `std_ratio` | Pre-processing |
| `ground_segmentation` | `method`, `ransac_params`, `csf_params` | Segmentation |
| `ceiling_segmentation` | `facing_up_threshold`, `min_inlier_density` | Segmentation |
| `wall_segmentation` | `rotation_threshold` | Segmentation |
| `objects_segmentation` | `eps`, `min_points` | VLM Clustering |
| `vlm` | `model`, `host`, `views_count` | VLM Clustering |
| `cityjson_export` | `enable`, `ball_radius`, `max_faces` | Exporting (pipeline-integrated path) |
| `result` | `output_folder_path` | All stages |

## E. Stage 1 — Pre-processing

[FIGURE PLACEHOLDER — Figure 4: Pre-processing stage flow diagram]

*Show: Raw LAS → merge chunks → statistical outlier removal → voxel downsampling → KD-tree binning → spatial reorder → write 1_pre_processing.las + index.json*

The pre-processing stage prepares the raw point cloud for spatially coherent chunk-by-chunk segmentation. Its design motivates three specific transformations:

**Statistical Outlier Removal.** Airborne LiDAR returns from vegetation, atmospheric artifacts, and multipath reflections produce isolated points far from the main scene surface. These outliers corrupt plane-fitting statistics if left in. Open3D's `remove_statistical_outlier` function removes points whose mean distance to their `nb_neighbors` nearest neighbours exceeds `std_ratio` standard deviations above the cloud-wide mean.

**Voxel Downsampling.** Dense scans produce redundant samples on flat surfaces. Voxel downsampling replaces each occupied grid cell with the centroid of its contained points, reducing point count while preserving structure. To maintain consistency between the downsampled geometry and the original LAS attribute dimensions (color channels, intensity, GPS time, etc.), the system uses a KD-tree nearest-neighbour lookup: each centroid inherits attributes from its closest original point.

**Spatial Reordering.** By default, LAS files store points in scan-line order (time-sequential capture). Processing such a file chunk-by-chunk means that each chunk contains geometrically distant points from different parts of the scene — catastrophic for local algorithms like RANSAC and DBSCAN. To fix this, the system constructs a uniform 3D grid of seed points whose count equals `⌈N / target_chunk_size⌉`. Each point is assigned to its nearest seed (a nearest-neighbour query against a seed KD-tree), and points within each bin are further sorted by their Z coordinate. The permutation is applied to both coordinates and all attribute arrays before writing. The bin boundaries (start and end global indices for each non-empty bin) are serialized to `1_pre_processing_index.json`.

The resulting spatial index is consumed by `LasIteratorKDTree` during the Segmentation stage, which reads each bin's points in a single streaming pass of the LAS file, producing geometrically local chunks.

## F. Stage 2 — Segmentation

[FIGURE PLACEHOLDER — Figure 5: Segmentation stage state machine diagram]

*Show states: Stage N preview (run → apply → display) ↔ confirm → Stage N+1 → ... → Finalize → full run over all chunks*

The Segmentation stage implements an interactive, stage-by-stage parameter tuning workflow for identifying the three dominant structural surface types: **ground**, **ceiling**, and **walls**. Its architecture separates the concern of "exploring parameters on a preview chunk" (setup pipeline) from "applying confirmed parameters to all chunks" (process pipeline).

### F.1 Setup Pipeline

A `Pipeline` object (`modules/utils/pipeline.py`) holds an ordered list of `PipelineStage` instances. Each stage carries:
- A human-readable `name` (e.g., `"Ground Segmentation"`)
- A `func` callable that wraps the underlying algorithm
- A `params` dictionary holding current parameter values
- An `enabled` flag (default `False`, set to `True` only when the user confirms the stage)

The `PipelineController` constructs the setup pipeline at startup from the enabled stages declared in `config.yml`. When the GUI's **Run Stage** button is pressed, the controller:
1. Restores the snapshot of the pre-stage chunk (erasing any effects of a previous preview run)
2. Merges stored params with any user-modified values from the input panel
3. Calls `stage.func(**call_kwargs)`, which mutates the in-memory `(pcd, points_attributes)` pair
4. Returns the updated point cloud to the viewer for display

The **snapshot mechanism** (`LasSnapshot`) serializes the current `(pcd, points_attributes)` pair to a temporary file before the first preview of each stage. This allows the user to freely re-run the stage with different parameters without effects accumulating across multiple runs.

When the user clicks **Confirm Stage**, the current preview result becomes the permanent state for the next stage, the stage is marked `enabled=True`, and the next `PipelineStage` is activated. Clicking **Skip Stage** marks the stage `enabled=False` and advances without retaining any effect.

### F.2 Process Pipeline

Once all stages are reviewed, clicking **Finalize & Run Full Processing** triggers `build_process_pipeline_from_setup()`, which clones each `enabled` setup stage into a process stage with its parameters frozen at their confirmed values. The controller then iterates over all chunks from `LasIteratorKDTree`, applies the process stages in sequence, and appends each processed chunk to `2_pipeline.las`.

The process pipeline is also serializable to YAML (`2_pipeline.yml`), enabling headless re-runs via `scripts/run_pipeline_from_file.py` without the GUI. This supports reproducibility: the YAML captures the full configuration of each stage used for a given run.

[FIGURE PLACEHOLDER — Figure 6: Sequence diagram showing user interaction from "Run Stage" to "Finalize"]

*Show user clicks → viewer method → background thread → controller method → model function → GUI update*

### F.3 Threading Model

Open3D's GUI framework requires that all widget mutations happen on the main thread. Long-running operations (RANSAC, full-file processing) run in a daemon background thread. Completion callbacks are posted back to the main thread via `gui.Application.instance.post_to_main_thread()`. A modal "loading" dialog is shown during long operations to communicate progress.

## G. Stage 3 — VLM Clustering

[FIGURE PLACEHOLDER — Figure 7: VLM Clustering stage pipeline diagram]

*Show: 2_pipeline.las → load+merge all chunks → DBSCAN on INIT points only → per-cluster multi-view render → VLM API call → classify → write 3_post_process.las + 3_objects.yml*

The VLM Clustering stage takes the structurally segmented point cloud from Stage 2 and focuses on the remaining **unclassified** points — those not identified as ground, wall, or ceiling. It applies DBSCAN to group these residual points into discrete object clusters and then uses a Vision Language Model to assign semantic labels to each cluster.

This stage is distinct from the Segmentation stage in three important ways: (1) it operates on the entire merged point cloud rather than chunk-by-chunk; (2) it introduces object-level clustering rather than surface detection; and (3) it invokes an external AI service for semantic labelling.

**Merge step.** All chunks from `2_pipeline.las` are concatenated in memory. This is intentional: DBSCAN requires a global view of the remaining unclassified points to form clusters that may span multiple spatial tiles from the Segmentation stage.

**DBSCAN step.** Clustering is applied exclusively to points with classification `INIT` (code 1) — those not yet assigned to any structural surface. Open3D's `cluster_dbscan` is used with user-tunable `eps` (neighbourhood radius) and `min_points` (density threshold). Cluster IDs are assigned globally using the thread-safe `ClustersManager` singleton, which allocates contiguous ID blocks atomically. These IDs are stored per-point in the `Clusters` extra dimension of the output LAS file.

**VLM step.** For each cluster with at least 50 points:
1. The cluster's bounding box in the processed cloud is computed.
2. The **original** (pre-downsampled) LAS file is scanned once in a single streaming pass, collecting full-resolution points from each cluster's bounding region. This "original resolution restoration" produces richer RGB images for the VLM than the downsampled cluster PCD.
3. `take_rich_media_screenshots()` renders the cluster from `views_count` (default: 4) evenly spaced isometric viewpoints using Open3D's `OffscreenRenderer`.
4. The PNG files are base64-encoded and submitted to the configured Ollama VLM endpoint with a structured prompt requesting a JSON response: `{"label": "...", "description": "..."}`.
5. The label is mapped to a classification code (car → 7, structure → 5, other → 8).

The object registry (`3_objects.yml`) stores the cluster ID, AI label, and axis-aligned bounding box for each identified cluster, enabling downstream systems to retrieve object locations without re-reading the LAS file.

## H. Stage 4 — Exporting

[FIGURE PLACEHOLDER — Figure 8: Exporting stage component diagram]

*Show: 3_post_process.las → read full LAS → split by class → [ground path | ceiling path | wall DBSCAN path | car path] → iterative mesh refinement → write_cityjson_mesh_part → citygml.city.json. Show the iteration loop with ball radii [1, 3, 5, 6, 9] annotated.*

The Exporting stage converts the fully classified point cloud from Stage 3 into a CityJSON v2 file (`citygml.city.json`). It is invoked headlessly via `scripts/citygml_export.py` and produces one or more `CityObject` entries for each surface type.

The stage processes four surface classes independently:

**Ground and Ceiling**: The classified points are extracted by class code and passed directly to the iterative mesh reconstruction pipeline.

**Walls**: Wall points are first clustered by DBSCAN (ε = 1 m, min_points = 50) to separate individual wall faces. Each cluster is then reconstructed independently. Clusters producing meshes below `min_surface_size` vertices/triangles (default: 200) are discarded as noise. Each accepted wall mesh is written as a separate `CityObject` with key `wallSurface_<idx>`.

**Cars**: Vehicle clusters are identified by the `Clusters` attribute (from Stage 3) cross-referenced with the `CAR` classification code. Each car cluster is reconstructed as its own `CityObject` (`carSurface_<idx>`).

**Iterative Mesh Refinement.** For surfaces requiring higher quality (e.g., ground and ceiling with more iterations), the export script applies a multi-pass approach using `ball_neighbourhood_mesh_v2`:

```
for each iteration i in [0 .. N-1]:
    mesh = create_ball_neighbourhood_mesh(current_pcd,
               ball_radius=radii[i],         # [1, 3, 5, 6, 9]
               min_neighbours=min_nbrs[i])   # [10, 7, 4, 2, 1]
    if i < N-1:
        current_pcd = mesh.sample_points_uniformly(0.25 × |current_pcd|)
```

Each pass operates on a progressively sparser point cloud (25% of the previous pass) at a larger ball radius, progressively filling gaps and smoothing the surface while the first pass captures fine detail at tight radius.

**CityJSON structure.** The output file is initialized with the point cloud's axis-aligned bounding box as `geographicalExtent`. Each mesh part is appended incrementally without loading the full JSON into memory. Ground, ceiling, and wall surfaces are exported as `Building` city objects with `MultiSurface` geometry at LoD 2. Car meshes are exported as `GenericCityObject`.

The Exporting stage can also be triggered automatically at the end of the Segmentation stage (if `cityjson_export.enable: true` in `config.yml`), using a simpler single-pass reconstruction directly from the `2_pipeline.las` file. The standalone `scripts/citygml_export.py` provides the full multi-pass, multi-class export from `3_post_process.las`.

## I. Classification Code Scheme

[FIGURE PLACEHOLDER — Figure 9: Classification code table with color legend]

The system defines eight point classification codes stored in the LAS `classification` dimension:

| Code | Constant | Semantic meaning | Stage assigned |
|---|---|---|---|
| 1 | `PCB_CLASSIFICATION_INIT` | Unclassified (initial) | Pre-processing reset |
| 2 | `PCB_CLASSIFICATION_GROUND` | Ground / floor surface | Segmentation |
| 3 | `PCB_CLASSIFICATION_WALL` | Vertical wall surface | Segmentation |
| 4 | `PCB_CLASSIFICATION_CEILING` | Ceiling / overhead surface | Segmentation |
| 5 | `PCB_CLASSIFICATION_STRUCTURE` | Non-object structure (pillar, curb, etc.) | VLM Clustering |
| 6 | `PCB_CLASSIFICATION_OBJECTS` | Object cluster (pre-VLM, intermediate) | VLM Clustering |
| 7 | `PCB_CLASSIFICATION_CAR` | Vehicle identified by VLM | VLM Clustering |
| 8 | `PCB_CLASSIFICATION_OTHER_OBJECTS` | Non-vehicle object (chair, etc.) | VLM Clustering |

Codes 1–4 are produced deterministically by the geometric RANSAC algorithms in the Segmentation stage. Code 6 is a transient intermediate label assigned during DBSCAN; codes 5, 7, and 8 replace it after VLM identification. The Exporting stage consumes codes 2, 3, 4, and 7 to reconstruct meshes for each surface type.
