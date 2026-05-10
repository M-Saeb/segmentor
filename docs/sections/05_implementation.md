# IV. Implementation

## A. Ground Segmentation (Segmentation Stage)

Ground segmentation classifies planar floor or terrain surfaces. The system supports two interchangeable methods selected by the `ground_segmentation.method` configuration key.

### A.1 Cloth Simulation Filter (CSF)

The CSF method models the ground surface by simulating a rigid cloth draped under gravity over an inverted point cloud. The implementation (`_cloth_simulation_filter` in `pcd_tools.py`) proceeds as follows:

1. Project points onto a 2D horizontal grid of cell size `resolution` (metres).
2. Build a Digital Elevation Model (DEM) by taking the minimum Z value in each cell.
3. Initialize a cloth at height `max_z + max(1.0, height_range)`, well above the scene.
4. Iteratively apply two forces: a **smoothing** force that pulls each cloth node toward the average of its 4-neighbours (controlled by `1/rigidness`), and a **gravity** force that closes the gap between the cloth and the DEM. The cloth is prevented from passing below the DEM.
5. Iteration stops when the maximum node displacement falls below 1 mm or `max_iter` is reached.
6. Points within a height threshold (`max(0.3, resolution × threshold_factor)` metres above the cloth surface) are classified as ground.

CSF is particularly suitable for outdoor scenes with vegetation or irregular terrain where a single-plane RANSAC fit would be incorrect.

### A.2 RANSAC Floor Segmentation

The RANSAC method (`_ransac_floor_segmentation`) applies a multi-plane RANSAC loop:

```
for up to num_iterations:
    fit plane P to the remaining point cloud (Open3D segment_plane)
    if is_floor(P):
        label P's inliers as GROUND
        record P in floor_planes[]
    remove P's inliers from the remaining cloud
```

A plane passes `is_floor` if it simultaneously satisfies three conditions:

1. **Horizontal orientation**: `|normal_z| > 0.95`, ensuring the plane normal points upward.
2. **Vertical position**: The plane's Z-intercept (`−d/c`) is below the median Z value of the full cloud, excluding ceilings.
3. **Non-collision**: The plane's XY footprint does not overlap (by more than 1 m²) the footprint of any already-accepted floor plane, preventing duplicate labelling of the same surface.

[FIGURE PLACEHOLDER — Figure 10: RANSAC floor segmentation example — before/after colored points]

*Show side-by-side renders: left = raw colored point cloud, right = ground points in green, unclassified in gray.*

## B. Ceiling Segmentation (Segmentation Stage)

Ceiling segmentation (`_apply_ransac_ceiling_detection`) uses the same RANSAC multi-plane loop structure as floor detection but with two differences:

1. **Vertical position**: Ceiling planes must appear in the **upper half** of the cloud's Z histogram (Z-intercept above the median Z), filtering out floor candidates.
2. **Density gate**: A plane is accepted only if its inlier density (points per m² of convex hull area in the plane's local 2D coordinate system) exceeds `min_inlier_density`. This rejects sparse partial planes from scan edges or occlusion artifacts while accepting solid ceiling surfaces.

Only points still classified `INIT` in the inlier set are updated, preventing the algorithm from overwriting ground labels.

## C. Wall Segmentation (Segmentation Stage)

Wall segmentation (`_apply_ransac_wall_detection`) applies the same RANSAC loop with an orientation check for nearly-vertical planes:

```
is_wall = |normal_z| < (1 - rotation_threshold)
```

A wall plane is accepted when its normal is nearly horizontal (XY-dominant). The `rotation_threshold` (default 0.9) controls how strictly vertical a plane must be. Only `INIT` points among the inliers are reclassified to `WALL`.

[FIGURE PLACEHOLDER — Figure 11: Wall and ceiling segmentation example]

*Show point cloud with ground (green), wall (blue), ceiling (red), and remaining unclassified (gray) color coding.*

## D. Ball-Neighbourhood Mesh Reconstruction (Exporting Stage)

The Exporting stage requires triangulated surfaces, not point clouds. Segmentor implements a custom **ball-neighbourhood growing** algorithm specifically designed for LiDAR building surfaces. Two variants exist in the codebase:

- **v1** (`ball_neighbourhood_mesh.py`): Used internally within the Segmentation stage's CityJSON path. Triangulates each face by fan-triangulation from the first convex hull vertex.
- **v2** (`ball_neighbourhood_mesh_v2.py`): Used by the standalone Exporting stage (`scripts/citygml_export.py`). Replaces fan-triangulation with Delaunay triangulation filtered by a point-in-polygon test, producing more uniform and watertight triangles for complex surfaces.

Both variants share the same growing loop structure:

1. **Initialization**: Select a seed point (default: lowest-Z as a ground anchor). Build a KD-tree over all points for O(log n) ball queries.

2. **Growing loop** (up to `max_faces` faces):
   a. Dequeue a centre point from the queue.
   b. Query all points within ball radius `r`. If sparse (< `min_neighbours`), expand by `adaptive_factor` (1.5×) to bridge LiDAR occlusion gaps.
   c. Fit a local plane to the neighbourhood via PCA (smallest singular vector → normal).
   d. Project the neighbourhood onto the local 2D plane coordinate system.
   e. Compute the 2D convex hull; its vertices define the face boundary.
   f. **v1**: Fan-triangulate (vertex 0 connected to all consecutive pairs). **v2**: Delaunay-triangulate the boundary vertices, then keep only triangles whose centroid lies inside the convex polygon (ray-casting test).
   g. Enqueue all boundary points not yet used as face centres.

3. **Post-processing**: Merge duplicate vertices, remove degenerate triangles and non-manifold edges, recompute normals.

[FIGURE PLACEHOLDER — Figure 12: Ball-neighbourhood mesh reconstruction steps]

*Show three sub-figures: (a) raw wall point cloud, (b) ball around a seed with neighbourhood and local plane highlighted, (c) resulting mesh with triangle wireframe.*

### D.1 Iterative Multi-Pass Refinement

The Exporting stage applies a multi-pass refinement strategy on top of the base algorithm. Rather than a single fixed ball radius, the script runs the reconstruction N times (configurable per surface type, default: 2 for ground/ceiling, 1 for walls/cars) using progressively larger radii and smaller minimum-neighbour requirements:

| Pass | Ball radius (m) | Min neighbours |
|---|---|---|
| 1 | 1 | 10 |
| 2 | 3 | 7 |
| 3 | 5 | 4 |
| 4 | 6 | 2 |
| 5 | 9 | 1 |

Between passes, the mesh is uniformly point-sampled at 25% of its previous count (`mesh.sample_points_uniformly`), producing a sparser but mesh-conditioned cloud for the next pass. The first pass captures fine local geometry; subsequent passes at larger radii fill gaps caused by LiDAR occlusions or low-density regions. This progression from tight to loose radius mirrors a coarse-to-fine strategy applied inside a single growing framework.

### D.2 Per-Surface-Type Processing

| Surface | Source data | Pre-processing | Min size filter |
|---|---|---|---|
| Ground | `GROUND` classified points | None | — |
| Ceiling | `CEILING` classified points | None | — |
| Walls | `WALL` classified points | DBSCAN (ε=1 m, min_pts=50) to separate individual faces | 200 vertices/triangles |
| Cars | `CAR` classified points, per `Clusters` ID | None | 50 vertices/triangles |

Wall surfaces receive an additional DBSCAN clustering step before reconstruction. Because a scene can contain many disconnected wall faces (each wall of a room, pillars, partitions), feeding all wall points to a single growing algorithm would produce a connected but geometrically incorrect mesh. Clustering first isolates individual surfaces, and each surface is reconstructed independently.

[FIGURE PLACEHOLDER — Figure 13: Multi-pass mesh refinement illustration — pass 1 vs pass 3 result on the same wall cluster]

## E. VLM Object Identification (VLM Clustering Stage)

[FIGURE PLACEHOLDER — Figure 14: VLM pipeline diagram for a single cluster]

*Show: cluster points → OffscreenRenderer (4 views) → 4 PNG files → base64 encode → Ollama API → JSON response → classification code*

The VLM identification pipeline (`identify_object_from_views` in `vlm_tools.py`) operates as follows:

1. The cluster's point cloud is loaded at the **original scan resolution** (pre-downsampling) by scanning the original LAS file once in a single streaming pass collecting points by bounding-box membership, distributing them to all clusters simultaneously. This single-pass design avoids O(K × file_size) I/O where K is the cluster count.

2. The original-resolution PCD is rendered from `views_count` evenly spaced azimuth angles at 35° elevation using Open3D's `OffscreenRenderer`, producing 640×480 PNG files against a light-blue background.

3. All images are base64-encoded and submitted in a single chat turn to the configured Ollama model. The prompt requests exactly: `{"label": "<word>", "description": "<30-word max>"}` and provides domain-specific guidelines.

4. The response is parsed by a regex-based JSON extractor. The label is normalized and mapped to a classification code by `_map_ai_label_to_classification`.

5. Up to 3 retries with 2-second delays handle network or parsing failures.

[FIGURE PLACEHOLDER — Figure 15: Example VLM input images (4 views) and output label for a car cluster]

*Show 4 isometric views of a car-like cluster with the VLM response JSON overlaid.*

## F. LAS I/O and Chunk Management

The LAS layer (`modules/models/las_tools/`) isolates all file format concerns from the rest of the system. The design separates reading into two iterator strategies:

**`LasIteratorBasic`**: Reads the LAS file sequentially in `chunk_size`-point batches using `laspy.open().chunk_iterator()`. Used in the VLM Clustering stage where the whole file must be merged.

**`LasIteratorKDTree`**: Reads logical chunks defined by `1_pre_processing_index.json`. For each logical chunk (spatial bin), it performs a streaming scan of the LAS file, extracting only the global index ranges that overlap the bin's `[start, end)` range. This assembles spatially coherent tiles without random-access I/O.

All non-XY Z dimensions (color channels, intensity, GPS time, scan angle, return number, etc.) are preserved as NumPy arrays in a `points_attributes` dictionary and carried through every processing step. The output LAS faithfully reproduces the original data with two additions: the `classification` dimension (updated in-place) and the `Clusters` extra bytes dimension (int32, appended to the header as an Extra Bytes record).

## G. Key Dependencies

| Library | Version | Role |
|---|---|---|
| Open3D | 0.19.0 | Point cloud algorithms (RANSAC, DBSCAN, KD-tree), OffscreenRenderer, GUI |
| laspy | 2.6.1 | LAS/LAZ file I/O |
| NumPy | 2.4.x | Array computation |
| SciPy | 1.17.x | ConvexHull, Delaunay, KDTree for mesh reconstruction |
| ollama | 0.6.1 | VLM HTTP client (Ollama runtime API) |
| cjio | 0.10.1 | CityJSON JSON encoding (FloatEncoder for precise coordinates) |
| PyYAML | 6.0.x | Configuration file parsing |
| loguru | 0.7.x | Structured logging |

The environment is managed as a Conda environment (`environment.yml`) pinned to Python 3.11.6 for reproducibility.
