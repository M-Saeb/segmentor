# Figure Placeholders

This file catalogs all figures referenced in the paper sections with descriptions of what each should show. Replace each entry with the actual image when available.

---

## Figure 1 — High-level MVC Architecture Diagram
**Referenced in**: Section III-B  
**Type**: Block/layer diagram  
**Description**: Three vertical lanes labeled **Model**, **Controller**, **Viewer** (left to right). Inside each lane, four horizontal groupings for the four stages: Pre-processing (top), Segmentation, VLM Clustering, Exporting (bottom). Note that the Exporting stage has no Viewer lane (headless only). Arrows cross between lanes to show: Viewer → Controller (user events), Controller → Model (algorithm calls), Model → Controller (results), Controller → Viewer (display updates). Include the file path for each major class.

---

## Figure 2 — End-to-End Data Flow Diagram
**Referenced in**: Section III-C  
**Type**: Pipeline / data-flow diagram  
**Description**: Horizontal left-to-right flow with four stage boxes. Each box is labeled with the stage name. Arrows between boxes are annotated with the file names produced. Annotate approximate file sizes or point counts for a typical dataset.

```
[Raw LAS]
    ──▶  Pre-processing  ──▶  1_pre_processing.las + 1_pre_processing_index.json
    ──▶  Segmentation    ──▶  2_pipeline.las + 2_pipeline.yml
    ──▶  VLM Clustering  ──▶  3_post_process.las + 3_objects.yml + vlm_object_traces/
    ──▶  Exporting       ──▶  citygml.city.json
```

---

## Figure 3 — SegmentorConfig Singleton Class Diagram
**Referenced in**: Section III-D  
**Type**: UML class diagram  
**Description**: Single `SegmentorConfig` class box. Show key attributes with types. Show `_instance` class attribute and the `__new__` / `initialize` / `get_instance` methods to illustrate the singleton pattern. Add a note: "First construction requires config_file; subsequent accesses return same instance."

---

## Figure 4 — Pre-processing Stage Flow Diagram
**Referenced in**: Section III-E  
**Type**: Flowchart  
**Description**: Sequential flowchart: Read raw LAS (chunk by chunk) → Merge all chunks → Statistical outlier removal → Voxel downsampling (with KD-tree attribute transfer) → 3D KD-tree seed grid → Assign each point to nearest seed → Sort by (bin_id, Z) → Write reordered LAS → Write index.json. Annotate each step with the relevant function name and parameter names.

---

## Figure 5 — Segmentation Stage State Machine Diagram
**Referenced in**: Section III-F  
**Type**: State machine diagram  
**Description**: States: Idle → Stage 1 Active (snapshot saved) → Run preview → Update display → (repeat) → Confirm (enabled=True) → Stage 2 Active → … → All stages done → Finalize → Full processing (all chunks) → Done. Show the "Skip" transition (enabled=False, advance without result) and the snapshot restore mechanism on repeated "Run" clicks.

---

## Figure 6 — Sequence Diagram: "Run Stage" to "Finalize"
**Referenced in**: Section III-F.3  
**Type**: UML sequence diagram  
**Description**: Actors: User, PipelineViewer, PipelineController, LasPipelineWrapper, pcd_tools. Show the sequence for one complete preview cycle: User clicks "Run Stage" → show_loading() → background thread starts → restore_snapshot() → RANSAC algorithm → set_current_chunk() → post_to_main_thread → update_preview_pcd() → hide_loading(). Then show the "Finalize" path: build_process_pipeline_from_setup() → iter over all chunks → write chunks → done.

---

## Figure 7 — VLM Clustering Stage Pipeline Diagram
**Referenced in**: Section III-G  
**Type**: Flowchart  
**Description**: 2_pipeline.las → merge_chunks (all into RAM) → DBSCAN on INIT points only (eps, min_points) → For each cluster ≥ 50 pts: compute bbox → restore original resolution PCD (single-pass original LAS scan) → render 4 isometric views → send to Ollama VLM → parse JSON response → assign classification code → write 3_post_process.las + 3_objects.yml.

---

## Figure 8 — Exporting Stage Component Diagram
**Referenced in**: Section III-H  
**Type**: Component / data-flow diagram  
**Description**: 3_post_process.las → read_full_las_file → split by class into four paths:
- GROUND points → mesh (2 passes) → write_cityjson_mesh_part("groundSurface", Building)
- CEILING points → mesh (2 passes) → write_cityjson_mesh_part("ceilingSurface", Building)
- WALL points → DBSCAN clusters → per-cluster mesh (1 pass) → write_cityjson_mesh_part("wallSurface_N", Building) [filter < 200 verts]
- CAR cluster points → per-cluster mesh (1 pass) → write_cityjson_mesh_part("carSurface_N", GenericCityObject) [filter < 50 verts]
All paths write incrementally to citygml.city.json. Show the iterative radii [1, 3, 5, 6, 9] as an annotation on the mesh pass box.

---

## Figure 9 — Classification Code Table with Color Legend
**Referenced in**: Section III-I  
**Type**: Table with colored squares  
**Description**: 8-row table: code number, color swatch, constant name, semantic label, stage assigned. Use standard viewer colors: INIT=gray, GROUND=green, WALL=blue, CEILING=red/orange, STRUCTURE=brown, OBJECTS=yellow (transient), CAR=cyan, OTHER_OBJECTS=purple.

---

## Figure 10 — RANSAC Floor Segmentation Example
**Referenced in**: Section IV-A.2  
**Type**: Side-by-side point cloud renders  
**Description**: Left: full raw point cloud colored by RGB. Right: same cloud after ground RANSAC — ground points in green, all others in gray. Annotate with the number of floor planes detected and the parameter values used.

---

## Figure 11 — Wall and Ceiling Segmentation Example
**Referenced in**: Section IV-C  
**Type**: Point cloud render  
**Description**: Single render showing the fully segmented output after the Segmentation stage. Color scheme: ground=green, wall=blue, ceiling=red, unclassified INIT=gray. Include a small legend.

---

## Figure 12 — Ball-Neighbourhood Mesh Reconstruction Steps
**Referenced in**: Section IV-D  
**Type**: Three-panel figure  
**Description**:
- Panel (a): Raw wall or ground point cloud (gray points)
- Panel (b): One seed point with ball drawn around it; neighbourhood points highlighted; local PCA plane shown as a translucent disk; convex hull boundary highlighted in orange
- Panel (c): Final reconstructed mesh with triangle wireframe visible

---

## Figure 13 — Multi-pass Mesh Refinement Comparison
**Referenced in**: Section IV-D.1  
**Type**: Side-by-side renders  
**Description**: Left: result after pass 1 only (ball_radius=1, tight, possibly sparse). Right: result after pass 2 (ball_radius=3, gap-filling). Use the same wall cluster for both panels. Annotate with triangle counts.

---

## Figure 14 — VLM Pipeline Diagram for a Single Cluster
**Referenced in**: Section IV-E  
**Type**: Pipeline diagram  
**Description**: Cluster point cloud → OffscreenRenderer (show 4 camera positions at 35° elevation, evenly spaced in azimuth) → 4 PNG files (show small thumbnails) → base64 encode → HTTP POST to Ollama (show request structure: model, messages with images) → JSON response (show example: `{"label": "car", "description": "..."}`) → classification code 7.

---

## Figure 15 — Example VLM Input and Output for a Car Cluster
**Referenced in**: Section IV-E  
**Type**: Photo/screenshot composite  
**Description**: 4 side-by-side isometric PNG renders of a car-like cluster (different azimuths). Below: the VLM prompt (abbreviated) and the model's raw JSON response. Arrow to the right showing final classification code 7 (CAR).

---

## Figure 16 — Runtime Breakdown Bar Chart
**Referenced in**: Section V-C  
**Type**: Grouped bar chart  
**Description**: Four groups (Pre-processing, Segmentation, VLM Clustering, Exporting). Each group has stacked or adjacent bars for each sub-operation. Y-axis: time in seconds. Annotated with dataset size and hardware specs (CPU model, RAM).

---

## Figure 17 — Confusion Matrix for Class Predictions (Segmentation Stage)
**Referenced in**: Section V-D  
**Type**: Heatmap / confusion matrix  
**Description**: 5×5 matrix (ground, wall, ceiling, objects, unclassified). Rows = predicted, columns = true. Color scale from white (0%) to dark blue (100%). Per-class precision and recall annotated on margins.

---

## Figure 18 — VLM Accuracy per Object Category (VLM Clustering Stage)
**Referenced in**: Section V-E  
**Type**: Grouped bar chart  
**Description**: X-axis: object categories (vehicle, chair, desk, bin, pillar, other). Each category has three bars: correct (green), incorrect (red), ambiguous/empty (gray). Y-axis: percentage.
