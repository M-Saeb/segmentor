# Figures Index

All figure drafts are in individual files below.  
Mermaid diagrams render in GitHub, VS Code with the Mermaid extension, and Obsidian.  
Figures marked ⚠️ require actual screenshots from a running pipeline before the paper is submitted.

| # | Title | Type | File | Status |
|:---:|---|---|---|:---:|
| 1 | High-level MVC Architecture | Mermaid `graph TB` | [fig01_mvc_architecture.md](fig01_mvc_architecture.md) | ✅ |
| 2 | End-to-End Data Flow | Mermaid `flowchart LR` | [fig02_data_flow.md](fig02_data_flow.md) | ✅ |
| 3 | SegmentorConfig Singleton | Mermaid `classDiagram` | [fig03_config_class.md](fig03_config_class.md) | ✅ |
| 4 | Pre-processing Stage Flow | Mermaid `flowchart TD` | [fig04_preprocessing_flow.md](fig04_preprocessing_flow.md) | ✅ |
| 5 | Segmentation Stage State Machine | Mermaid `stateDiagram-v2` | [fig05_segmentation_state_machine.md](fig05_segmentation_state_machine.md) | ✅ |
| 6 | Sequence: "Run Stage" → "Finalize" | Mermaid `sequenceDiagram` | [fig06_sequence_diagram.md](fig06_sequence_diagram.md) | ✅ |
| 7 | VLM Clustering Stage Pipeline | Mermaid `flowchart TD` | [fig07_vlm_clustering_pipeline.md](fig07_vlm_clustering_pipeline.md) | ✅ |
| 8 | Exporting Stage Component Diagram | Mermaid `flowchart TD` | [fig08_exporting_stage.md](fig08_exporting_stage.md) | ✅ |
| 9 | Classification Code Table | Markdown table | [fig09_classification_table.md](fig09_classification_table.md) | ✅ |
| 10 | RANSAC Floor Segmentation Example | Screenshot composite | [fig10_ransac_floor_example.md](fig10_ransac_floor_example.md) | ⚠️ |
| 11 | Full Segmentation Stage Output | Screenshot | [fig11_segmentation_output.md](fig11_segmentation_output.md) | ⚠️ |
| 12 | Ball-Neighbourhood Mesh Steps | Mermaid `flowchart LR` | [fig12_mesh_reconstruction_steps.md](fig12_mesh_reconstruction_steps.md) | ✅ |
| 13 | Multi-pass Mesh Refinement | Screenshot composite | [fig13_multipass_comparison.md](fig13_multipass_comparison.md) | ⚠️ |
| 14 | VLM Pipeline for a Single Cluster | Mermaid `flowchart LR` | [fig14_vlm_pipeline.md](fig14_vlm_pipeline.md) | ✅ |
| 15 | Example VLM Input / Output (car) | Screenshot composite | [fig15_vlm_example.md](fig15_vlm_example.md) | ⚠️ |
| 16 | Runtime Breakdown | Markdown table | [fig16_runtime_breakdown.md](fig16_runtime_breakdown.md) | ⚠️ fill numbers |
| 17 | Confusion Matrix (Segmentation) | Markdown table | [fig17_confusion_matrix.md](fig17_confusion_matrix.md) | ⚠️ fill numbers |
| 18 | VLM Object Identification Accuracy | Markdown table | [fig18_vlm_accuracy.md](fig18_vlm_accuracy.md) | ⚠️ fill numbers |

## Screenshot figures — how to produce them

| Figure | What to capture | When to run |
|:---:|---|---|
| 10 | Two side-by-side Open3D renders of the same chunk before and after ground RANSAC | After confirming the Ground stage in the Segmentation GUI |
| 11 | One Open3D render of a chunk colored by all four classes | After all three Segmentation sub-stages are confirmed |
| 13 | Two renders of the same wall/ground cluster at pass 1 (radius=1 m) vs pass 2 (radius=3 m) | Run `scripts/citygml_export.py` with `--debug-mesh` output |
| 15 | Four isometric PNGs of a car cluster + VLM exchange | Found in `<output>/vlm_object_traces/<cluster_id>/` after VLM Clustering |
