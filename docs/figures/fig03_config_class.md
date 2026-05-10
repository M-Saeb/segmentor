# Figure 3 — SegmentorConfig Singleton Class Diagram

**Referenced in**: Section III-D  
**Type**: UML class diagram

## Recommended IEEE Format

**Diagram type**: Simplified UML class box (TikZ or tabular)

Draw a single three-section box: top section = class name `SegmentorConfig` with `«singleton»` stereotype; middle section = key attributes (8–10, trimmed to the most architecturally significant); bottom section = public interface methods (`initialize`, `get_instance`, `validate_config_file`). Add a note balloon to the right explaining the `__new__` guard behavior.

**LaTeX approach**: TikZ with three `\node[draw, text width=5cm]` stacked vertically and a `\node[draw, callout rectangle]` note. Keep the attribute list to ≤10 lines to fit in one column. Alternatively use `\begin{tabular}{|l|}` with `\hline` separators — cleaner for IEEE reviewers who prefer plain tables.

---

```mermaid
classDiagram
    class SegmentorConfig {
        <<singleton>>
        +str input_file
        +bool reset_classification
        +int chunk_size
        +dict downsample
        +dict filter_noise
        +dict previewer
        +dict ground_segmentation
        +dict ceiling_segmentation
        +dict wall_segmentation
        +dict objects_segmentation
        +dict vlm
        +dict cityjson_export
        +dict result
        +bool collect_runtime_report
        ---
        -_instance : SegmentorConfig
        -_config_path : str
        ---
        +initialize(config_file)$ SegmentorConfig
        +get_instance()$ SegmentorConfig
        +validate_config_file(data)
        -__new__(config_file)
        -__init__(config_file)
    }

    note for SegmentorConfig "First call requires config_file.\nSubsequent calls return the same instance.\nRaises RuntimeError if re-initialized\nwith a different path."
```
