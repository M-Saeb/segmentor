# Figure 15 — Example VLM Input and Output for a Car Cluster

**Referenced in**: Section IV-E  
**Type**: Screenshot composite  
**Status**: ⚠️ Four isometric PNG renders must be taken from actual data (found in `vlm_object_traces/` after a VLM Clustering run).

## Recommended IEEE Format

**Diagram type**: 1×4 image strip + caption listing (mixed figure)

Arrange the four isometric view renders in a single row using four `\subfloat` panels, each labeled with its azimuth angle (225°, 315°, 45°, 135°). Below the image strip, include the VLM JSON response and classification mapping as a compact `\begin{lstlisting}[language=json]` block inside the figure environment (or in the caption). This keeps all the evidence — visual input to the VLM and its structured output — in a single self-contained figure.

**LaTeX approach**: `\usepackage{subfig, listings}`. Four `\subfloat` with `width=0.23\linewidth` each, no `\hfill` gaps (tight strip). The `lstlisting` block below uses `basicstyle=\scriptsize\ttfamily, frame=single`. Use `\begin{figure*}` to span both columns — four images plus code block are too wide for a single column.

## Layout

```
┌───────────┬───────────┬───────────┬───────────┐
│           │           │           │           │
│  View 1   │  View 2   │  View 3   │  View 4   │
│  az=225°  │  az=315°  │  az= 45°  │  az=135°  │
│           │           │           │           │
│  01.png   │  02.png   │  03.png   │  04.png   │
└───────────┴───────────┴───────────┴───────────┘
                         │
                         ▼
              Ollama API (gemma3:4b)
                         │
                         ▼
```

**VLM prompt sent (abbreviated):**

```
You see several isometric views around a LiDAR point cloud region
(same scene from different angles) with a light blue background.
The scene is for a parking lot area with only the object we want
to identify, with the photos rotating around the object.
You must reply with a single JSON object ONLY, in this exact form:
{"label": "<single word object name>", "description": "<30 words max>"}
```

**Example VLM response:**

```json
{
  "label": "car",
  "description": "a silver sedan parked facing north, clearly visible roof and hood shape"
}
```

**Classification mapping applied:**

| VLM `label` | Normalized | Code | Constant |
|---|---|:---:|---|
| `"car"` | `car` | **7** | `PCB_CLASSIFICATION_CAR` |
| `"van"` | `van` | **7** | `PCB_CLASSIFICATION_CAR` |
| `"chair"` | `chair` | **8** | `PCB_CLASSIFICATION_OTHER_OBJECTS` |
| `"empty"` | `empty` | **5** | `PCB_CLASSIFICATION_STRUCTURE` |
| `"unknown"` | `unknown` | **5** | `PCB_CLASSIFICATION_STRUCTURE` |

## How to produce the screenshots

After running the VLM Clustering stage, the renders are already saved to:

```
<output_folder>/vlm_object_traces/<cluster_index>/01.png
<output_folder>/vlm_object_traces/<cluster_index>/02.png
<output_folder>/vlm_object_traces/<cluster_index>/03.png
<output_folder>/vlm_object_traces/<cluster_index>/04.png
```

Pick a cluster labelled `"car"` from `3_objects.yml`, then composite its four PNGs side-by-side.
