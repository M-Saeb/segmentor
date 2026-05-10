# Figure 14 — VLM Pipeline Diagram for a Single Cluster

**Referenced in**: Section IV-E  
**Type**: Pipeline / flowchart

## Recommended IEEE Format

**Diagram type**: Subsystem block diagram with labeled data artifacts (TikZ)

Group boxes into three visually distinct subsystems: **OffscreenRenderer** (four view boxes in a 2×2 grid), **Ollama API** (a single box showing the POST endpoint and model name), and **Label Mapper** (a small decision table). Connect the subsystems left-to-right with labeled arrows for the data artifacts that flow between them: `4 × PNG` (Renderer → base64 encoder), `base64[]` (encoder → Ollama), `JSON {label, description}` (Ollama → Mapper), `classification code` (Mapper → output). Draw subsystem boundaries as dashed rectangles with a bold italic label.

**LaTeX approach**: TikZ with `\node[draw, dashed, inner sep=6pt, fit=(...)]` for subsystem bounding boxes and `\node[font=\bfseries\itshape, above]` for subsystem labels. Data artifact labels on arrows use `\node[font=\scriptsize\ttfamily, above, midway]`. Fits in one column (landscape-oriented layout, ~8 cm tall).

---

```mermaid
flowchart LR
    PCD["Cluster PCD\n(original resolution\nfrom raw LAS scan)"]

    subgraph RENDER["OffscreenRenderer  640×480\nlight-blue background"]
        R1["View 1\naz = 225°\nelev = 35°"]
        R2["View 2\naz = 315°\nelev = 35°"]
        R3["View 3\naz =  45°\nelev = 35°"]
        R4["View 4\naz = 135°\nelev = 35°"]
    end

    B64["base64-encode\n4 PNG files"]

    subgraph OLLAMA["Ollama API  (localhost:11434)"]
        REQ["POST /api/chat\nmodel: gemma3:4b\nmessages:\n  role: user\n  content: structured prompt\n  images: [b64_1, b64_2, b64_3, b64_4]"]
        RESP["Response content:\n'{\"label\": \"car\",\n  \"description\": \"a silver sedan …\"}'"]
        REQ --> RESP
    end

    subgraph MAP["_map_ai_label_to_classification()"]
        direction TB
        M1["'car' / 'van' / 'truck' → 7  CAR"]
        M2["'empty' / 'wall' / 'ground' → 5  STRUCTURE"]
        M3["anything else → 8  OTHER_OBJECTS"]
    end

    PCD --> RENDER
    RENDER --> R1 & R2 & R3 & R4
    R1 & R2 & R3 & R4 --> B64
    B64 --> OLLAMA
    RESP --> MAP
```
