# Figure 8 — Exporting Stage Component Diagram

**Referenced in**: Section III-H  
**Type**: Component / data-flow diagram

## Recommended IEEE Format

**Diagram type**: Fan-out / fan-in block diagram (TikZ)

Draw a single input box (`3_post_process.las`) fanning out via four arrows to four parallel processing columns: **Ground**, **Ceiling**, **Walls**, **Cars**. Each column contains two or three stacked boxes (select → mesh → write). All four columns fan back in to a single output box (`citygml.city.json`). Below the diagram, include the iterative refinement schedule as a small companion `\begin{table}` (Pass / Radius / Min-neighbours). The parallel structure makes the independence of the four surface types visually obvious.

**LaTeX approach**: TikZ `\matrix` with one column per surface type; `\draw[->, thick]` fan-out from input and fan-in to output. Use `fill=gray!15` to shade each column's bounding box as a visual grouping. Span two columns (`figure*`) for readability.

---

```mermaid
flowchart TD
    SRC["📂 3_post_process.las"]
    READ["read_full_las_file()\nFull cloud into memory"]
    INIT_CJ["init_cityjson_file()\ngeographicalExtent from point cloud AABB"]
    OUT["📄 citygml.city.json\n(CityJSON v2)"]

    SRC --> READ --> INIT_CJ

    subgraph GROUND["GROUND  (classification == 2)"]
        G1["select_by_classification(2)"]
        G2["_pcd_to_mesh()  2 passes\nPass 1: radius=1 m  min_nbrs=10\nPass 2: radius=3 m  min_nbrs=7\n(between passes: sample 25% of mesh)"]
        G3["write_cityjson_mesh_part()\nmesh_id = 'groundSurface'\ntype = Building · MultiSurface · LoD2"]
        G1 --> G2 --> G3
    end

    subgraph CEILING["CEILING  (classification == 4)"]
        CE1["select_by_classification(4)"]
        CE2["_pcd_to_mesh()  2 passes\nPass 1: radius=1 m  min_nbrs=10\nPass 2: radius=3 m  min_nbrs=7"]
        CE3["write_cityjson_mesh_part()\nmesh_id = 'ceilingSurface'\ntype = Building · MultiSurface · LoD2"]
        CE1 --> CE2 --> CE3
    end

    subgraph WALLS["WALLS  (classification == 3)"]
        W1["select_by_classification(3)"]
        W2["cluster_dbscan()\nε=1 m  min_pts=50\n→ N independent wall faces"]
        W3["_pcd_to_mesh() per cluster  1 pass\nradius=1 m  min_nbrs=10\nfilter: mesh < 200 verts → discard"]
        W4["write_cityjson_mesh_part()\nmesh_id = 'wallSurface_N'\ntype = Building · MultiSurface · LoD2"]
        W1 --> W2 --> W3 --> W4
    end

    subgraph CARS["CARS  (classification == 7, per Clusters ID)"]
        CA1["select by Clusters > 0\n∩ classification == 7"]
        CA2["_pcd_to_mesh() per cluster  1 pass\nradius=1 m  min_nbrs=10\nfilter: mesh < 50 verts → discard"]
        CA3["write_cityjson_mesh_part()\nmesh_id = 'carSurface_N'\ntype = GenericCityObject · MultiSurface · LoD2"]
        CA1 --> CA2 --> CA3
    end

    READ --> GROUND & CEILING & WALLS & CARS
    G3 & CE3 & W4 & CA3 --> OUT
```

### Iterative mesh refinement schedule

| Pass | Ball radius (m) | Min neighbours | Input cloud |
|:---:|:---:|:---:|---|
| 1 | 1 | 10 | original classified PCD |
| 2 | 3 | 7 | 25% uniform sample of pass-1 mesh |
| 3 | 5 | 4 | 25% uniform sample of pass-2 mesh |
| 4 | 6 | 2 | 25% uniform sample of pass-3 mesh |
| 5 | 9 | 1 | 25% uniform sample of pass-4 mesh |

Ground and ceiling use 2 passes by default. Walls and cars use 1 pass.
