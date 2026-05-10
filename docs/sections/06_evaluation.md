# V. Evaluation

[NOTE: This section requires data from the author. The structure below defines what should be measured; populate with actual numbers once experiments are run.]

## A. Dataset

[TODO: Describe the dataset(s) used:
- Scanner hardware (manufacturer, model, wavelength)
- Scan type: indoor / outdoor / both
- Number of scans
- Point counts (min, max, mean)
- Whether ground truth labels are available (manual annotation? how?)
- File sizes]

Example placeholder text:
> Experiments were conducted on [N] LiDAR scans acquired using a [TODO: scanner model] at [TODO: location type, e.g., a multi-story university building and an outdoor parking facility]. The dataset comprises [TODO: X] million points across [TODO: N] scenes. Ground truth semantic labels were manually annotated by [TODO: N] experts and validated against visual inspection of the original RGB imagery.

## B. Evaluation Metrics

We evaluate the system along four dimensions:

**Segmentation accuracy** is measured using standard per-class metrics computed against ground truth labels: precision, recall, and F1-score for each class (ground, wall, ceiling, objects). Intersection-over-Union (IoU) is also reported for comparison with learning-based methods.

**Runtime** is measured as wall-clock time for each of the four stages on a representative scan. Timing is broken down into sub-operations within each stage.

**VLM accuracy** is measured as the fraction of clusters correctly labelled by the VLM relative to ground truth object annotations. We report per-category accuracy (vehicle vs. non-vehicle) separately.

**Memory footprint** is measured as peak resident memory during each stage. This is particularly relevant for the VLM Clustering stage's merge step, which loads the entire classified cloud into RAM.

## C. Runtime Results

[FIGURE PLACEHOLDER — Figure 16: Runtime breakdown bar chart]

*Grouped bar chart with four groups (Pre-processing, Segmentation, VLM Clustering, Exporting), each showing sub-operation timing. X-axis: time in seconds. Annotate with dataset size (millions of points).*

| Stage | Operation | Time (s) | Notes |
|---|---|---|---|
| Pre-processing | Noise removal | [TODO] | `nb_neighbors=20, std_ratio=2.0` |
| Pre-processing | Voxel downsample | [TODO] | `voxel_size=0.2 m` |
| Pre-processing | KD-tree ordering | [TODO] | `target=25,000 pts/tile` |
| Segmentation | Ground RANSAC | [TODO] | 10 outer iterations |
| Segmentation | Ceiling RANSAC | [TODO] | Up to 100 iterations |
| Segmentation | Wall RANSAC | [TODO] | Up to 100 iterations |
| VLM Clustering | Merge chunks | [TODO] | Full file in memory |
| VLM Clustering | DBSCAN | [TODO] | `eps=0.45, min_points=15` |
| VLM Clustering | Original resolution restore | [TODO] | Single-pass scan per cluster batch |
| VLM Clustering | Image generation | [TODO] | 4 views per cluster |
| VLM Clustering | VLM inference | [TODO] | Per cluster, Gemma3:4b |
| Exporting | Read full LAS | [TODO] | 3_post_process.las |
| Exporting | Ground mesh (2 passes) | [TODO] | ball_radii=[1,3] |
| Exporting | Ceiling mesh (2 passes) | [TODO] | ball_radii=[1,3] |
| Exporting | Wall DBSCAN + meshes | [TODO] | ε=1 m, 1 pass per cluster |
| Exporting | Car meshes | [TODO] | 1 pass per cluster |

## D. Segmentation Accuracy (Segmentation Stage)

[FIGURE PLACEHOLDER — Figure 17: Confusion matrix for class predictions]

*5×5 confusion matrix: rows = predicted, columns = true (ground, wall, ceiling, objects, unclassified). Color scale by count or fraction.*

| Class | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Ground | [TODO] | [TODO] | [TODO] | [TODO] |
| Wall | [TODO] | [TODO] | [TODO] | [TODO] |
| Ceiling | [TODO] | [TODO] | [TODO] | [TODO] |
| Objects | [TODO] | [TODO] | [TODO] | [TODO] |
| **Mean** | [TODO] | [TODO] | [TODO] | [TODO] |

## E. VLM Object Identification Accuracy (VLM Clustering Stage)

[FIGURE PLACEHOLDER — Figure 18: VLM accuracy per category]

*Bar chart showing correct / incorrect / ambiguous for each object category (car, chair, desk, etc.)*

| Category | Correct | Incorrect | Ambiguous | Accuracy (%) |
|---|---|---|---|---|
| Vehicle (car/van) | [TODO] | [TODO] | [TODO] | [TODO] |
| Furniture | [TODO] | [TODO] | [TODO] | [TODO] |
| Structure/pillar | [TODO] | [TODO] | [TODO] | [TODO] |
| **Overall** | [TODO] | [TODO] | [TODO] | [TODO] |

## F. Mesh Reconstruction Quality (Exporting Stage)

[TODO: If available, report mesh quality metrics such as:
- Surface coverage (% of classified points within a threshold distance of the reconstructed mesh)
- Triangle count per surface type
- Visual comparison of single-pass vs. multi-pass reconstruction]

## G. Effect of Pre-processing on Segmentation Quality

[TODO: If feasible, include an ablation study comparing Segmentation results with and without the spatial reordering step, to demonstrate the importance of KD-tree chunking.]

## H. Comparison with Related Tools

[TODO: If feasible, include a comparison with at least one alternative tool (e.g., CloudCompare's RANSAC Shape Detection plugin) on the same dataset, reporting accuracy and runtime.]
