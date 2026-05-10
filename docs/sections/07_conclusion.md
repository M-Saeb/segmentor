# VI. Conclusion

We presented Segmentor, an interactive four-stage system for semantic segmentation and 3D model export of LiDAR point clouds. The four stages — Pre-processing, Segmentation, VLM Clustering, and Exporting — form a complete and reproducible pipeline from raw scan to interoperable CityJSON model. Each stage has a clearly scoped task: Pre-processing produces geometrically coherent processing tiles; the Segmentation stage identifies structural surfaces (ground, walls, ceiling) through RANSAC-based plane detection with interactive user guidance; the VLM Clustering stage isolates and semantically labels discrete objects through density clustering and multi-view AI inference; and the Exporting stage converts the classified cloud into triangulated CityJSON geometry via iterative ball-neighbourhood mesh reconstruction.

The system's MVC architecture enables both interactive GUI-driven workflows and headless scripted execution, supporting both exploratory use and reproducible batch processing. The spatial KD-tree indexing strategy in the Pre-processing stage is a key enabler of segmentation correctness: by ensuring that each chunk contains geometrically local points, it prevents RANSAC from fitting planes across disconnected scene regions.

[TODO: Add a paragraph summarizing the quantitative results from Section V once experiments are complete.]

### Limitations

Several limitations remain for future work:

**VLM dependency on rendering quality.** The quality of VLM-based classification depends on the visual fidelity of the isometric renders. Objects with sparse point clouds or unusual aspect ratios may produce uninformative images, leading the model to default to "empty" or "unknown" labels.

**Memory requirement for the VLM Clustering stage.** The merge step loads the entire classified point cloud into RAM before running DBSCAN. For very large datasets (>200M points), this may exceed available memory. A tile-based DBSCAN variant that stitches cluster boundaries would be needed.

**Single global RANSAC configuration per stage.** The Segmentation stage applies the same RANSAC parameters to all spatial tiles. For scenes with multiple floor levels or walls at varying orientations, adaptive per-tile parameter selection would improve recall.

**No learning-based segmentation comparison.** The current implementation does not include a neural-network-based alternative as a baseline, which limits quantitative comparison to geometry-only methods. Integrating a pretrained segmentation model as an optional Segmentation stage would strengthen the evaluation.

**Wall face separation depends on DBSCAN in Exporting stage.** The quality of individual wall mesh reconstruction in the Exporting stage depends on DBSCAN separating distinct wall surfaces correctly. In dense or cluttered scenes, adjacent walls with overlapping bounding regions may be merged into a single cluster.

### Future Work

Future directions include: (a) a streaming DBSCAN implementation for the VLM Clustering stage to remove the full-merge memory requirement; (b) extending the Exporting stage to produce LoD1 building footprints derived from ground surface boundaries; (c) replacing the single VLM call with a model ensemble for improved label confidence estimation; (d) a web-based front-end for remote operation; and (e) integration with BIM (Building Information Modelling) formats such as IFC for direct use in architectural workflows.

### Acknowledgements

[TODO: Add funding sources, scanner hardware providers, dataset contributors, and any other acknowledgements.]
