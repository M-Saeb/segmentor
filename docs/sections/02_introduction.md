# I. Introduction

Three-dimensional LiDAR scanning has become a routine data-acquisition technique across architecture, civil engineering, autonomous driving, and urban planning. A single survey session can produce tens or hundreds of millions of points stored in the widely adopted LAS/LAZ format. Converting this raw geometry into semantically labelled models — distinguishing ground surfaces from walls, ceilings, and discrete objects — remains a laborious, parameter-sensitive task that typically requires domain expertise, custom scripting, and repeated trial-and-error.

Existing open-source point cloud processing tools [TODO: cite CloudCompare, PDAL, LAStools] provide a broad set of algorithms but generally lack (a) a guided, stage-by-stage interactive workflow that lets a non-expert user preview and tune each algorithm before committing to a full run, (b) automatic semantic labelling of residual objects using modern AI, and (c) a seamless path from raw scan to an interoperable 3D city model.

This paper presents **Segmentor**, a Python-based system that fills this gap. The processing pipeline is organized into four clearly named stages that reflect the task each performs:

1. **Pre-processing**: Noise removal, voxel downsampling, and spatial reordering of the raw point cloud into geometrically coherent processing tiles via KD-tree grid binning.

2. **Segmentation Stage**: An interactive GUI-driven workflow where the user tunes RANSAC-based plane detection for the three dominant structural surfaces — ground, ceiling, and walls — previewing results on a representative subset before applying confirmed parameters to the entire dataset.

3. **VLM Clustering Stage**: DBSCAN clustering of the remaining unclassified points (those not identified as structural surfaces), followed by automated semantic object identification using a locally hosted Vision Language Model (VLM) that reasons over multi-view isometric renders of each cluster.

4. **Exporting Stage**: Conversion of the fully classified point cloud into a CityJSON v2 file. Each surface class (ground, ceiling, walls, vehicles) is independently reconstructed into a triangle mesh using an iterative ball-neighbourhood growing algorithm, then serialized as a CityObject with LoD2 geometry.

This stage naming was deliberately chosen to reflect what the system is actually doing: the segmentation stage identifies structural geometry; the VLM clustering stage finds and names discrete objects; the exporting stage produces the deliverable artefact. Together, the four stages provide a complete, reproducible pipeline from raw scan to interoperable 3D model.

The remainder of this paper is structured as follows. Section II surveys related work. Section III describes the overall system architecture. Section IV details the algorithms and implementation. Section V presents evaluation results. Section VI concludes.

[TODO: Add 1–2 sentences characterizing the primary use case (indoor/outdoor, specific building type, urban scale, etc.) after confirming with the author.]
