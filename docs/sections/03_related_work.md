# II. Related Work

## A. Point Cloud Segmentation

Point cloud segmentation is the task of partitioning a set of 3D points into semantically meaningful groups. Methods broadly fall into three categories: region-growing, model-fitting, and learning-based approaches.

**Model-fitting methods** such as RANSAC (Random Sample Consensus) [TODO: cite Fischler & Bolles 1981] detect geometric primitives (planes, spheres, cylinders) by iteratively sampling minimal point subsets, fitting a model, and counting inliers within a distance threshold. RANSAC is well-suited to indoor and urban scenes where dominant planar structures (floors, walls, ceilings, roads) are present and can be isolated reliably regardless of point density variation. The Cloth Simulation Filter (CSF) [TODO: cite Zhang et al. 2016] models the ground surface as a cloth draped under gravity over an inverted terrain DEM; it is widely used in vegetation-heavy outdoor scenes. Segmentor supports both methods interchangeably for ground segmentation.

**Density-based clustering** methods such as DBSCAN [TODO: cite Ester et al. 1996] group points by local density without requiring a prior cluster count. After planar structures are removed, DBSCAN cleanly isolates discrete objects. This two-stage strategy (plane removal → clustering) is a standard pattern in scene-understanding pipelines [TODO: cite relevant work].

**Learning-based methods** — PointNet [TODO: cite Qi et al. 2017], PointNet++ [TODO: cite], and transformer-based successors — learn per-point features directly from raw coordinates. While these achieve state-of-the-art accuracy on benchmark datasets, they require annotated training data and GPU inference that may not be available in a practitioner's field setup. Segmentor takes a complementary approach: geometric algorithms handle planar structure detection, and a VLM handles free-form object recognition using a model already pretrained on image data.

## B. Interactive Point Cloud Tools

Commercial and open-source tools such as CloudCompare [TODO: cite], LAStools [TODO: cite], and PDAL [TODO: cite] expose comprehensive processing libraries but require the user to script or manually chain operations. The Open3D library [TODO: cite Zhou et al. 2018] provides both algorithmic primitives (RANSAC plane fitting, DBSCAN, KD-tree) and a Python-bindable GUI framework, which Segmentor uses as its rendering and interaction foundation. To the best of our knowledge, no existing tool integrates the combination of interactive stage-by-stage preview, AI-assisted residual classification, and CityJSON export in a single, user-guided pipeline.

## C. Vision Language Models for 3D Understanding

Vision Language Models (VLMs) — including CLIP [TODO: cite], LLaVA [TODO: cite], Gemma [TODO: cite], and Qwen-VL [TODO: cite] — extend large language model capabilities to image understanding. Their application to 3D point cloud scenes is indirect: points must first be projected or rendered into 2D images before the model can reason about them. Multi-view rendering strategies [TODO: cite relevant work] capture multiple perspectives of a 3D object to compensate for the information loss of any single projection. Segmentor adopts this strategy, rendering four (configurable) isometric views per cluster before sending them jointly to a locally deployed VLM via the Ollama inference runtime.

## D. CityJSON and 3D City Modelling

CityJSON [TODO: cite Ledoux et al. 2019] is a JSON encoding of the CityGML conceptual model standardised by the Open Geospatial Consortium (OGC). It represents urban objects — buildings, roads, vegetation, city furniture — at multiple Levels of Detail (LoD). Automatic conversion of LiDAR scans to CityJSON/CityGML has been studied [TODO: cite relevant works] but typically requires specialized reconstruction pipelines. Segmentor provides a lightweight, geometry-driven path from classified point cloud layers directly to LoD2 CityJSON geometries, without requiring a full reconstruction pipeline.

[TODO: The author should review and expand the references above, replace all [TODO: cite ...] with real BibTeX keys, and confirm whether there are additional related tools or papers to position against.]
