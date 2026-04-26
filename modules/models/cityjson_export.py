"""
CityJSON export for pipeline results (OGC CityJSON JSON encoding).

This is **not** CityGML XML; filename ``citygml.city.json`` is historical. Content
follows CityJSON v2.x via ``cjio``, aligned with CityGML semantics. Native
CityGML 3 XML export would be a separate toolchain.

Ground, wall, and ceiling points are exported as separate ``BuildingPart`` city
objects under one parent ``Building``, each with its own LOD1 bbox shell —
similar in intent to thematic GroundSurface / WallSurface / Ceiling (roof)
surfaces in CityGML.
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING

import numpy as np
from cjio.cityjson import CityJSON
from cjio.floatEncoder import FloatEncoder

from modules.models.post_processing_core import merge_chunks
from modules.utils import constants as C
from modules.utils.log_utils import get_dual_logger

import open3d as o3d

if TYPE_CHECKING:
    from modules.utils import config as config_module

log = get_dual_logger(__name__)

# Match CityJSON / cjio JSON float formatting (see cjio ``save`` command).
json.encoder.c_make_encoder = None
json.encoder.float = FloatEncoder

BUILDING_STRUCTURE_CLASSIFICATIONS: tuple[int, ...] = (
    C.PCB_CLASSIFICATION_GROUND,
    C.PCB_CLASSIFICATION_WALL,
    C.PCB_CLASSIFICATION_CEILING,
)

BUILDING_CITYOBJECT_ID = "segmentor-building-main"

PART_ID_BY_CLASS: dict[int, str] = {
    C.PCB_CLASSIFICATION_GROUND: "segmentor-building-ground",
    C.PCB_CLASSIFICATION_WALL: "segmentor-building-walls",
    C.PCB_CLASSIFICATION_CEILING: "segmentor-building-ceiling",
}

PART_CLASS_ORDER: tuple[int, ...] = (
    C.PCB_CLASSIFICATION_GROUND,
    C.PCB_CLASSIFICATION_WALL,
    C.PCB_CLASSIFICATION_CEILING,
)

_EPS_PAD = 1e-3


def _pad_bbox(xyz_min: np.ndarray, xyz_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz_min = np.asarray(xyz_min, dtype=np.float64).copy()
    xyz_max = np.asarray(xyz_max, dtype=np.float64).copy()
    for i in range(3):
        if math.isclose(xyz_min[i], xyz_max[i], abs_tol=1e-9):
            xyz_min[i] -= _EPS_PAD
            xyz_max[i] += _EPS_PAD
        else:
            xyz_min[i] -= _EPS_PAD
            xyz_max[i] += _EPS_PAD
    return xyz_min, xyz_max


def _axis_aligned_bbox_faces(
    xyz_min: np.ndarray, xyz_max: np.ndarray
) -> tuple[list[list[float]], list[list[list[int]]]]:
    """Return 8 corner vertices and MultiSurface boundaries (local vertex indices)."""
    x0, y0, z0 = (float(xyz_min[0]), float(xyz_min[1]), float(xyz_min[2]))
    x1, y1, z1 = (float(xyz_max[0]), float(xyz_max[1]), float(xyz_max[2]))

    verts: list[list[float]] = [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]
    boundaries: list[list[list[int]]] = [
        [[0, 1, 2, 3]],
        [[4, 7, 6, 5]],
        [[0, 1, 5, 4]],
        [[1, 2, 6, 5]],
        [[2, 3, 7, 6]],
        [[3, 0, 4, 7]],
    ]
    return verts, boundaries


def _append_building_part_shoebox(
    *,
    vertices: list[list[float]],
    vertex_offset: int,
    xyz_part: np.ndarray,
) -> tuple[list[list[list[int]]] | None, int]:
    """
    Append 8 bbox vertices for ``xyz_part`` and return MultiSurface boundaries
    (global indices) plus new vertex count. Returns (None, 0) if no points.
    """
    if xyz_part.size == 0 or len(xyz_part) == 0:
        return None, 0

    xyz_min = xyz_part.min(axis=0)
    xyz_max = xyz_part.max(axis=0)
    xyz_min, xyz_max = _pad_bbox(xyz_min, xyz_max)

    local_verts, local_boundaries = _axis_aligned_bbox_faces(xyz_min, xyz_max)
    vertices.extend(local_verts)

    global_boundaries: list[list[list[int]]] = []
    for face in local_boundaries:
        ring = face[0]
        global_boundaries.append([[vertex_offset + i for i in ring]])

    return global_boundaries, len(local_verts)


def build_cityjson_for_building_shell(
    xyz: np.ndarray,
    classification: np.ndarray,
) -> CityJSON:
    """
    Build CityJSON with one ``Building`` and up to three ``BuildingPart`` children
    (ground, walls, ceiling), each with LOD1 MultiSurface bbox geometry from
    points of that classification.
    """
    classification = np.asarray(classification, dtype=np.int64)

    vertices: list[list[float]] = []
    city_objects: dict[str, dict] = {}
    children: list[str] = []
    offset = 0

    for cls in PART_CLASS_ORDER:
        part_id = PART_ID_BY_CLASS[cls]
        mask = classification == cls
        xyz_part = np.asarray(xyz[mask], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_part)
        o3d.visualization.draw_geometries([pcd])

        boundaries, n_v = _append_building_part_shoebox(
            vertices=vertices,
            vertex_offset=offset,
            xyz_part=xyz_part,
        )
        # Visual debug: display bbox as Open3D LineSet if boundaries exist
        if boundaries is not None and n_v > 0:

            bbox_vertices = np.array(vertices[offset:offset + n_v])
            # The 12 box edges, assuming 8 vertices in order from _axis_aligned_bbox_faces
            lines = np.array([
                [0,1],[1,2],[2,3],[3,0],   # bottom face
                [4,5],[5,6],[6,7],[7,4],   # top face
                [0,4],[1,5],[2,6],[3,7],   # vertical edges
            ])
            colors = [[1, 0, 0] for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(bbox_vertices),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([line_set])
        if boundaries is None or n_v == 0:
            continue

        offset += n_v
        children.append(part_id)
        city_objects[part_id] = {
            "type": "BuildingPart",
            "parents": [BUILDING_CITYOBJECT_ID],
            "geometry": [
                {
                    "type": "MultiSurface",
                    "lod": "1",
                    "boundaries": boundaries,
                }
            ],
        }

    if not children:
        cm = CityJSON()
        cm.j["metadata"] = {
            "title": "segmentor-pipeline-export",
        }
        return cm

    city_objects[BUILDING_CITYOBJECT_ID] = {
        "type": "Building",
        "children": children,
    }

    j = {
        "type": "CityJSON",
        "version": "2.0",
        "metadata": {
            "title": "segmentor-pipeline-export",
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4979",
        },
        "CityObjects": city_objects,
        "vertices": vertices,
    }
    cm = CityJSON(j=j)
    cm.compress()
    return cm


def write_pipeline_cityjson(
    result_folder_path: str,
    pipeline_las_path: str | None = None,
    cfg: config_module.SegmentorConfig | None = None,
) -> str:
    """
    Merge pipeline LAS, derive building CityJSON (parts per ground/wall/ceiling),
    write ``citygml.city.json``.

    Returns path to the written JSON file.
    """
    from modules.models import output_tools
    from modules.utils.config import SegmentorConfig

    cfg = cfg or SegmentorConfig.get_instance()
    out_path = output_tools.get_cityjson_citygml_path(result_folder_path)

    if pipeline_las_path is None:
        pipeline_las_path = output_tools.get_pipeline_output_las_path(result_folder_path)

    if not os.path.isfile(pipeline_las_path):
        raise FileNotFoundError(f"Pipeline LAS not found: {pipeline_las_path}")

    chunk_size = int(getattr(cfg, "chunk_size", 5_000_000))
    xyz, merged_attrs = merge_chunks(input_path=pipeline_las_path, chunk_size=chunk_size)
    classification = np.asarray(merged_attrs.get("classification"))
    if classification.shape[0] != len(xyz):
        raise ValueError("classification length does not match merged point count")

    mask = np.isin(classification, BUILDING_STRUCTURE_CLASSIFICATIONS)
    log.info(
        "CityJSON export: %d / %d points in GROUND/WALL/CEILING classes",
        int(np.sum(mask)),
        len(xyz),
    )

    cm = build_cityjson_for_building_shell(xyz, classification)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json_str = json.dumps(cm.j, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write(json_str)
    log.info("Wrote CityJSON to %s", out_path)
    return out_path
