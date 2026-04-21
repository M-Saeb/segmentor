"""
CityJSON export for pipeline results (OGC CityJSON JSON encoding).

This is **not** CityGML XML; filename ``citygml.city.json`` is historical. Content
follows CityJSON v2.x via ``cjio``, aligned with CityGML semantics. Native
CityGML 3 XML export would be a separate toolchain.
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
_EPS_PAD = 1e-3


def _axis_aligned_bbox_faces(
    xyz_min: np.ndarray, xyz_max: np.ndarray
) -> tuple[list[list[float]], list[list[list[int]]]]:
    """Return 8 corner vertices and MultiSurface boundaries (index rings)."""
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
    # Six quads; outward winding for each face.
    boundaries: list[list[list[int]]] = [
        [[0, 1, 2, 3]],
        [[4, 7, 6, 5]],
        [[0, 1, 5, 4]],
        [[1, 2, 6, 5]],
        [[2, 3, 7, 6]],
        [[3, 0, 4, 7]],
    ]
    return verts, boundaries


def build_cityjson_for_building_shell(
    xyz_building: np.ndarray,
) -> CityJSON:
    """
    Build a CityJSON document with one ``Building`` and LOD1 MultiSurface
    shoebox around the axis-aligned bbox of the given points.
    """
    if xyz_building.size == 0:
        cm = CityJSON()
        cm.j["metadata"] = {
            "title": "segmentor-pipeline-export",
        }
        return cm

    xyz_min = xyz_building.min(axis=0)
    xyz_max = xyz_building.max(axis=0)
    for i in range(3):
        if math.isclose(xyz_min[i], xyz_max[i], abs_tol=1e-9):
            xyz_min[i] -= _EPS_PAD
            xyz_max[i] += _EPS_PAD
        else:
            xyz_min[i] -= _EPS_PAD
            xyz_max[i] += _EPS_PAD

    verts, boundaries = _axis_aligned_bbox_faces(xyz_min, xyz_max)

    geom = {
        "type": "MultiSurface",
        "lod": "1",
        "boundaries": boundaries,
    }
    city_obj = {
        "type": "Building",
        "geometry": [geom],
    }

    j = {
        "type": "CityJSON",
        "version": "2.0",
        "metadata": {
            "title": "segmentor-pipeline-export",
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4979",
        },
        "CityObjects": {BUILDING_CITYOBJECT_ID: city_obj},
        "vertices": verts,
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
    Merge pipeline LAS, derive a building shell CityJSON, write ``citygml.city.json``.

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
    xyz_b = np.asarray(xyz[mask], dtype=np.float64)

    log.info(
        "CityJSON export: %d / %d points used for building bbox (classes GROUND/WALL/CEILING)",
        int(xyz_b.shape[0]),
        len(xyz),
    )

    cm = build_cityjson_for_building_shell(xyz_b)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json_str = json.dumps(cm.j, separators=(",", ":"))
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write(json_str)
    log.info("Wrote CityJSON to %s", out_path)
    return out_path
