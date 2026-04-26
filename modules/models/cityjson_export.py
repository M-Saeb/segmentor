"""
Incremental CityJSON mesh export utilities.

This module provides an append-friendly API so multiple calls can write
different meshes (for example ground, walls, cars) into the same CityJSON file.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import open3d as o3d
from cjio.floatEncoder import FloatEncoder

from modules.models import output_tools
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

# Match CityJSON / cjio JSON float formatting.
json.encoder.c_make_encoder = None
json.encoder.float = FloatEncoder


def _cityjson_base_payload() -> dict[str, Any]:
    return {
        "type": "CityJSON",
        "version": "2.0",
        "metadata": {
            "title": "segmentor-pipeline-export-v4",
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/25832",
        },
        "CityObjects": {},
        "vertices": [],
    }


def _resolve_output_path(
    result_folder_path: str | None = None,
    output_json_path: str | None = None,
) -> str:
    if output_json_path:
        return output_json_path
    return output_tools.get_cityjson_citygml_path(result_folder_path)


def _load_or_init_cityjson(path: str) -> dict[str, Any]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid CityJSON payload in {path}: expected object")
        payload.setdefault("type", "CityJSON")
        payload.setdefault("version", "2.0")
        payload.setdefault("metadata", {
            "title": "segmentor-pipeline-export-v4",
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/25832",
        })
        payload.setdefault("CityObjects", {})
        payload.setdefault("vertices", [])
        if not isinstance(payload["CityObjects"], dict) or not isinstance(payload["vertices"], list):
            raise ValueError(f"Invalid CityJSON structure in {path}")
        return payload
    return _cityjson_base_payload()


def _mesh_to_boundaries_and_vertices(
    mesh: o3d.geometry.TriangleMesh,
    vertex_offset: int,
) -> tuple[list[list[float]], list[list[list[int]]]]:
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.remove_duplicated_vertices()
    mesh_copy.remove_duplicated_triangles()
    mesh_copy.remove_degenerate_triangles()
    mesh_copy.remove_non_manifold_edges()
    mesh_copy.remove_unreferenced_vertices()
    # Make triangle winding consistent where possible to reduce viewer artifacts
    # caused by back-face culling.
    try:
        mesh_copy.orient_triangles()
    except Exception:
        pass

    mesh_vertices = np.asarray(mesh_copy.vertices, dtype=np.float64)
    mesh_triangles = np.asarray(mesh_copy.triangles, dtype=np.int64)
    if mesh_vertices.size == 0 or mesh_triangles.size == 0:
        return [], []

    boundaries: list[list[list[int]]] = []
    for tri in mesh_triangles:
        boundaries.append(
            [[
                vertex_offset + int(tri[0]),
                vertex_offset + int(tri[1]),
                vertex_offset + int(tri[2]),
            ]]
        )
    return mesh_vertices.tolist(), boundaries


def write_cityjson_mesh_part(
    mesh_id: str,
    mesh: o3d.geometry.TriangleMesh,
    result_folder_path: str | None = None,
    output_json_path: str | None = None,
    append_to_existing: bool = False,
) -> str:
    """
    Append a mesh part to a CityJSON file.

    Main parameters:
    - mesh_id: city object id (examples: groundSurface, wallSurface, cars)
    - mesh: Open3D triangle mesh for the object geometry

    Optional path parameters follow project conventions:
    - result_folder_path: run output folder used with output_tools
    - output_json_path: explicit CityJSON output path (overrides result_folder_path)
    - append_to_existing: if True, append boundaries to an existing mesh_id geometry.
      If False (default), replace that mesh_id geometry with the latest mesh.

    Returns:
    - The written CityJSON output file path.
    """
    if not mesh_id or not isinstance(mesh_id, str):
        raise ValueError("mesh_id must be a non-empty string")
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("mesh must be an open3d.geometry.TriangleMesh")

    out_path = _resolve_output_path(
        result_folder_path=result_folder_path,
        output_json_path=output_json_path,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload = _load_or_init_cityjson(out_path)
    vertices = payload["vertices"]
    city_objects = payload["CityObjects"]

    if not isinstance(vertices, list):
        raise ValueError("CityJSON 'vertices' must be a list")
    if not isinstance(city_objects, dict):
        raise ValueError("CityJSON 'CityObjects' must be an object")

    mesh_vertices, boundaries = _mesh_to_boundaries_and_vertices(
        mesh=mesh,
        vertex_offset=len(vertices),
    )
    if not boundaries:
        log.warning("Skipping mesh_id '%s': mesh has no triangles", mesh_id)
        return out_path

    vertices.extend(mesh_vertices)

    existing = city_objects.get(mesh_id)
    if existing is None:
        city_objects[mesh_id] = {
            "type": "GenericCityObject",
            "geometry": [
                {
                    "type": "MultiSurface",
                    "lod": "2",
                    "boundaries": boundaries,
                }
            ],
        }
    else:
        geometry = existing.get("geometry")
        if not isinstance(geometry, list) or len(geometry) == 0:
            existing["geometry"] = [
                {
                    "type": "MultiSurface",
                    "lod": "2",
                    "boundaries": boundaries,
                }
            ]
        else:
            first_geom = geometry[0]
            if first_geom.get("type") != "MultiSurface":
                raise ValueError(
                    f"CityObject '{mesh_id}' exists but first geometry is not MultiSurface"
                )
            if append_to_existing:
                first_geom.setdefault("boundaries", [])
                if not isinstance(first_geom["boundaries"], list):
                    raise ValueError(
                        f"CityObject '{mesh_id}' has invalid boundaries (expected list)"
                    )
                first_geom["boundaries"].extend(boundaries)
            else:
                first_geom["lod"] = "2"
                first_geom["boundaries"] = boundaries

    json_str = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write(json_str + "\n")
    log.info("Wrote mesh '%s' to CityJSON: %s", mesh_id, out_path)
    return out_path
