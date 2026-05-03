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
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/7416",
        },
        "CityObjects": {},
        "vertices": [],
    }


def init_cityjson_file(
    output_json_path: str,
    geographical_extent: list[float] | None = None,
) -> None:
    """
    Create (or overwrite) a CityJSON file with metadata only, no geometry yet.

    Call this once before the first write_cityjson_mesh_part call so that
    geographicalExtent and referenceSystem are set on the initial payload
    rather than being filled in by the first mesh write.
    """
    payload = _cityjson_base_payload()
    if geographical_extent is not None:
        if len(geographical_extent) != 6:
            raise ValueError("geographical_extent must be a list of 6 floats: [minx, miny, minz, maxx, maxy, maxz]")
        payload["metadata"]["geographicalExtent"] = [float(v) for v in geographical_extent]
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(json_str + "\n")
    log.info("Initialised CityJSON file: %s", output_json_path)


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
            "title": "segmentor-pipeline-export",
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


def _point_cloud_to_box_solid(
    point_cloud: o3d.geometry.PointCloud,
    vertex_offset: int,
) -> tuple[list[list[float]], list[list[list[list[int]]]]]:
    points = np.asarray(point_cloud.points, dtype=np.float64)
    if points.size == 0:
        return [], []

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = maxs - mins
    if np.all(extents <= 0.0):
        # Degenerate cloud: still emit a tiny box so the object remains visible.
        extents = np.array([1e-3, 1e-3, 1e-3], dtype=np.float64)
    else:
        # Keep non-zero dimensions and minimally inflate collapsed axes.
        extents = np.where(extents <= 0.0, 1e-3, extents)

    half_x, half_y, half_z = (extents / 2.0).tolist()
    cx, cy, cz = center.tolist()

    box_vertices = [
        [cx - half_x, cy - half_y, cz - half_z],  # 0
        [cx + half_x, cy - half_y, cz - half_z],  # 1
        [cx + half_x, cy + half_y, cz - half_z],  # 2
        [cx - half_x, cy + half_y, cz - half_z],  # 3
        [cx - half_x, cy - half_y, cz + half_z],  # 4
        [cx + half_x, cy - half_y, cz + half_z],  # 5
        [cx + half_x, cy + half_y, cz + half_z],  # 6
        [cx - half_x, cy + half_y, cz + half_z],  # 7
    ]

    # CityJSON Solid boundaries hierarchy:
    # [shells -> surfaces -> rings -> vertex indices]
    surfaces = [
        [[vertex_offset + 0, vertex_offset + 1, vertex_offset + 2, vertex_offset + 3]],  # bottom
        [[vertex_offset + 4, vertex_offset + 5, vertex_offset + 6, vertex_offset + 7]],  # top
        [[vertex_offset + 0, vertex_offset + 1, vertex_offset + 5, vertex_offset + 4]],  # front
        [[vertex_offset + 1, vertex_offset + 2, vertex_offset + 6, vertex_offset + 5]],  # right
        [[vertex_offset + 2, vertex_offset + 3, vertex_offset + 7, vertex_offset + 6]],  # back
        [[vertex_offset + 3, vertex_offset + 0, vertex_offset + 4, vertex_offset + 7]],  # left
    ]
    boundaries = [surfaces]
    return box_vertices, boundaries


def write_cityjson_mesh_part(
    mesh_id: str,
    mesh: o3d.geometry.TriangleMesh,
    result_folder_path: str | None = None,
    output_json_path: str | None = None,
    append_to_existing: bool = False,
    city_object_type: str = "GenericCityObject",
) -> str:
    """
    Append a mesh part to a CityJSON file.

    Main parameters:
    - mesh_id: city object id (examples: groundSurface, wallSurface, cars)
    - mesh: Open3D triangle mesh for the object geometry
    - city_object_type: CityJSON object type string (default "Building").
      Use "GenericCityObject" for objects that are not part of the building
      structure (e.g. vehicles).

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
    if city_object_type not in {"Building", "GenericCityObject"}:
        raise ValueError("city_object_type must be 'Building' or 'GenericCityObject'")

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
            "type": city_object_type,
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
        ensure_ascii=False,
    )
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write(json_str + "\n")
    log.info("Wrote mesh '%s' to CityJSON: %s", mesh_id, out_path)
    return out_path


def write_point_cloud_to_citygml_json_box(
    shape_id: str,
    point_cloud: o3d.geometry.PointCloud,
    result_folder_path: str | None = None,
    output_json_path: str | None = None,
    append_to_existing: bool = False,
) -> str:
    """
    Append a point cloud as a cube (Solid) to a CityJSON file.

    The cube is axis-aligned and centered on the point cloud AABB center, with
    side length equal to the largest AABB extent.
    """
    if not shape_id or not isinstance(shape_id, str):
        raise ValueError("mesh_id must be a non-empty string")
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise TypeError("point_cloud must be an open3d.geometry.PointCloud")

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

    box_vertices, boundaries = _point_cloud_to_box_solid(
        point_cloud=point_cloud,
        vertex_offset=len(vertices),
    )
    if not boundaries:
        log.warning("Skipping mesh_id '%s': point cloud has no points", shape_id)
        return out_path

    vertices.extend(box_vertices)

    existing = city_objects.get(shape_id)
    if existing is None:
        city_objects[shape_id] = {
            "type": "GenericCityObject",
            "geometry": [
                {
                    "type": "Solid",
                    "lod": "1",
                    "boundaries": boundaries,
                }
            ],
        }
    else:
        geometry = existing.get("geometry")
        if not isinstance(geometry, list) or len(geometry) == 0:
            existing["geometry"] = [
                {
                    "type": "Solid",
                    "lod": "1",
                    "boundaries": boundaries,
                }
            ]
        else:
            first_geom = geometry[0]
            if first_geom.get("type") != "Solid":
                raise ValueError(
                    f"CityObject '{shape_id}' exists but first geometry is not Solid"
                )
            if append_to_existing:
                first_geom.setdefault("boundaries", [])
                if not isinstance(first_geom["boundaries"], list):
                    raise ValueError(
                        f"CityObject '{shape_id}' has invalid boundaries (expected list)"
                    )
                first_geom["boundaries"].extend(boundaries)
            else:
                first_geom["lod"] = "1"
                first_geom["boundaries"] = boundaries

    json_str = json.dumps(
        payload,
        ensure_ascii=False,
    )
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write(json_str + "\n")
    log.info("Wrote point-cloud cube '%s' to CityJSON: %s", shape_id, out_path)
    return out_path
