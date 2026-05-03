#!/usr/bin/env python3
"""
Stage-4 CityGML/CityJSON export:
- Read a post-processed LAS file (`3_post_process.las` by default),
- Reconstruct meshes for ground, ceiling, walls, and cars,
- Write all meshes to a single CityJSON file,
- Optionally display the result in an Open3D visualiser window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.models.las_tools import las_io  # noqa: E402
from modules.models.cityjson_export import init_cityjson_file, write_cityjson_mesh_part  # noqa: E402
from modules.models.mesh_tools.ball_neighbourhood_mesh_v2 import (  # noqa: E402
    create_ball_neighbourhood_mesh,
)
from modules.models import output_tools  # noqa: E402
from modules.utils import config as config_module  # noqa: E402
from modules.utils import constants  # noqa: E402
from modules.utils import log_utils  # noqa: E402

log = log_utils.get_dual_logger(__name__)

# Default meshing parameters per iteration step.
_DEFAULT_BALL_RADII = [1, 3, 5, 6, 9]
_DEFAULT_MIN_NEIGHBOURS = [10, 7, 4, 2, 1]
_MAX_SUPPORTED_ITERATIONS = len(_DEFAULT_BALL_RADII)


# ---------------------------------------------------------------------------
# Mesh construction helpers
# ---------------------------------------------------------------------------

def _pcd_to_mesh(pcd: o3d.geometry.PointCloud, iterations: int) -> o3d.geometry.TriangleMesh:
    """Iteratively refine a point cloud into a mesh."""
    if not (1 <= iterations <= _MAX_SUPPORTED_ITERATIONS):
        raise ValueError(
            f"iterations must be between 1 and {_MAX_SUPPORTED_ITERATIONS}, got {iterations}"
        )
    current_pcd = pcd
    mesh = None
    for i in range(iterations):
        mesh = create_ball_neighbourhood_mesh(
            current_pcd,
            ball_radius=_DEFAULT_BALL_RADII[i],
            min_neighbours=_DEFAULT_MIN_NEIGHBOURS[i],
            max_faces=10_000_000,
        )
        if i < iterations - 1:
            sample_count = max(100, int(len(current_pcd.points) * 0.25))
            current_pcd = mesh.sample_points_uniformly(sample_count)
    return mesh


def _select_by_classification(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    classification_value: int,
) -> o3d.geometry.PointCloud:
    mask = np.asarray(points_attributes["classification"]) == classification_value
    indices = np.where(mask)[0]
    return pcd.select_by_index(indices)


# ---------------------------------------------------------------------------
# Per-stage mesh builders
# ---------------------------------------------------------------------------

def build_ground_mesh(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    iterations: int,
) -> o3d.geometry.TriangleMesh:
    ground_pcd = _select_by_classification(pcd, points_attributes, constants.PCB_CLASSIFICATION_GROUND)
    if len(ground_pcd.points) < 4:
        raise RuntimeError("Not enough ground points to build a mesh.")
    log.info("Building ground mesh from %d points (%d iterations)", len(ground_pcd.points), iterations)
    return _pcd_to_mesh(ground_pcd, iterations)


def build_ceiling_mesh(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    iterations: int,
) -> o3d.geometry.TriangleMesh:
    ceiling_pcd = _select_by_classification(pcd, points_attributes, constants.PCB_CLASSIFICATION_CEILING)
    if len(ceiling_pcd.points) < 4:
        raise RuntimeError("Not enough ceiling points to build a mesh.")
    log.info("Building ceiling mesh from %d points (%d iterations)", len(ceiling_pcd.points), iterations)
    return _pcd_to_mesh(ceiling_pcd, iterations)


def build_wall_meshes(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    iterations: int,
    min_surface_size: int = 100,
) -> list[o3d.geometry.TriangleMesh]:
    wall_pcd = _select_by_classification(pcd, points_attributes, constants.PCB_CLASSIFICATION_WALL)
    if len(wall_pcd.points) == 0:
        log.warning("No wall points found; skipping wall meshes.")
        return []

    log.info("Clustering %d wall points with DBSCAN...", len(wall_pcd.points))
    labels = np.asarray(wall_pcd.cluster_dbscan(eps=1, min_points=50))
    unique_labels = np.unique(labels)
    log.info("Found %d wall clusters", (unique_labels >= 0).sum())

    meshes: list[o3d.geometry.TriangleMesh] = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_pcd = wall_pcd.select_by_index(np.where(labels == label)[0])
        try:
            mesh = _pcd_to_mesh(cluster_pcd, iterations)
        except Exception as ex:
            log.warning("Skipping wall cluster %d: %s", label, ex)
            continue
        if len(mesh.vertices) < min_surface_size or len(mesh.triangles) < min_surface_size:
            # too small to be meaningful, likely noise
            continue
        mesh.paint_uniform_color(np.random.rand(3))
        meshes.append(mesh)
        log.info("Built wall mesh %d (%d vertices)", len(meshes), len(mesh.vertices))
    return meshes


def build_car_meshes(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    iterations: int,
    min_surface_size: int = 100,
) -> list[o3d.geometry.TriangleMesh]:
    cluster_vals = np.asarray(points_attributes["Clusters"])
    class_vals = np.asarray(points_attributes["classification"])
    unique_clusters = np.unique(cluster_vals)
    unique_clusters = unique_clusters[unique_clusters > 0]

    meshes: list[o3d.geometry.TriangleMesh] = []
    for cluster_id in unique_clusters:
        mask = cluster_vals == cluster_id
        if constants.PCB_CLASSIFICATION_CAR not in class_vals[mask]:
            continue
        cluster_pcd = pcd.select_by_index(np.where(mask)[0])
        try:
            mesh = _pcd_to_mesh(cluster_pcd, iterations)
        except Exception as ex:
            log.warning("Skipping car cluster %d: %s", cluster_id, ex)
            continue
        if mesh is None or len(mesh.vertices) < min_surface_size or len(mesh.triangles) < min_surface_size:
            # too small to be meaningful, likely noise
            continue
        mesh.paint_uniform_color(np.random.rand(3))
        meshes.append(mesh)
        log.info("Built car mesh %d for cluster %d", len(meshes), cluster_id)
    return meshes


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _visualise(meshes: list[o3d.geometry.TriangleMesh], window_name: str = "CityGML Export") -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for mesh in meshes:
        vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_io_paths(
    cfg: config_module.SegmentorConfig,
    input_override: str | None,
    output_override: str | None,
) -> tuple[str, str, str]:
    result_cfg = getattr(cfg, "result", {}) or {}
    result_folder = str(result_cfg.get("output_folder_path", "")).strip()
    if result_folder:
        result_folder = os.path.abspath(result_folder)

    default_input = (
        os.path.join(result_folder, "3_post_process.las")
        if result_folder
        else "3_post_process.las"
    )
    default_output = (
        output_tools.get_cityjson_citygml_path(result_folder)
        if result_folder
        else "citygml.city.json"
    )

    input_path = os.path.abspath(input_override) if input_override else os.path.abspath(default_input)
    output_path = os.path.abspath(output_override) if output_override else os.path.abspath(default_output)
    return input_path, output_path, result_folder or os.path.dirname(output_path) or os.getcwd()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a post-processed LAS file to CityGML/CityJSON with mesh reconstruction."
    )
    parser.add_argument("--config", "-c", default="config.yml", help="Config YAML path.")
    parser.add_argument("--input", "-i", default=None, help="Override input LAS path.")
    parser.add_argument("--output", "-o", default=None, help="Override output CityJSON path.")
    parser.add_argument(
        "--display",
        type=bool,
        default=True,
        help="Open an Open3D window to visualise the exported meshes.",
    )
    parser.add_argument(
        "--ground-iterations",
        type=int,
        default=2,
        metavar="N",
        help=f"Mesh refinement iterations for the ground stage (1–{_MAX_SUPPORTED_ITERATIONS}, default 2).",
    )
    parser.add_argument(
        "--ceiling-iterations",
        type=int,
        default=2,
        metavar="N",
        help=f"Mesh refinement iterations for the ceiling stage (1–{_MAX_SUPPORTED_ITERATIONS}, default 2).",
    )
    parser.add_argument(
        "--wall-iterations",
        type=int,
        default=1,
        metavar="N",
        help=f"Mesh refinement iterations for the wall stage (1–{_MAX_SUPPORTED_ITERATIONS}, default 1).",
    )
    parser.add_argument(
        "--min-wall-surface-size",
        type=int,
        default=200,
        metavar="N",
        help=f"Minimum number of vertices/triangles for a surface to be included (default 200).",
    )
    parser.add_argument(
        "--car-iterations",
        type=int,
        default=1,
        metavar="N",
        help=f"Mesh refinement iterations for the car stage (1–{_MAX_SUPPORTED_ITERATIONS}, default 1).",
    )
    parser.add_argument(
        "--min-car-surface-size",
        type=int,
        default=50,
        metavar="N",
        help=f"Minimum number of vertices/triangles for a surface to be included (default 50).",
    )
    args = parser.parse_args()

    for flag, value in [
        ("--ground-iterations", args.ground_iterations),
        ("--ceiling-iterations", args.ceiling_iterations),
        ("--wall-iterations", args.wall_iterations),
        ("--car-iterations", args.car_iterations),
    ]:
        if not (1 <= value <= _MAX_SUPPORTED_ITERATIONS):
            print(
                f"{flag} must be between 1 and {_MAX_SUPPORTED_ITERATIONS}, got {value}",
                file=sys.stderr,
            )
            return 1

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = config_module.SegmentorConfig.initialize(config_path)
    input_path, output_path, result_folder = _resolve_io_paths(cfg, args.input, args.output)

    if not os.path.isfile(input_path):
        print(f"Input LAS not found: {input_path}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)
        log.info("Removed existing output file: %s", output_path)

    log.info("Reading LAS: %s", input_path)
    try:
        pcd, points_attributes = las_io.read_full_las_file(input_path)
    except Exception as ex:
        print(f"Failed to read LAS file: {ex}", file=sys.stderr)
        return 1

    log.info("Loaded %d points", len(pcd.points))

    bbox = pcd.get_axis_aligned_bounding_box()
    min_b = np.asarray(bbox.get_min_bound()).tolist()
    max_b = np.asarray(bbox.get_max_bound()).tolist()
    geographical_extent = [min_b[0], min_b[1], min_b[2], max_b[0], max_b[1], max_b[2]]

    init_cityjson_file(
        output_json_path=output_path,
        geographical_extent=geographical_extent,
    )
    log.info("CityJSON initialised with extent %s", geographical_extent)

    all_meshes: list[o3d.geometry.TriangleMesh] = []

    # --- Ground ---
    try:
        ground_mesh = build_ground_mesh(pcd, points_attributes, args.ground_iterations)
        ground_mesh.paint_uniform_color(np.random.rand(3))
        write_cityjson_mesh_part(
            mesh_id="groundSurface",
            mesh=ground_mesh,
            output_json_path=output_path,
            city_object_type="Building",
        )
        all_meshes.append(ground_mesh)
        log.info("Ground mesh written (%d vertices, %d triangles)", len(ground_mesh.vertices), len(ground_mesh.triangles))
    except Exception as ex:
        log.error("Ground mesh failed: %s", ex)

    # --- Ceiling ---
    try:
        ceiling_mesh = build_ceiling_mesh(pcd, points_attributes, args.ceiling_iterations)
        ceiling_mesh.paint_uniform_color(np.random.rand(3))
        write_cityjson_mesh_part(
            mesh_id="ceilingSurface",
            mesh=ceiling_mesh,
            output_json_path=output_path,
            city_object_type="Building",
        )
        all_meshes.append(ceiling_mesh)
        log.info("Ceiling mesh written (%d vertices, %d triangles)", len(ceiling_mesh.vertices), len(ceiling_mesh.triangles))
    except Exception as ex:
        log.error("Ceiling mesh failed: %s", ex)

    # --- Walls ---
    try:
        wall_meshes = build_wall_meshes(
            pcd,
            points_attributes,
            args.wall_iterations,
            min_surface_size=args.min_wall_surface_size
        )
        for wall_idx, wall_mesh in enumerate(wall_meshes):
            write_cityjson_mesh_part(
                mesh_id=f"wallSurface_{wall_idx}",
                mesh=wall_mesh,
                output_json_path=output_path,
                city_object_type="Building",
            )
        all_meshes.extend(wall_meshes)
        log.info("Wall meshes written: %d", len(wall_meshes))
    except Exception as ex:
        log.error("Wall meshes failed: %s", ex)

    # --- Cars ---
    try:
        car_meshes = build_car_meshes(
            pcd,
            points_attributes,
            args.car_iterations,
            min_surface_size=args.min_car_surface_size,
        )
        for car_idx, car_mesh in enumerate(car_meshes):
            write_cityjson_mesh_part(
                mesh_id=f"carSurface_{car_idx}",
                mesh=car_mesh,
                output_json_path=output_path,
                city_object_type="GenericCityObject",
            )
        all_meshes.extend(car_meshes)
        log.info("Car meshes written: %d", len(car_meshes))
    except Exception as ex:
        log.error("Car meshes failed: %s", ex)

    if not all_meshes:
        print("No meshes were produced; output may be empty.", file=sys.stderr)
        return 1

    print(f"Wrote CityJSON to: {output_path}")

    if args.display:
        log.info("Opening visualiser with %d meshes", len(all_meshes))
        _visualise(all_meshes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
