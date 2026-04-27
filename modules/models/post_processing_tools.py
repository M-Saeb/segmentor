"""
Post-processing: merge pipeline LAS (`2_pipeline.las`), run DBSCAN (`segment_objects_dbscan`),
VLM per cluster, write `3_post_process.las` and object registry YAML.

Pipeline output is classification-only (no DBSCAN clusters); clusters are created here.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
from modules.models import output_tools
from modules.models import pcd_tools
from modules.models.vlm_tools import (
    empty_objects_registry,
    identify_object_from_views,
    register_identified_cluster,
    write_identified_objects_yaml,
)
from modules.models.las_tools import las_io
from modules.models.las_tools.las_iterator import LasIteratorBasic
from modules.models.clusters_manager import ClustersManager
from modules.models.rich_media_tools import take_rich_media_screenshots
from modules.models.runtime_report import RuntimeReport
from modules.utils import constants
from modules.utils.config import SegmentorConfig
from modules.utils.log_utils import get_dual_logger
from toolbox.restore_origina_pcd import restore_original_points_in_inliers

if TYPE_CHECKING:
    from modules.utils import config as config_module

log = get_dual_logger(__name__)


def _want_runtime_report() -> bool:
    return bool(getattr(SegmentorConfig.get_instance(), "collect_runtime_report", False))


def _get_vlm_views_count() -> int:
    cfg = SegmentorConfig.get_instance()
    vlm_cfg = getattr(cfg, "vlm", {}) or {}
    views_count = int(vlm_cfg.get("views_count", 6))
    if views_count <= 0:
        raise ValueError("vlm.views_count must be greater than 0.")
    return views_count


def merge_chunks(input_path: str, chunk_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xyz_parts: list[np.ndarray] = []
    attr_parts: dict[str, list[np.ndarray]] = {}

    iterator = LasIteratorBasic(file_path=input_path, chunk_size=chunk_size)
    total_chunks = iterator.get_total_items_number()
    for index in range(total_chunks):
        if not iterator.iterate_chunk(0 if index == 0 else 1):
            break
        pcd, points_attributes = iterator.get_current_chunk()
        log.info("Merging chunk %d", index)
        xyz_parts.append(np.asarray(pcd.points, dtype=np.float64))
        for dim, arr in points_attributes.items():
            attr_parts.setdefault(dim, []).append(np.asarray(arr))

    if not xyz_parts:
        for index, (pcd, points_attributes) in enumerate(
            las_io.read_chunk_las_file(input_path, chunk_size)
        ):
            log.info("Merging chunk %d (fallback reader)", index)
            xyz_parts.append(np.asarray(pcd.points, dtype=np.float64))
            for dim, arr in points_attributes.items():
                attr_parts.setdefault(dim, []).append(np.asarray(arr))

    if not xyz_parts:
        raise RuntimeError("No points were read from the input.")

    xyz = np.vstack(xyz_parts)
    merged_attrs: dict[str, np.ndarray] = {
        dim: np.concatenate(chunks, axis=0) for dim, chunks in attr_parts.items()
    }
    return xyz, merged_attrs


def write_las_merged(
    input_path: str,
    chunk_size: int,
    output_path: str,
    xyz: np.ndarray,
    merged_attrs: dict[str, np.ndarray],
) -> None:
    n = len(xyz)
    chunk_size = int(chunk_size) if chunk_size is not None else n
    if chunk_size <= 0:
        chunk_size = max(n, 1)

    original_header = las_io.get_original_las_header(input_path)
    base_header = las_io.create_new_las_header(original_header)
    las_io.ensure_object_clusters_in_header(base_header)
    las_io.init_las_file(output_path, base_header)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_pcd = o3d.geometry.PointCloud()
        chunk_pcd.points = o3d.utility.Vector3dVector(xyz[start:end])
        chunk_attrs = {dim: np.asarray(arr)[start:end] for dim, arr in merged_attrs.items()}
        las_io.write_chunk_las_file(
            pcd=chunk_pcd,
            points_attributes=chunk_attrs,
            full_path=output_path,
            header=base_header,
        )


def resolve_io_paths(
    cfg: "config_module.SegmentorConfig",
    input_override: str | None,
    output_override: str | None,
) -> tuple[str, str, str]:
    result_cfg = getattr(cfg, "result", {}) or {}
    result_folder = str(result_cfg.get("output_folder_path", "")).strip()
    if result_folder:
        result_folder = os.path.abspath(result_folder)

    if result_folder:
        default_input = output_tools.get_pipeline_output_las_path(result_folder)
    else:
        default_input = str(getattr(cfg, "input_file", "2_pipeline.las"))
    default_output = os.path.join(result_folder, "3_post_process.las") if result_folder else "3_post_process.las"

    input_path = os.path.abspath(input_override) if input_override else os.path.abspath(default_input)
    output_path = os.path.abspath(output_override) if output_override else os.path.abspath(default_output)
    objects_folder = result_folder or os.path.dirname(output_path) or os.getcwd()
    return input_path, output_path, os.path.abspath(objects_folder)


def cluster_ids_from_attributes(merged_attrs: dict[str, np.ndarray], n_points: int) -> np.ndarray:
    if "Clusters" not in merged_attrs:
        return np.array([], dtype=np.int32)
    objects_clusters = np.asarray(merged_attrs["Clusters"], dtype=np.int32)
    if objects_clusters.shape[0] != n_points:
        raise ValueError("Clusters length does not match point count.")
    return np.asarray(sorted(np.unique(objects_clusters[objects_clusters > 0])), dtype=np.int32)


def _axis_aligned_bounds_from_pcd(pcd: o3d.geometry.PointCloud) -> dict[str, list[float]]:
    bbox = pcd.get_axis_aligned_bounding_box()
    return {
        "min": np.asarray(bbox.get_min_bound(), dtype=float).tolist(),
        "max": np.asarray(bbox.get_max_bound(), dtype=float).tolist(),
    }



def run_object_vlm_post_processing(
    merged_attrs: dict[str, np.ndarray],
    xyz: np.ndarray,
    original_input_path: str,
    output_folder_path: str,
    ollama_model: str,
    vlm_traces_subdir: str,
) -> None:
    classification = np.asarray(merged_attrs.get("classification"))
    if classification.shape[0] != len(xyz):
        raise ValueError("classification length does not match point count.")

    cluster_ids = cluster_ids_from_attributes(merged_attrs, len(xyz))
    os.makedirs(output_folder_path, exist_ok=True)
    registry = empty_objects_registry()
    if len(cluster_ids) == 0:
        log.info("No object clusters after DBSCAN; skipping VLM.")
        write_identified_objects_yaml(output_folder_path, registry)
        return

    traces_dir = os.path.join(output_folder_path, vlm_traces_subdir)
    os.makedirs(traces_dir, exist_ok=True)

    objects_clusters = np.asarray(merged_attrs["Clusters"], dtype=np.int32)
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    n_points = len(xyz)

    log.info("Running VLM post-processing for %d clusters", len(cluster_ids))

    rpt = RuntimeReport.get_instance() if _want_runtime_report() else None

    for cluster_id in cluster_ids.tolist():
        cluster_mask = objects_clusters == int(cluster_id)
        if not np.any(cluster_mask):
            log.warning("Skipping empty cluster %s", cluster_id)
            continue
        indices = np.where(cluster_mask)[0]
        view_pcd = pcd_all.select_by_index(indices.tolist())

        view_attrs: dict[str, np.ndarray] = {}
        for dim, arr in merged_attrs.items():
            arr_np = np.asarray(arr)
            if arr_np.shape[0] == n_points:
                view_attrs[dim] = arr_np[indices]
            else:
                view_attrs[dim] = arr

        try:
            pcd_tools.apply_rgb_colors_to_pcd(view_pcd, view_attrs)
        except ValueError as ex:
            log.warning("Cluster %s has invalid RGB (%s); skipping VLM.", cluster_id, ex)
            continue

        try:
            restored_cluster_pcd = restore_original_points_in_inliers(
                view_pcd,
                original_input_path,
            )
            if len(restored_cluster_pcd.points) > 0:
                screenshot_pcd = restored_cluster_pcd
            else:
                screenshot_pcd = view_pcd
        except Exception as ex:
            log.warning(
                "Failed restoring original points for cluster %s (%s); using cluster points.",
                cluster_id,
                ex,
            )
            screenshot_pcd = view_pcd

        t_img = time.perf_counter()
        view_paths = take_rich_media_screenshots(
            screenshot_pcd,
            traces_dir,
            number_of_views=_get_vlm_views_count(),
        )

        if rpt is not None:
            rpt.add_image_creation_seconds(time.perf_counter() - t_img)
        try:
            record = identify_object_from_views(
                view_paths,
                cluster_id=int(cluster_id),
                chunk_index=0,
                ollama_model=ollama_model,
                runtime_report=rpt,
            )
            record["bounds"] = _axis_aligned_bounds_from_pcd(screenshot_pcd)
            register_identified_cluster(registry, record)
            classification[cluster_mask] = record["classification"]
            log.info("Cluster %s identified as [[[ %s ]]]: %s.", cluster_id, record.get("ai_label"), record.get("description"))
        except Exception as ex:
            log.warning("VLM call failed for cluster %s: %s", cluster_id, ex, exc_info=True)

    write_identified_objects_yaml(output_folder_path, registry)
    merged_attrs["classification"] = classification


def ensure_object_clusters_array(merged_attrs: dict[str, np.ndarray], n_points: int) -> None:
    if "Clusters" not in merged_attrs:
        merged_attrs["Clusters"] = np.zeros(n_points, dtype=np.int32)
        return
    objects_clusters = np.asarray(merged_attrs["Clusters"], dtype=np.int32)
    if objects_clusters.shape[0] != n_points:
        raise ValueError("Clusters length does not match point count.")
    merged_attrs["Clusters"] = objects_clusters


def run_dbscan_on_merged(
    xyz: np.ndarray,
    merged_attrs: dict[str, np.ndarray],
    eps: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, dict[str, np.ndarray]]:
    """Run `segment_objects_dbscan` on a full merged cloud (mutates attrs copy)."""
    ensure_object_clusters_array(merged_attrs, len(xyz))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    ClustersManager.get_instance().reset(start=1)
    return pcd_tools.segment_objects_dbscan(pcd, merged_attrs, eps=eps, min_points=min_points)


def run_post_process_pipeline(
    *,
    input_path: str,
    output_path: str,
    objects_folder_path: str,
    chunk_size: int,
    eps: float,
    min_points: int,
    ollama_model: str,
    vlm_traces_subdir: str,
    original_input_path: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Merge LAS → DBSCAN → VLM → returns final xyz and attrs (caller may write LAS).

    `merged_attrs` is copied internally before DBSCAN so callers can retain baselines.
    """
    xyz, merged_attrs = merge_chunks(input_path=input_path, chunk_size=chunk_size)
    working = copy_merged_attributes(merged_attrs)
    ensure_object_clusters_array(working, len(xyz))
    _, working = run_dbscan_on_merged(xyz, working, eps=eps, min_points=min_points)
    if _want_runtime_report():
        RuntimeReport.get_instance().reset()
    run_object_vlm_post_processing(
        merged_attrs=working,
        xyz=xyz,
        original_input_path=original_input_path,
        output_folder_path=objects_folder_path,
        ollama_model=ollama_model,
        vlm_traces_subdir=vlm_traces_subdir,
    )
    write_las_merged(
        input_path=input_path,
        chunk_size=chunk_size,
        output_path=output_path,
        xyz=xyz,
        merged_attrs=working,
    )
    if _want_runtime_report():
        RuntimeReport.get_instance().write_yaml(objects_folder_path)
    return xyz, working


def copy_merged_attributes(attrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, copy=True) for k, v in attrs.items()}


# --- Preview coloring (no compression; used by PostProcessingController) ---

_COLOR_GROUND = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_COLOR_INIT = np.array([0.5, 0.5, 0.5], dtype=np.float64)


def point_indices_init_and_ground(classification: np.ndarray) -> np.ndarray:
    c = np.asarray(classification, dtype=np.int64)
    return np.where(
        (c == constants.PCB_CLASSIFICATION_INIT) | (c == constants.PCB_CLASSIFICATION_GROUND)
    )[0]


def point_indices_post_dbscan_preview(classification: np.ndarray) -> np.ndarray:
    """INIT, GROUND, or OBJECTS (clustered); excludes wall/ceiling."""
    c = np.asarray(classification, dtype=np.int64)
    return np.where(
        (c == constants.PCB_CLASSIFICATION_INIT)
        | (c == constants.PCB_CLASSIFICATION_GROUND)
        | (c == constants.PCB_CLASSIFICATION_OBJECTS)
    )[0]


def build_init_ground_preview_pcd(
    xyz: np.ndarray,
    points_attributes: dict[str, np.ndarray],
) -> o3d.geometry.PointCloud:
    classification = np.asarray(points_attributes["classification"])
    idx = point_indices_init_and_ground(classification)
    if idx.size == 0:
        return o3d.geometry.PointCloud()
    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(xyz[idx])
    cc = classification[idx]
    colors = np.empty((len(idx), 3), dtype=np.float64)
    colors[cc == constants.PCB_CLASSIFICATION_GROUND] = _COLOR_GROUND
    colors[cc == constants.PCB_CLASSIFICATION_INIT] = _COLOR_INIT
    sub.colors = o3d.utility.Vector3dVector(colors)
    return sub


def build_dbscan_preview_pcd(
    xyz: np.ndarray,
    points_attributes: dict[str, np.ndarray],
    rng: np.random.Generator | None = None,
) -> o3d.geometry.PointCloud:
    """
    Colors: random per Clusters > 0, green ground, gray INIT noise, yellow-ish for OBJECTS without id omitted — unclustered INIT stays gray.
    """
    if rng is None:
        rng = np.random.default_rng()

    classification = np.asarray(points_attributes["classification"], dtype=np.int64)
    objects_clusters = np.asarray(
        points_attributes.get("Clusters", np.zeros(len(xyz), dtype=np.int32)),
        dtype=np.int32,
    )
    idx = point_indices_post_dbscan_preview(classification)
    if idx.size == 0:
        return o3d.geometry.PointCloud()

    sub = o3d.geometry.PointCloud()
    sub.points = o3d.utility.Vector3dVector(xyz[idx])
    cls_s = classification[idx]
    oc_s = objects_clusters[idx]

    colors = np.empty((len(idx), 3), dtype=np.float64)
    unique_ids = np.unique(oc_s[oc_s > 0])
    id_to_color: dict[int, np.ndarray] = {}
    for cid in unique_ids.tolist():
        id_to_color[int(cid)] = rng.random(3).astype(np.float64)

    for i in range(len(idx)):
        c = int(cls_s[i])
        oid = int(oc_s[i])
        if c == constants.PCB_CLASSIFICATION_GROUND:
            colors[i] = _COLOR_GROUND
        elif c == constants.PCB_CLASSIFICATION_INIT:
            colors[i] = _COLOR_INIT
        elif oid > 0:
            colors[i] = id_to_color[oid]
        else:
            colors[i] = _COLOR_INIT

    sub.colors = o3d.utility.Vector3dVector(colors)
    return sub
