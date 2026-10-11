#!/usr/bin/env python3
"""
Stage-3 post-processing:
- Read clustered objects from stage-2 output LAS (`Clusters`),
- Identify each cluster via VLM from multiple isometric views,
- Relabel each cluster with the LAS code of its VLM class,
- Write `3_post_process.las` and a `data.yml` next to each cluster's screenshots.
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

from modules.models import output_tools  # noqa: E402
from modules.models.rich_media_tools import take_rich_media_screenshots  # noqa: E402
from modules.models.las_tools import las_io  # noqa: E402
from modules.models.las_tools.las_iterator import LasIteratorBasic  # noqa: E402
from modules.models.vlm_tools import (  # noqa: E402
    identify_object_from_views,
    make_cluster_record,
    write_cluster_data_yaml,
)
from modules.models import pcd_tools  # noqa: E402
from toolbox.restore_origina_pcd import restore_original_pcd_data  # noqa: E402
from modules.utils import config as config_module  # noqa: E402
from modules.utils import log_utils  # noqa: E402

log = log_utils.get_dual_logger(__name__)


def _get_vlm_views_count(cfg: config_module.SegmentorConfig) -> int:
    vlm_cfg = getattr(cfg, "vlm", {}) or {}
    views_count = int(vlm_cfg.get("views_count", 6))
    if views_count <= 0:
        raise ValueError("vlm.views_count must be greater than 0.")
    return views_count



def _merge_chunks(input_path: str, chunk_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
        # Fallback to raw reader in case iterator yielded nothing unexpectedly.
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


def _write_las(
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


def _resolve_io_paths(
    cfg: config_module.SegmentorConfig,
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


def _cluster_ids_from_attributes(merged_attrs: dict[str, np.ndarray], n_points: int) -> np.ndarray:
    if "Clusters" not in merged_attrs:
        raise KeyError(
            "Input LAS is missing 'Clusters'. Run pipeline stage with segment_objects_dbscan first."
        )
    objects_clusters = np.asarray(merged_attrs["Clusters"], dtype=np.int32)
    if objects_clusters.shape[0] != n_points:
        raise ValueError("Clusters length does not match point count.")
    return np.asarray(sorted(np.unique(objects_clusters[objects_clusters > 0])), dtype=np.int32)


def _run_object_vlm_post_processing(
    merged_attrs: dict[str, np.ndarray],
    xyz: np.ndarray,
    original_input_path: str,
    output_folder_path: str,
    vlm_traces_subdir: str,
) -> None:
    classification = np.asarray(merged_attrs.get("classification"))
    if classification.shape[0] != len(xyz):
        raise ValueError("classification length does not match point count.")

    cluster_ids = _cluster_ids_from_attributes(merged_attrs, len(xyz))
    os.makedirs(output_folder_path, exist_ok=True)
    if len(cluster_ids) == 0:
        log.info("No object clusters found in stage-2 output.")
        return

    traces_dir = os.path.join(output_folder_path, vlm_traces_subdir)
    os.makedirs(traces_dir, exist_ok=True)

    cfg = config_module.SegmentorConfig.get_instance()
    unique_trace_dirs = bool(
        (getattr(cfg, "objects_segmentation", {}) or {}).get("create_unique_trace_dirs", False)
    )

    objects_clusters = np.asarray(merged_attrs["Clusters"], dtype=np.int32)
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    n_points = len(xyz)

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

        # Sandbox-style restoration: use cluster inliers to recover original points in the same bbox.
        try:
            restored_cluster_pcd = restore_original_pcd_data(
                view_pcd,
                original_input_path,
            )
            if len(restored_cluster_pcd.points) > 0:
                screenshot_pcd = restored_cluster_pcd
            else:
                screenshot_pcd = view_pcd
        except Exception as ex:
            log.warning(
                "Failed restoring original points for cluster %s (%s); using pipeline cluster points.",
                cluster_id,
                ex,
            )
            screenshot_pcd = view_pcd

        vlm_views_count = _get_vlm_views_count(cfg)

        cluster_traces_dir = os.path.join(traces_dir, str(cluster_id)) if unique_trace_dirs else traces_dir
        view_paths = take_rich_media_screenshots(
            screenshot_pcd,
            cluster_traces_dir,
            number_of_views=vlm_views_count,
        )
        try:
            record = identify_object_from_views(
                view_paths,
                cluster_id=int(cluster_id),
                chunk_index=0,
            )
            log.info("Cluster %s identified as %s / %s.", cluster_id, record["vlm_class"], record["name"])
        except Exception as ex:
            log.warning("VLM call failed for cluster %s: %s", cluster_id, ex, exc_info=True)
            record = make_cluster_record(cluster_id=int(cluster_id), description=f"VLM call failed: {ex}")
        classification[cluster_mask] = record["classification"]
        write_cluster_data_yaml(cluster_traces_dir, record)

    merged_attrs["classification"] = classification


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stage-3 post-processing on stage-2 cluster IDs and classify cars with VLM."
    )
    parser.add_argument("--config", "-c", default="config.yml", help="Config YAML path.")
    parser.add_argument("--input", "-i", default=None, help="Optional override for input LAS path.")
    parser.add_argument("--output", "-o", default=None, help="Optional override for output LAS path.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    cfg = config_module.SegmentorConfig.initialize(config_path)
    input_path, output_path, objects_folder_path = _resolve_io_paths(cfg, args.input, args.output)
    if not os.path.isfile(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(objects_folder_path, exist_ok=True)

    obj_cfg = getattr(cfg, "objects_segmentation", {}) or {}
    vlm_traces_subdir = str(obj_cfg.get("vlm_traces_subdir", "vlm_object_traces"))
    original_input_path = os.path.abspath(str(getattr(cfg, "input_file", "")).strip())
    if not original_input_path or not os.path.isfile(original_input_path):
        print(
            "Config input_file must point to the original LAS used for restoration.",
            file=sys.stderr,
        )
        return 1

    try:
        xyz, merged_attrs = _merge_chunks(input_path=input_path, chunk_size=int(cfg.chunk_size))
        _run_object_vlm_post_processing(
            merged_attrs=merged_attrs,
            xyz=xyz,
            original_input_path=original_input_path,
            output_folder_path=objects_folder_path,
            vlm_traces_subdir=vlm_traces_subdir,
        )
    except (KeyError, ValueError, RuntimeError) as ex:
        print(str(ex), file=sys.stderr)
        return 1

    _write_las(
        input_path=input_path,
        chunk_size=int(cfg.chunk_size),
        output_path=output_path,
        xyz=xyz,
        merged_attrs=merged_attrs,
    )
    print(f"Wrote {len(xyz):,} points to {output_path}")
    print(f"Wrote per-cluster data.yml under {os.path.join(objects_folder_path, vlm_traces_subdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
