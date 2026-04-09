#!/usr/bin/env python3
"""
Re-segment object points in a LAS file with DBSCAN: classification 5 → per-cluster 6, 7, …

Reads the full file via LasPipelineWrapper (chunk merge), runs Open3D DBSCAN only on points
with classification 5 (objects), assigns sequential classifications starting at 6 for
each cluster; DBSCAN noise remains 5. All other points and attributes are unchanged.

Run from the project root so `modules` imports resolve, e.g.:

    python scripts/post_processing.py -i input.las -o output.las
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.models import LasPipelineWrapper  # noqa: E402
from modules.models.ai_object_detector import (  # noqa: E402
    identify_object_from_views,
    merge_identified_objects_record,
)
from modules.utils import config as config_module  # noqa: E402
from modules.utils import constants  # noqa: E402
from modules.utils import log_utils  # noqa: E402

log = log_utils.get_dual_logger(__name__)

_SKIP_DIMS = frozenset({"object_cluster_id"})


def _mask_indices_for_object_vlm_view(
    classification: np.ndarray,
    object_cluster_id: np.ndarray,
    cluster_id: int,
) -> np.ndarray:
    """Ground + selected object points only (same view policy as segment_objects)."""
    classification = np.asarray(classification)
    object_cluster_id = np.asarray(object_cluster_id, dtype=np.int32)
    ground = classification == constants.PCB_CLASSIFICATION_GROUND
    this_object = object_cluster_id == cluster_id
    # return np.where(ground | this_object)[0]
    return np.where(this_object)[0]


def _apply_rgb_colors_to_pcd(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict[str, np.ndarray],
) -> None:
    """Set pcd.colors from LAS red/green/blue attributes."""
    red = np.asarray(points_attributes.get("red"))
    green = np.asarray(points_attributes.get("green"))
    blue = np.asarray(points_attributes.get("blue"))
    n = len(pcd.points)
    if not (len(red) == len(green) == len(blue) == n):
        raise ValueError("RGB attribute lengths do not match point count")
    rgb = np.vstack((red, green, blue)).T.astype(float)
    max_val = np.max(rgb)
    if max_val > 0:
        rgb /= max_val
    pcd.colors = o3d.utility.Vector3dVector(rgb)


def _take_isometric_screenshots(
    pcd: o3d.geometry.PointCloud,
    dir_name: str,
    file_prefix: str,
) -> tuple[str, str]:
    """
    Save front/back isometric PNGs and return paths:
      (<dir>/<prefix>_front_isometric.png, <dir>/<prefix>_back_isometric.png)
    """
    os.makedirs(dir_name, exist_ok=True)
    width, height = 640, 480
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("cloud", pcd, mat)

    center = np.asarray(pcd.get_center(), dtype=float)
    bounds = pcd.get_axis_aligned_bounding_box()
    diag = float((bounds.get_max_bound() - bounds.get_min_bound()).max())
    radius = max(1e-6, diag * 1.0)
    up = [0.0, 0.0, 1.0]

    elev = np.deg2rad(35.0)
    z_off = radius * np.sin(elev)
    horiz_r = radius * np.cos(elev)

    views = {
        "front_isometric": 225.0,
        "back_isometric": 45.0,
    }
    out_paths: dict[str, str] = {}
    for name, az_deg in views.items():
        theta = np.deg2rad(az_deg)
        x_off = horiz_r * np.cos(theta)
        y_off = horiz_r * np.sin(theta)
        eye = center + np.array([x_off, y_off, z_off], dtype=float)
        renderer.setup_camera(60.0, center, eye, up)
        img = renderer.render_to_image()
        fname = f"{file_prefix}_{name}.png"
        out_path = os.path.join(dir_name, fname)
        o3d.io.write_image(out_path, img)
        out_paths[name] = out_path
        log.debug("Saved %s", out_path)

    return out_paths["front_isometric"], out_paths["back_isometric"]


def _merge_chunks(wrap: LasPipelineWrapper) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xyz_parts: list[np.ndarray] = []
    attr_parts: dict[str, list[np.ndarray]] = {}

    for _index, pcd, points_attributes in wrap.iter_input_chunks():
        log.info("Merging chunk %d", _index)
        xyz_parts.append(np.asarray(pcd.points, dtype=np.float64))
        for dim, arr in points_attributes.items():
            attr_parts.setdefault(dim, []).append(np.asarray(arr))

    if not xyz_parts:
        raise RuntimeError("No points were read from the input.")

    log.info("Merged %d chunks", len(xyz_parts))
    xyz = np.vstack(xyz_parts)
    merged_attrs: dict[str, np.ndarray] = {
        dim: np.concatenate(chunks, axis=0) for dim, chunks in attr_parts.items()
    }
    return xyz, merged_attrs


def _write_las(
    wrap: LasPipelineWrapper,
    output_path: str,
    xyz: np.ndarray,
    merged_attrs: dict[str, np.ndarray],
) -> None:
    """Write LAS in slices of ``wrap.config.chunk_size`` (append), no sort VLR."""
    n = len(xyz)
    chunk_size = int(wrap.config.chunk_size)
    if chunk_size <= 0:
        chunk_size = max(n, 1)

    base_header = wrap._create_new_las_filer_header()
    with laspy.open(output_path, mode="w", header=base_header) as _writer:
        pass

    log.info("Writing %d points to %s in chunks of %d", n, output_path, chunk_size)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        new_las_header = wrap._create_new_las_filer_header()
        with laspy.open(output_path, mode="a", header=new_las_header) as writer:
            las_chunk = laspy.LasData(new_las_header)
            las_chunk.x = xyz[start:end, 0]
            las_chunk.y = xyz[start:end, 1]
            las_chunk.z = xyz[start:end, 2]
            for dim, arr in merged_attrs.items():
                if dim in _SKIP_DIMS:
                    continue
                setattr(las_chunk, dim, arr[start:end])
            writer.append_points(las_chunk.points)
            log.debug("Appended points [%d, %d)", start, end)


def _apply_dbscan_object_clusters(
    merged_attrs: dict[str, np.ndarray],
    xyz: np.ndarray,
    eps: float,
    min_points: int,
) -> dict[str, np.ndarray]:
    """Update ``merged_attrs['classification']`` in place for class-5 points."""
    if "classification" not in merged_attrs:
        raise KeyError("merged_attrs must contain 'classification'")

    classification_attr = np.asarray(merged_attrs["classification"])
    if classification_attr.shape[0] != len(xyz):
        raise ValueError("classification length does not match point count")

    orig_dtype = classification_attr.dtype
    classification_attr = classification_attr.copy()
    obj_mask = classification_attr == constants.PCB_CLASSIFICATION_OBJECTS
    n_obj = int(np.sum(obj_mask))
    if n_obj == 0:
        log.info("No points with classification %d (objects); output unchanged.", constants.PCB_CLASSIFICATION_OBJECTS)
        return {
            "object_cluster_id": np.zeros(len(xyz), dtype=np.int32),
            "cluster_ids": np.array([], dtype=np.int32),
        }

    obj_indices = np.where(obj_mask)[0]
    subset = o3d.geometry.PointCloud()
    subset.points = o3d.utility.Vector3dVector(xyz[obj_mask])
    labels = np.asarray(
        subset.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    raw_to_seq: dict[int, int] = {}
    seq = 1
    for raw in sorted(np.unique(labels[labels >= 0])):
        raw_to_seq[int(raw)] = seq
        seq += 1

    n_clustered = int(np.sum(labels >= 0))
    n_noise = int(np.sum(labels < 0))
    log.info(
        "DBSCAN on %d object points: %d clustered, %d noise (-1); %d unique clusters.",
        n_obj,
        n_clustered,
        n_noise,
        len(raw_to_seq),
    )

    start = constants.PCB_CLASSIFICATION_OBJECTS_CLUSTER_START
    max_class = 30
    class_span = max_class - start + 1  # 25 values: 6..30
    object_cluster_id = np.zeros(len(xyz), dtype=np.int32)
    for j, gidx in enumerate(obj_indices):
        lab = int(labels[j])
        if lab < 0:
            continue
        s = raw_to_seq[lab]
        wrapped_class = start + ((s - 1) % class_span)
        classification_attr[gidx] = wrapped_class
        object_cluster_id[gidx] = int(s)

    merged_attrs["classification"] = classification_attr.astype(orig_dtype, copy=False)
    return {
        "object_cluster_id": object_cluster_id,
        "cluster_ids": np.asarray(sorted(raw_to_seq.values()), dtype=np.int32),
    }


def _run_object_vlm_post_processing(
    merged_attrs: dict[str, np.ndarray],
    xyz: np.ndarray,
    object_cluster_id: np.ndarray,
    cluster_ids: np.ndarray,
    output_folder_path: str,
    ollama_model: str = "gemma3:4b",
) -> None:
    """Per-cluster screenshots + VLM classification + YAML merge."""
    if len(cluster_ids) == 0:
        return
    os.makedirs(output_folder_path, exist_ok=True)
    traces_dir = os.path.join(output_folder_path, "vlm_object_traces")
    os.makedirs(traces_dir, exist_ok=True)

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(xyz)
    cls_arr = np.asarray(merged_attrs["classification"])
    npt = len(cls_arr)

    for cid in cluster_ids:
        cid = int(cid)
        try:
            indices = _mask_indices_for_object_vlm_view(cls_arr, object_cluster_id, cid)
            if len(indices) == 0:
                log.warning("Object VLM post-processing: empty view for cluster %s; skipping", cid)
                continue

            view_pcd = pcd_all.select_by_index(indices)
            view_attrs: dict[str, np.ndarray] = {}
            for k, v in merged_attrs.items():
                arr = np.asarray(v)
                if arr.shape[0] == npt:
                    view_attrs[k] = arr[indices]
                else:
                    view_attrs[k] = v

            try:
                _apply_rgb_colors_to_pcd(view_pcd, view_attrs)
            except ValueError as ex:
                log.warning(
                    "Object VLM post-processing: RGB unavailable for cluster %s (%s); skipping.",
                    cid,
                    ex,
                )
                continue

            prefix = f"pp_{cid:05d}"

            front_path, back_path = _take_isometric_screenshots(view_pcd, traces_dir, prefix)
            try:
                record = identify_object_from_views(
                    front_path,
                    back_path,
                    cluster_id=cid,
                    chunk_index=0,
                    ollama_model=ollama_model,
                )
                log.info("VLM: identified %s as %s", cid, record["ai_label"])
                predicted_class = record["classification"]
                cluster_point_mask = object_cluster_id == cid
                if np.any(cluster_point_mask):
                    merged_attrs["classification"][cluster_point_mask] = predicted_class
                merge_identified_objects_record(output_folder_path, record)
            except Exception as ex:
                log.warning(
                    "Object VLM post-processing: VLM/YAML failed for cluster %s: %s",
                    cid,
                    ex,
                    exc_info=True,
                )
        except Exception as ex:
            log.warning(
                "Object VLM post-processing failed for cluster %s: %s",
                cid,
                ex,
                exc_info=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DBSCAN re-segmentation of LAS points classified as objects (5) into classes 6, 7, …"
    )
    parser.add_argument("--input", "-i", required=True, help="Input .las or .laz path")
    parser.add_argument("--output", "-o", required=True, help="Output .las path")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yml",
        help="YAML config path (chunk_size and objects_segmentation defaults)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        metavar="E",
        help="DBSCAN eps (default: objects_segmentation.eps from config)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=None,
        metavar="N",
        help="DBSCAN min_points (default: objects_segmentation.min_points from config)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if not os.path.isfile(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    cfg = config_module.SegmentorConfig.initialize(args.config)
    cfg.input_file = input_path
    cfg.reset_classification = False

    obj_cfg = getattr(cfg, "objects_segmentation", {}) or {}
    eps = float(args.eps if args.eps is not None else obj_cfg.get("eps", 0.05))
    min_points = int(
        args.min_points if args.min_points is not None else obj_cfg.get("min_points", 10)
    )
    if eps <= 0:
        print("eps must be positive.", file=sys.stderr)
        return 1
    if min_points < 2:
        print("min_points must be >= 2.", file=sys.stderr)
        return 1

    wrap = LasPipelineWrapper()
    xyz, merged_attrs = _merge_chunks(wrap)

    try:
        dbscan_meta = _apply_dbscan_object_clusters(merged_attrs, xyz, eps=eps, min_points=min_points)
    except KeyError as e:
        print(f"Missing required attribute: {e}", file=sys.stderr)
        return 1

    result_cfg = getattr(cfg, "result", {}) or {}
    output_folder_path = str(result_cfg.get("output_folder_path", "")).strip()
    if output_folder_path:
        _run_object_vlm_post_processing(
            merged_attrs=merged_attrs,
            xyz=xyz,
            object_cluster_id=np.asarray(dbscan_meta["object_cluster_id"], dtype=np.int32),
            cluster_ids=np.asarray(dbscan_meta["cluster_ids"], dtype=np.int32),
            output_folder_path=output_folder_path,
            ollama_model="gemma3:4b",
        )
    else:
        log.warning(
            "result.output_folder_path is not set in config; skipping VLM screenshots/YAML."
        )

    _write_las(wrap, output_path, xyz, merged_attrs)
    print(f"Wrote {len(xyz):,} points to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
