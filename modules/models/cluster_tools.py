"""
Single-pass original LAS → per-cluster PointCloud mapping.

Instead of re-opening the original file once per cluster, scan it once and
distribute points to all clusters simultaneously.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import laspy

from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)


def _cluster_bboxes_from_xyz(
    xyz: np.ndarray,
    objects_clusters: np.ndarray,
    cluster_ids: list[int],
) -> dict[int, o3d.geometry.AxisAlignedBoundingBox]:
    bboxes: dict[int, o3d.geometry.AxisAlignedBoundingBox] = {}
    for cid in cluster_ids:
        pts = xyz[objects_clusters == cid]
        if len(pts) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        bboxes[cid] = pcd.get_axis_aligned_bounding_box()
    return bboxes


def build_original_pcds_per_cluster(
    original_path: str,
    xyz: np.ndarray,
    objects_clusters: np.ndarray,
    cluster_ids: list[int],
    chunk_size: int = 10_000_000,
) -> dict[int, o3d.geometry.PointCloud]:
    """
    Read the original LAS file once and return one PointCloud per cluster.

    Points are distributed to clusters via bounding-box membership computed
    from the processed cloud (``xyz`` + ``objects_clusters``).  A point that
    falls inside multiple bounding boxes is added to each matching cluster.

    Parameters
    ----------
    original_path:
        Path to the full-resolution LAS/LAZ file.
    xyz:
        Processed point positions, shape (N, 3).
    objects_clusters:
        Per-point cluster label array, shape (N,).  Labels <= 0 are ignored.
    cluster_ids:
        Ordered list of positive cluster IDs to process.
    chunk_size:
        Points read per laspy chunk (controls peak RAM usage during the scan).

    Returns
    -------
    dict[int, o3d.geometry.PointCloud]
        ``{cluster_id: pcd}`` — colors are normalised to [0, 1] when the
        source file contains RGB channels; otherwise the PCD has no colors.
    """
    if not cluster_ids:
        return {}

    bboxes = _cluster_bboxes_from_xyz(xyz, objects_clusters, cluster_ids)
    if not bboxes:
        return {}

    valid_ids = list(bboxes.keys())
    lo = np.array([np.asarray(bboxes[cid].get_min_bound()) for cid in valid_ids], dtype=np.float64)  # (K, 3)
    hi = np.array([np.asarray(bboxes[cid].get_max_bound()) for cid in valid_ids], dtype=np.float64)  # (K, 3)

    xyz_acc: dict[int, list[np.ndarray]] = {cid: [] for cid in valid_ids}
    rgb_acc: dict[int, list[np.ndarray]] = {cid: [] for cid in valid_ids}

    has_rgb: bool | None = None  # resolved on first chunk

    log.info("Building original PCDs for %d clusters (single pass over %s)", len(valid_ids), original_path)

    with laspy.open(original_path) as f:
        for chunk_pts in f.chunk_iterator(chunk_size):
            chunk_xyz = np.vstack((chunk_pts.x, chunk_pts.y, chunk_pts.z)).T.astype(np.float64)

            if has_rgb is None:
                try:
                    _ = chunk_pts.red
                    has_rgb = True
                except Exception:
                    has_rgb = False

            if has_rgb:
                chunk_rgb = np.vstack((
                    np.asarray(chunk_pts.red, dtype=np.float64),
                    np.asarray(chunk_pts.green, dtype=np.float64),
                    np.asarray(chunk_pts.blue, dtype=np.float64),
                )).T  # (N, 3)
            else:
                chunk_rgb = None

            # Vectorised point-in-box test: one cluster at a time to keep memory flat
            for i, cid in enumerate(valid_ids):
                in_box = ((chunk_xyz >= lo[i]) & (chunk_xyz <= hi[i])).all(axis=1)
                if not np.any(in_box):
                    continue
                xyz_acc[cid].append(chunk_xyz[in_box])
                if chunk_rgb is not None:
                    rgb_acc[cid].append(chunk_rgb[in_box])

    log.debug("Finished scanning original file; building PCDs for %d clusters.", len(valid_ids))


    result: dict[int, o3d.geometry.PointCloud] = {}
    for cid in valid_ids:
        log.debug("Building PCD for cluster %d with %d accumulated points.", cid, sum(len(arr) for arr in xyz_acc[cid]))
        if not xyz_acc[cid]:
            log.debug("Cluster %s had no matching points in the original file.", cid)
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(xyz_acc[cid]))

        if rgb_acc[cid]:
            colors = np.vstack(rgb_acc[cid])
            max_val = colors.max()
            if max_val > 0:
                colors = colors / max_val
            pcd.colors = o3d.utility.Vector3dVector(colors)

        result[cid] = pcd

    log.info("Built original PCDs for %d / %d clusters.", len(result), len(valid_ids))
    return result
