"""Map downsampled inlier selections back onto the full-resolution point cloud."""

from __future__ import annotations
import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from toolbox.read_las_file import read_whole_las_file


def _points_inside_axis_aligned_box(
    xyz: np.ndarray, bbox: o3d.geometry.AxisAlignedBoundingBox
) -> np.ndarray:
    """Boolean mask (N,) for points inside the box; bounds are inclusive (Open3D convention)."""
    lo = np.asarray(bbox.get_min_bound(), dtype=np.float64)
    hi = np.asarray(bbox.get_max_bound(), dtype=np.float64)
    return np.logical_and((xyz >= lo).all(axis=1), (xyz <= hi).all(axis=1))


def get_original_pcd(
    original_path: str,
    bounding_box: o3d.geometry.AxisAlignedBoundingBox | None = None,
) -> o3d.geometry.PointCloud:
    """
    Load the original point cloud from a LAS/LAZ file.

    When the file includes ``red`` / ``green`` / ``blue``, colors are attached and normalized
    to ``[0, 1]`` using the global maximum across the file — same rule as
    ``restore_inlier_rgb_from_las()`` in ``sandbox.py``.

    Parameters
    ----------
    original_path
        Path to the LAS/LAZ file.
    bounding_box
        If given (e.g. from :func:`inlier_bound_box`), only points inside this axis-aligned
        box are kept.
    """

    all_pcd = o3d.geometry.PointCloud()
    with laspy.open(original_path) as f:
        for index, points in enumerate(f.chunk_iterator(10_000_000)):
            pcd = o3d.geometry.PointCloud()
            xyz = np.vstack((points.x, points.y, points.z)).T
            chunk_mask: np.ndarray | None = None
            chunk_mask = _points_inside_axis_aligned_box(xyz, bounding_box)
            if not np.any(chunk_mask):
                continue
            xyz = xyz[chunk_mask]

            pcd.points = o3d.utility.Vector3dVector(xyz)

            red = np.asarray(points.red, dtype=float)
            green = np.asarray(points.green, dtype=float)
            blue = np.asarray(points.blue, dtype=float)
            rgb = np.vstack((red, green, blue)).T.astype(np.float64)
            if chunk_mask is not None:
                rgb = rgb[chunk_mask]
            max_val = np.max(rgb)
            if max_val > 0:
                rgb /= max_val  # normalize to [0, 1] like display_pcd_in_rgb / PCDPreviewer
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            all_pcd += pcd
    return all_pcd


def _inlier_points_to_numpy(inlier_pcd: o3d.geometry.PointCloud | np.ndarray) -> np.ndarray:
    if isinstance(inlier_pcd, o3d.geometry.PointCloud):
        pts = np.asarray(inlier_pcd.points, dtype=np.float64)
    else:
        pts = np.asarray(inlier_pcd, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("inlier_pcd must be an Nx3 array or Open3D PointCloud with 3D points.")
    return pts


def _estimate_voxel_size_from_inliers(inliers: np.ndarray) -> float:
    """Rough voxel spacing from nearest-neighbor distances among inlier centroids."""
    n = inliers.shape[0]
    if n < 2:
        return 0.1
    tree = cKDTree(inliers)
    k = min(2, n)
    sample_idx = np.linspace(0, n - 1, num=min(2000, n), dtype=int)
    dists, _ = tree.query(inliers[sample_idx], k=k)
    if k == 1:
        return float(np.median(dists))
    return float(np.median(dists[:, 1]))


def inlier_bound_box(
    inlier_pcd: o3d.geometry.PointCloud | np.ndarray,
) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Axis-aligned bounding box of the inlier points.

    Parameters
    ----------
    inlier_pcd
        Inlier points as an Open3D PointCloud or ``(N, 3)`` array.

    Returns
    -------
    open3d.geometry.AxisAlignedBoundingBox
        Minimal axis-aligned box containing all inliers.

    Raises
    ------
    ValueError
        If ``inlier_pcd`` is empty or not valid 3D points.
    """
    inliers = _inlier_points_to_numpy(inlier_pcd)
    if inliers.shape[0] == 0:
        raise ValueError("Cannot compute bounding box for empty inliers.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inliers)
    return pcd.get_axis_aligned_bounding_box()

def inlier_mutliple_bound_boxes(
    inlier_pcd: o3d.geometry.PointCloud | np.ndarray,
) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Axis-aligned bounding box of the inlier points.

    Parameters
    ----------
    inlier_pcd
        Inlier points as an Open3D PointCloud or ``(N, 3)`` array.

    Returns
    -------
    open3d.geometry.AxisAlignedBoundingBox
        Minimal axis-aligned box containing all inliers.

    Raises
    ------
    ValueError
        If ``inlier_pcd`` is empty or not valid 3D points.
    """
    bounding_boxes = []
    pcds = []
    labels = np.asarray(inlier_pcd.cluster_dbscan(eps=0.3, min_points=5))
    labale_count = len(list(labels))
    print("labels", labale_count)
    print("for", len(inlier_pcd.points), "we found", labale_count, "labels")
    if labels.size == 0:
        return []
    for label in labels:
        pcd = inlier_pcd.select_by_index(np.where(labels == label)[0])
        color = np.random.rand(3)
        pcd.paint_uniform_color(color)
        pcds.append(pcd)
        bounding_box = pcd.get_axis_aligned_bounding_box()
        bounding_boxes.append(bounding_box)

    o3d.visualization.draw_geometries(pcds)
    return bounding_boxes
    

def restore_original_points_in_inliers(
    inlier_pcd: o3d.geometry.PointCloud | np.ndarray,
    original_path: str,
) -> o3d.geometry.PointCloud:
    """
    Return full-resolution points that lie in the same voxel regions as the downsampled inliers.

    After voxel downsampling, each remaining point is a per-voxel centroid. Any original point
    whose nearest inlier centroid is within half the voxel space diagonal is kept (same rule
    as assigning a point to that voxel's representative).

    Parameters
    ----------
    inlier_pcd
        Downsampled inlier points (Open3D PointCloud or ``(N, 3)`` array).
    original_path
        Path to the full-resolution PCD (or any format Open3D can read).
    voxel_size
        Voxel edge length used when downsampling. If omitted, a value is estimated from
        nearest-neighbor spacing among ``inlier_pcd`` (less reliable if inliers are sparse).

    Returns
    -------
    open3d.geometry.PointCloud
        Subset of the original cloud (points, and colors/normals if present) matching the inliers.
    """
    bounding_box = inlier_bound_box(inlier_pcd)
    original_pcd = get_original_pcd(original_path, bounding_box)
    return original_pcd
