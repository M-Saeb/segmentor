"""Map downsampled inlier selections back onto the full-resolution point cloud."""

from __future__ import annotations
import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from toolbox.read_las_file import read_whole_las_file

def get_original_pcd(original_path: str) -> o3d.geometry.PointCloud:
    all_pcd = o3d.geometry.PointCloud()
    with laspy.open(original_path) as f:
        for points in f.chunk_iterator(1_000_000):
            pcd = o3d.geometry.PointCloud()
            points_attributes = {}
            xyz = np.vstack((points.x, points.y, points.z)).T
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Store all attributes except x, y, z
            for dim in points.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    if dim not in points_attributes:
                        points_attributes[dim] = []
                    points_attributes[dim].append(getattr(points, dim))

            for dim, arrays in points_attributes.items():
                points_attributes[dim] = np.concatenate(arrays)

            # Initialize classification attribute for all points
            num_points = len(pcd.points)
            
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


def restore_original_points_in_inliers(
    inlier_pcd: o3d.geometry.PointCloud | np.ndarray,
    original_path: str,
    voxel_size: float | None = None,
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
    inliers = _inlier_points_to_numpy(inlier_pcd)
    full = get_original_pcd(original_path)
    original = np.asarray(full.points, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("Original point cloud must have 3D points.")

    if inliers.shape[0] == 0:
        print("No inliers found")
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return out

    if original.shape[0] == 0:
        return o3d.geometry.PointCloud()

    vs = voxel_size if voxel_size is not None else _estimate_voxel_size_from_inliers(inliers)
    if vs <= 0:
        raise ValueError("voxel_size must be positive.")

    # Max distance from voxel centroid to any corner of the voxel cube (Open3D leaf size).
    capture_radius = vs * np.sqrt(3.0) / 2.0

    tree = cKDTree(inliers)
    dist, _ = tree.query(original, k=1)
    mask = dist <= capture_radius

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(original[mask])
    if full.has_colors():
        print("has colors")
        c = np.asarray(full.colors)
        if len(c) == len(original):
            out.colors = o3d.utility.Vector3dVector(c[mask])
    if full.has_normals():
        print("has normals")
        n = np.asarray(full.normals)
        if len(n) == len(original):
            out.normals = o3d.utility.Vector3dVector(n[mask])

    print("in points", len(inlier_pcd.points))
    print("out points", len(out.points))
    return out
