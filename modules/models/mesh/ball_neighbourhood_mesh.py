"""
lidar_mesh.py
=============
Mesh reconstruction from LiDAR point clouds using a ball-neighborhood
growing algorithm, designed for complex building wall surfaces.

Algorithm (as described):
  1. Start from a point (lowest Z, i.e. ground-anchor)
  2. Cast a sphere of radius R around it
  3. Collect all points inside the sphere → neighbourhood group
     (points already consumed as face-centres are skipped,
      but shared vertices ARE reused across faces)
  4. Compute the best-fit plane of the neighbourhood, project onto it,
     find the convex boundary → these boundary points become face vertices
  5. Triangulate the face (fan triangulation from centroid)
  6. Push every boundary vertex onto the queue and repeat from step 2

Dependencies:
    pip install open3d numpy scipy
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree, ConvexHull
from collections import deque
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────
#  Core: single neighbourhood mesh expansion
# ─────────────────────────────────────────────────────────────

def _fit_plane(pts: np.ndarray):
    """
    Fit a plane to pts via PCA.
    Returns (normal, centroid) where normal is the minor eigenvector.
    """
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]          # smallest singular value → normal direction
    return normal, centroid


def _project_to_plane(pts: np.ndarray, normal: np.ndarray, centroid: np.ndarray):
    """
    Project 3-D points onto a local 2-D coordinate system on the plane.
    Returns (2-D coords, basis vectors u/v).
    """
    # Build orthonormal basis on the plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    shifted = pts - centroid
    coords_2d = np.column_stack([shifted @ u, shifted @ v])
    return coords_2d, u, v


def _boundary_indices_2d(coords_2d: np.ndarray) -> Optional[np.ndarray]:
    """
    Return indices of the convex-hull boundary points in the 2-D projection.
    Returns None if fewer than 3 non-collinear points exist.
    """
    if len(coords_2d) < 3:
        return None
    try:
        hull = ConvexHull(coords_2d)
        return hull.vertices          # indices into coords_2d (and thus into pts)
    except Exception:
        return None


def _fan_triangulate(vertex_indices: np.ndarray) -> list:
    """
    Fan triangulation: connect first vertex to every consecutive pair.
    Works well for convex polygons (which our hull always is).
    """
    triangles = []
    v0 = vertex_indices[0]
    for i in range(1, len(vertex_indices) - 1):
        triangles.append([v0, vertex_indices[i], vertex_indices[i + 1]])
    return triangles


# ─────────────────────────────────────────────────────────────
#  Main reconstruction function
# ─────────────────────────────────────────────────────────────

def reconstruct_mesh(
    pcd: o3d.geometry.PointCloud,
    ball_radius: float,
    min_neighbours: int = 4,
    max_faces: int = 50_000,
    start_index: Optional[int] = None,
    adaptive_radius: bool = True,
    adaptive_factor: float = 1.5,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a triangle mesh from a LiDAR point cloud using the
    ball-neighbourhood growing algorithm.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Input point cloud. Does NOT need normals pre-computed.
    ball_radius : float
        Radius of the neighbourhood sphere (same units as your point cloud).
        Rule of thumb for LiDAR building scans: 2–5× the average point
        spacing. Use open3d's estimate_point_spacing() helper below.
    min_neighbours : int
        Minimum points inside the ball to attempt face creation.
        Smaller → denser mesh; larger → smoother but with more holes.
    max_faces : int
        Safety cap to prevent runaway growth on very large clouds.
    start_index : int or None
        Index of the seed point. None → lowest-Z point (ground anchor).
    adaptive_radius : bool
        If True, expand ball_radius by adaptive_factor when a neighbourhood
        is too sparse (< min_neighbours). Helps bridge small gaps in the
        point cloud (common in LiDAR occlusions).
    adaptive_factor : float
        Multiplier applied when adaptive radius kicks in.

    Returns
    -------
    open3d.geometry.TriangleMesh
        Reconstructed mesh with merged duplicate vertices.
    """

    pts = np.asarray(pcd.points)
    n_pts = len(pts)
    if n_pts < 4:
        raise ValueError("Need at least 4 points to build a mesh.")

    # ── KD-tree for fast radius queries ──────────────────────
    tree = KDTree(pts)

    # ── Seed point ────────────────────────────────────────────
    if start_index is None:
        start_index = int(np.argmin(pts[:, 2]))   # lowest Z = ground anchor

    # ── Tracking state ────────────────────────────────────────
    # visited_as_centre: points already used as a face-centre seed
    #   (they won't re-seed a new face, but CAN appear as shared vertices)
    visited_as_centre = np.zeros(n_pts, dtype=bool)

    all_vertices = pts.tolist()          # global vertex list (index = point index)
    all_triangles: list[list[int]] = []

    queue: deque[int] = deque([start_index])
    visited_as_centre[start_index] = True

    face_count = 0

    # ── Main loop ─────────────────────────────────────────────
    while queue and face_count < max_faces:

        centre_idx = queue.popleft()
        centre_pt  = pts[centre_idx]
        r          = ball_radius

        # ── Step 2 & 3: ball query ────────────────────────────
        neighbour_idxs = tree.query_ball_point(centre_pt, r)

        # Adaptive radius: expand if too sparse
        if adaptive_radius and len(neighbour_idxs) < min_neighbours:
            neighbour_idxs = tree.query_ball_point(centre_pt, r * adaptive_factor)

        # Still too few → skip this centre
        if len(neighbour_idxs) < min_neighbours:
            continue

        neighbour_pts = pts[neighbour_idxs]

        # ── Step 4a: fit plane ────────────────────────────────
        if len(neighbour_pts) < 3:
            continue
        normal, centroid = _fit_plane(neighbour_pts)

        # ── Step 4b: project to 2-D ───────────────────────────
        coords_2d, _, _ = _project_to_plane(neighbour_pts, normal, centroid)

        # ── Step 4c: convex hull boundary ─────────────────────
        boundary_local = _boundary_indices_2d(coords_2d)
        if boundary_local is None:
            continue

        # Map local hull indices → global point indices
        boundary_global = [neighbour_idxs[i] for i in boundary_local]

        if len(boundary_global) < 3:
            continue

        # ── Step 5: fan-triangulate ───────────────────────────
        new_tris = _fan_triangulate(np.array(boundary_global))
        all_triangles.extend(new_tris)
        face_count += len(new_tris)

        # ── Step 6: push boundary vertices onto queue ─────────
        for vi in boundary_global:
            if not visited_as_centre[vi]:
                visited_as_centre[vi] = True
                queue.append(vi)

    if not all_triangles:
        raise RuntimeError(
            "No faces were created. Try increasing ball_radius or "
            "decreasing min_neighbours."
        )

    if face_count >= max_faces:
        print(f"Face count is greater than max_faces: {face_count} >= {max_faces}")

    # ── Build Open3D mesh ─────────────────────────────────────
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(np.array(all_vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles, dtype=np.int32))

    # Merge duplicate vertices (shared vertices across faces)
    mesh.merge_close_vertices(1e-6)

    # Remove degenerate triangles (zero-area faces from collinear points)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Recompute normals for correct shading
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh


# ─────────────────────────────────────────────────────────────
#  Utility: estimate a good ball_radius from the cloud itself
# ─────────────────────────────────────────────────────────────

def estimate_ball_radius(pcd: o3d.geometry.PointCloud, k: int = 10, factor: float = 3.0) -> float:
    """
    Estimate a sensible ball_radius as factor × mean k-NN distance.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
    k : int
        Number of nearest neighbours used for spacing estimate.
    factor : float
        Multiplier. 2.5–4.0 works well for LiDAR building walls.

    Returns
    -------
    float : recommended ball_radius
    """
    pts = np.asarray(pcd.points)
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)   # k+1 because first hit is self
    mean_spacing = dists[:, 1:].mean()
    return float(mean_spacing * factor)

