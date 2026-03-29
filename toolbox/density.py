import numpy as np
from scipy.spatial import ConvexHull

def get_inlier_density(point_array, inlier_indices, plane_model):
    a, b, c, d = plane_model

    # Get the inlier points as a numpy array
    inlier_points = point_array[inlier_indices]

    # --- Project points onto the plane's local 2D coordinate system ---
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Build two orthogonal axes on the plane surface
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project each point onto (u, v) axes → 2D coordinates
    points_2d = np.column_stack([
        inlier_points @ u,
        inlier_points @ v
    ])

    # --- Compute convex hull area ---
    hull = ConvexHull(points_2d)
    area = hull.volume  # in 2D, .volume gives the area

    density = len(inlier_indices) / area
    return density

