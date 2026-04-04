import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d

def _build_plane_hull_visualization(inlier_points, hull_indices):
    """Create Open3D geometries for inlier points and convex hull boundary."""

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(inlier_points)
    point_cloud.paint_uniform_color([0.2, 0.6, 1.0])  # blue points

    hull_points = inlier_points[hull_indices]
    hull_line_points = np.vstack([hull_points, hull_points[0]])  # close polygon loop

    lines = [[i, i + 1] for i in range(len(hull_line_points) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(hull_line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1.0, 0.2, 0.2])  # red boundary

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    return [point_cloud, line_set, frame]

def get_inlier_density(point_array, inlier_indices, plane_model, debug=False):
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

    if debug:
        print("==== visualizing ===")
        print("density:", density)
        print("area:", area)
        print("inlier_indices:", len(inlier_indices))
        geometries = _build_plane_hull_visualization(inlier_points, hull.vertices)
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Inlier plane area: {area:.4f}",
            point_show_normal=False,
        )

    return density

