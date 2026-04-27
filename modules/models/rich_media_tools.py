import os
import open3d as o3d
import numpy as np

from modules.utils import log_utils

log = log_utils.get_dual_logger(__name__)

def _create_views_dict(number_of_views: int):
    start_azimuth_deg = 225.0
    azimuth_step_deg = 360 / number_of_views
    return {
        f"{i+1:02d}": (start_azimuth_deg + i * azimuth_step_deg) % 360
        for i in range(number_of_views)
    }

def take_rich_media_screenshots(pcd: o3d.geometry.PointCloud, dir_name: str, number_of_views: int):
    """
    Save isometric PNGs of the point cloud (RGB via pcd.colors).
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    width, height = 640, 480
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    # Light blue clear color (RGBA, linear)
    renderer.scene.set_background(np.array([0.8, 0.9, 1.0, 1.0], dtype=float))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("cloud", pcd, mat)

    center = np.asarray(pcd.get_center(), dtype=float)
    bounds = pcd.get_axis_aligned_bounding_box()
    diag = float((bounds.get_max_bound() - bounds.get_min_bound()).max())

    radius = max(1e-6, diag * 1.0)
    up = [0.0, 0.0, 1.0]

    elev_deg = 35.0
    elev = np.deg2rad(elev_deg)
    z_off = radius * np.sin(elev)
    horiz_r = radius * np.cos(elev)

    views = _create_views_dict(number_of_views)
    paths = []
    for name, az_deg in views.items():
        theta = np.deg2rad(az_deg)
        x_off = horiz_r * np.cos(theta)
        y_off = horiz_r * np.sin(theta)
        eye = center + np.array([x_off, y_off, z_off], dtype=float)

        renderer.setup_camera(60.0, center, eye, up)
        img = renderer.render_to_image()
        path = os.path.join(dir_name, f"{name}.png")
        o3d.io.write_image(path, img)
        log.debug("Saved %s/%s", dir_name, path)
        paths.append(path)
    return paths

