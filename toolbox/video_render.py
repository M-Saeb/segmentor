import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d


def _image_to_uint8_rgb(image: o3d.geometry.Image) -> np.ndarray:
    """Convert an Open3D image to uint8 RGB."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def render_rotating_isometric_video(
    pcd: o3d.geometry.PointCloud,
    output_path: str = "temp/video.mp4",
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_sec: float = 6.0,
    elev_deg: float = 35.0,
    start_azimuth_deg: float = 225.0,
) -> str:
    """
    Render a rotating isometric video from an Open3D point cloud.

    The camera rotates 360 degrees around the cloud center while preserving an
    isometric-like elevation.
    """
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.array([0.8, 0.9, 1.0, 1.0], dtype=float))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 8.0
    renderer.scene.add_geometry("cloud", pcd, mat)

    center = np.asarray(pcd.get_center(), dtype=float)
    bounds = pcd.get_axis_aligned_bounding_box()
    diag = float((bounds.get_max_bound() - bounds.get_min_bound()).max())
    radius = max(1e-6, diag * 1.0)
    up = [0.0, 0.0, 1.0]

    elev = np.deg2rad(elev_deg)
    z_off = radius * np.sin(elev)
    horiz_r = radius * np.cos(elev)

    frame_count = max(1, int(round(fps * duration_sec)))
    frames = []
    for i in range(frame_count):
        t = i / frame_count
        azimuth_deg = start_azimuth_deg + 360.0 * t
        theta = np.deg2rad(azimuth_deg)
        x_off = horiz_r * np.cos(theta)
        y_off = horiz_r * np.sin(theta)
        eye = center + np.array([x_off, y_off, z_off], dtype=float)

        renderer.setup_camera(60.0, center, eye, up)
        frame = renderer.render_to_image()
        frames.append(_image_to_uint8_rgb(frame))


        imageio.mimwrite(
            str(output_file),
            frames,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    return str(output_file)


if __name__ == "__main__":
    # Demo usage:
    # python toolbox/video_render.py
    # Uses /tmp/las/data.las and writes temp/video.mp4
    input_las = "/tmp/las/data.las"
    if not os.path.isfile(input_las):
        raise FileNotFoundError(f"Demo input file not found: {input_las}")

    import laspy

    las = laspy.read(input_las)
    pcd_demo = o3d.geometry.PointCloud()
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    pcd_demo.points = o3d.utility.Vector3dVector(points)

    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.float64)
        max_val = float(np.max(rgb)) if rgb.size > 0 else 0.0
        if max_val > 0:
            rgb /= max_val
            pcd_demo.colors = o3d.utility.Vector3dVector(rgb)

    out = render_rotating_isometric_video(pcd_demo, output_path="temp/video.mp4")
    print(f"Saved video to: {out}")
