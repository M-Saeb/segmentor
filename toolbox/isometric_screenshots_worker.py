"""CLI for subprocess isometric PNG rendering (isolates Open3D OffscreenRenderer)."""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import open3d as o3d

def _render_isometric_impl(pcd, dir_name: str, file_prefix: str) -> None:
    """
    Save front and back isometric PNGs using OffscreenRenderer.

    Intended for use in a dedicated subprocess (see :func:`take_isometric_screenshots`)
    so Filament/EGL is not created repeatedly in the parent process.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    width, height = 640, 480
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

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

    views = {
        "front_isometric": 225.0,
        "back_isometric": 45.0,
    }

    for name, az_deg in views.items():
        theta = np.deg2rad(az_deg)
        x_off = horiz_r * np.cos(theta)
        y_off = horiz_r * np.sin(theta)
        eye = center + np.array([x_off, y_off, z_off], dtype=float)

        renderer.setup_camera(60.0, center, eye, up)
        img = renderer.render_to_image()
        if file_prefix:
            fname = f"{file_prefix}_{name}.png"
        else:
            fname = f"{name}.png"
        o3d.io.write_image(os.path.join(dir_name, fname), img)
        print(f"Saved {dir_name}/{fname}")


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "usage: isometric_screenshots_worker.py <point_cloud.ply> <out_dir> <file_prefix>",
            file=sys.stderr,
        )
        sys.exit(2)
    ply_path, dir_name, file_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        sys.exit(1)
    _render_isometric_impl(pcd, dir_name, file_prefix)


if __name__ == "__main__":
    main()
