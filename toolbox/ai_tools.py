import base64
import os
import re

import open3d as o3d
import numpy as np
from ollama import chat

from modules.utils import constants
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

IDENTIFIED_OBJECTS_FILENAME = "identified_objects.yml"


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _parse_ai_label_from_response(text: str) -> str:
    if not text or not str(text).strip():
        return "unknown"
    t = str(text).strip()
    # Prefer JSON-like "label": "foo"
    m = re.search(r'"label"\s*:\s*"([^"]*)"', t, re.I)
    if m:
        return m.group(1).strip() or "unknown"
    m = re.search(r'"ai_label"\s*:\s*"([^"]*)"', t, re.I)
    if m:
        return m.group(1).strip() or "unknown"
    # Single line: use first line, strip punctuation
    line = t.split("\n")[0].strip()
    line = re.sub(r'^[\s\[\]"\']+|[\s\[\]"\']+$', "", line)
    if len(line) > 80:
        line = line[:80] + "..."
    return line or "unknown"

def _map_ai_label_to_classification(ai_label: str) -> int:
    if ai_label == "car":
        return constants.PCB_CLASSIFICATION_CAR
    elif ai_label == "empty" or ai_label == "unknown":
        return constants.PCB_CLASSIFICATION_INIT
    else:
        return constants.PCB_CLASSIFICATION_OTHER_OBJECTS

def identify_object_from_views(
    front_path: str,
    back_path: str,
    *,
    cluster_id: int,
    chunk_index: int,
    ollama_model: str = "gemma3:4b",
) -> dict:
    """
    Send front/back isometric PNGs to Ollama and return a record for the YAML registry.
    """
    prompt = (
        "You see two isometric views (front and back) of a LiDAR point cloud region with a white background. "
        "All the points are in gray colors, except for a plane that is identified using RANSAC, that will be in red color."
        "Your job is to identify if the plane is: GROUND, WALL, CEILING, or OTHER. "
        "OTHER can refer to other objects like chairs, tables, cars, people, trees, etc., "
        "Planes that only have a section of a clear plane, like only section of a ground, should still be classified as GROUND or WALL or CEILING."
        "Your response should be a single word only: GROUND, WALL, CEILING, or OTHER."
    )

    b64_front = _encode_image_b64(front_path)
    b64_back = _encode_image_b64(back_path)

    try:
        response = chat(
            model=ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_front, b64_back],
                }
            ],
        )
    except Exception as ex:
        log.warning("identify_object_from_views: Ollama chat failed: %s", ex)
        raise

    raw = ""
    if isinstance(response, dict):
        raw = (response.get("message") or {}).get("content") or ""
    else:
        msg = getattr(response, "message", None)
        raw = getattr(msg, "content", "") if msg else str(response)

    return raw.strip()

    ai_label = _parse_ai_label_from_response(raw)

    return {
        "chunk_index": int(chunk_index),
        "cluster_id": int(cluster_id),
        "classification": _map_ai_label_to_classification(ai_label),
        "ai_label": ai_label,
        "raw_response": raw.strip(),
    }



def take_isometric_screenshots(pcd, dir_name, file_prefix=""):
    """
    Save front and back isometric PNGs of the point cloud (RGB via pcd.colors).

    If file_prefix is empty, writes:
        {dir_name}/front_isometric.png, {dir_name}/back_isometric.png
    Otherwise:
        {dir_name}/{file_prefix}_front_isometric.png, etc.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Scale point cloud 3x around its center to make it visually larger.
    scaled_pcd = o3d.geometry.PointCloud(pcd)
    scaled_pcd.scale(10.0, center=scaled_pcd.get_center())

    width, height = 640, 480
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Match light blue background used in sandbox.draw_geometries_with_light_blue_background.
    renderer.scene.set_background(np.array([0.8, 0.9, 1.0, 1.0], dtype=float))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("cloud", scaled_pcd, mat)

    center = np.asarray(scaled_pcd.get_center(), dtype=float)
    bounds = scaled_pcd.get_axis_aligned_bounding_box()
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
        log.debug("Saved %s/%s", dir_name, fname)

def take_screenshot(pcd, dir_name):
    """
    Save two screenshots of the point cloud, approximating
    CloudCompare's "front isometric" and "back isometric" views.

    Images are saved as:
        {dir_name}/front_isometric.png
        {dir_name}/back_isometric.png
    """
    take_isometric_screenshots(pcd, dir_name, file_prefix="")



def get_plane_id_from_ai(pcd_inliers, remain_pcd, debug=False):
    remain_pcd_color = [0.5, 0.5, 0.5] # gray
    remain_pcd.paint_uniform_color(remain_pcd_color)
    pcd = pcd_inliers # + remain_pcd
    take_screenshot(pcd, "temp")
    response = identify_object_from_views(
        "temp/front_isometric.png",
        "temp/back_isometric.png",
        cluster_id=0,
        chunk_index=0,
        ollama_model="gemma3:4b",
    )
    print("response: %s", response)
    return response