import base64
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

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
        "You see two isometric views (front and back) of a LiDAR point cloud region with a light blue background. "
        "The points are in their original colors, with the objects standing straight up. "
        "Your job is to identify the object and respond with a single word: CAR, OTHER, or NONE."
        "CAR refer to cars, trucks, buses, etc., "
        "OTHER refer to identifiable objects like chairs, tables, people, trees, etc., "
        "NONE refer to none identifiable object, or structures like walls, ground, pillars, ceiling, stairs, and vents."
        "Some specific guidelines:"
        "- If you see a bunch of random point, return NONE"
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

    Rendering runs in a subprocess so each call gets a clean Open3D/Filament/EGL
    lifecycle (avoids segfaults from repeated OffscreenRenderer in the parent).

    If file_prefix is empty, writes:
        {dir_name}/front_isometric.png, {dir_name}/back_isometric.png
    Otherwise:
        {dir_name}/{file_prefix}_front_isometric.png, etc.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    worker = Path(__file__).resolve().parent / "isometric_screenshots_worker.py"
    fd, ply_tmp = tempfile.mkstemp(suffix=".ply", prefix="o3d_iso_")
    os.close(fd)
    try:
        if not o3d.io.write_point_cloud(ply_tmp, pcd):
            raise RuntimeError("Failed to write temporary PLY for isometric screenshots.")
        subprocess.run(
            [
                sys.executable,
                str(worker),
                ply_tmp,
                os.path.abspath(dir_name),
                file_prefix,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as ex:
        log.warning("take_isometric_screenshots: subprocess failed: %s", ex)
        raise
    finally:
        try:
            os.unlink(ply_tmp)
        except OSError:
            pass

def take_screenshot(pcd, dir_name):
    """
    Save two screenshots of the point cloud, approximating
    CloudCompare's "front isometric" and "back isometric" views.

    Images are saved as:
        {dir_name}/front_isometric.png
        {dir_name}/back_isometric.png
    """
    take_isometric_screenshots(pcd, dir_name, file_prefix="")



def get_plane_id_from_ai(pcd_inliers, debug=False):
    take_screenshot(pcd_inliers, "temp")
    response = identify_object_from_views(
        "temp/front_isometric.png",
        "temp/back_isometric.png",
        cluster_id=0,
        chunk_index=0,
        ollama_model="gemma3:4b",
    )
    print("====== response", response)
    return response