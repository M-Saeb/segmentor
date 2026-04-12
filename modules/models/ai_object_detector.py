import base64
import os
import re

import yaml
from ollama import chat

from modules.utils import constants
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

IDENTIFIED_OBJECTS_FILENAME = "3_objects.yml"


def identified_objects_yaml_path(output_folder_path: str) -> str:
    return os.path.join(output_folder_path, IDENTIFIED_OBJECTS_FILENAME)


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
    normalized = str(ai_label).strip().lower()
    if normalized in ["car", "truck", "bus" "motorcycle", "bike", "bicycle"]:
        return constants.PCB_CLASSIFICATION_CAR
    elif normalized in ["empty", "unknown", "structure", "none", "stairs", "vent", "pillar", "ground", "wall", "ceiling"]:
        return constants.PCB_CLASSIFICATION_STRUCTURE
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
        "The scene is for a parking lot area with only the object we want to identify. "
        "Reply with a single JSON object only, no markdown, in this exact form:\n"
        '{"label": "<short English object type name>"}\n'
        "Examples of label: chair, table, car, person, tree, unknown."
        "Some guidelines:\n"
        '1. Some photos could be a piece of a wall, ground, pillars, or no actually identifiable object, name those "empty"\n'
        '2. Only focus on the single object that is at the center of the photo, not the edges\n'
        '3. If you are not sure what the object is, name it "empty"\n'
        '4. The identified object should be standing upright EX: if you see a pole, but it is side ways, name it "empty"\n'
        '5. The colors of the identified object should make sense to its natural colors, EX: a "tree" without brown or green color should not be identified as a tree, instead put it as "unknown" '
        '6. The identified object should make sense in the context of the place, EX: a "candy bar" does not belong in a parking lot, so it should be identified as "unknown"\n'
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

    ai_label = _parse_ai_label_from_response(raw)
    normalized_ai_label = str(ai_label).strip().lower()

    return {
        "chunk_index": int(chunk_index),
        "cluster_id": int(cluster_id),
        "classification": _map_ai_label_to_classification(normalized_ai_label),
        "ai_label": normalized_ai_label,
        "raw_response": raw.strip(),
    }


def _empty_objects_registry() -> dict:
    return {
        "cars": {"total": 0, "items": []},
        "other_objects": {"total": 0, "items": []},
    }


def _bucket_key_for_record(record: dict) -> str:
    if int(record.get("classification", constants.PCB_CLASSIFICATION_OBJECTS)) == constants.PCB_CLASSIFICATION_CAR:
        return "cars"
    return "other_objects"


def _normalize_objects_registry(loaded: dict | None) -> dict:
    """Build cars/other_objects registry from YAML. Migrates legacy ``objects``/``counters`` files."""
    if not isinstance(loaded, dict):
        return _empty_objects_registry()

    cars_block = loaded.get("cars")
    other_block = loaded.get("other_objects")
    if isinstance(cars_block, dict) and isinstance(other_block, dict):
        car_items = list(cars_block.get("items") or [])
        other_items = list(other_block.get("items") or [])
        return {
            "cars": {"total": len(car_items), "items": car_items},
            "other_objects": {"total": len(other_items), "items": other_items},
        }

    legacy = loaded.get("objects")
    if isinstance(legacy, list):
        reg = _empty_objects_registry()
        for item in legacy:
            if not isinstance(item, dict):
                continue
            reg[_bucket_key_for_record(item)]["items"].append(item)
        reg["cars"]["total"] = len(reg["cars"]["items"])
        reg["other_objects"]["total"] = len(reg["other_objects"]["items"])
        return reg

    return _empty_objects_registry()


def append_identified_cluster_to_objects_yaml(output_folder_path: str, record: dict) -> None:
    """
    Append one cluster identification to ``3_objects.yml`` under ``cars`` or ``other_objects``.

    Each bucket has ``total`` (always ``len(items)``) and ``items`` (list of dicts). Items should
    include VLM fields plus ``bounds`` with ``min``/``max`` XYZ lists from the axis-aligned box;
    callers supply bounds. Legacy rows migrated from the old ``objects`` list may omit ``bounds``.
    """
    path = identified_objects_yaml_path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    data = _empty_objects_registry()
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            data = _normalize_objects_registry(loaded)
        except Exception as ex:
            log.warning("append_identified_cluster_to_objects_yaml: could not read %s: %s", path, ex)

    bucket = _bucket_key_for_record(record)
    data[bucket]["items"].append(record)
    data["cars"]["total"] = len(data["cars"]["items"])
    data["other_objects"]["total"] = len(data["other_objects"]["items"])

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    log.info(
        "Updated %s (bucket=%s cars=%d other=%d)",
        path,
        bucket,
        data["cars"]["total"],
        data["other_objects"]["total"],
    )
