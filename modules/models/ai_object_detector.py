import base64
import os
import re
import time

import yaml
from ollama import chat

from modules.models.runtime_report import RuntimeReport
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
    if normalized in ["car", "truck", "bus", "motorcycle", "bike", "bicycle"]:
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
    runtime_report: RuntimeReport | None = None,
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

    t0 = time.perf_counter()
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
    finally:
        if runtime_report is not None:
            runtime_report.add_vlm_response_seconds(time.perf_counter() - t0)

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


def empty_objects_registry() -> dict:
    return {
        "cars": {"total": 0, "items": []},
        "other_objects": {"total": 0, "items": []},
    }


def _bucket_key_for_record(record: dict) -> str:
    if int(record.get("classification", constants.PCB_CLASSIFICATION_OBJECTS)) == constants.PCB_CLASSIFICATION_CAR:
        return "cars"
    return "other_objects"


def _slim_record_for_yaml(record: dict) -> dict:
    return {
        "cluster_id": int(record["cluster_id"]),
        "ai_label": str(record.get("ai_label", "")),
        "bounds": record["bounds"],
    }


def register_identified_cluster(registry: dict, record: dict) -> None:
    """Append ``cluster_id``, ``ai_label``, and ``bounds`` to the appropriate bucket; updates ``total`` counts."""
    bucket = _bucket_key_for_record(record)
    registry[bucket]["items"].append(_slim_record_for_yaml(record))
    registry["cars"]["total"] = len(registry["cars"]["items"])
    registry["other_objects"]["total"] = len(registry["other_objects"]["items"])


def write_identified_objects_yaml(output_folder_path: str, registry: dict) -> None:
    path = identified_objects_yaml_path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    log.info(
        "Wrote %s (cars=%d other=%d)",
        path,
        registry["cars"]["total"],
        registry["other_objects"]["total"],
    )
