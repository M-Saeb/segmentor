import base64
import os
import re
import time
import json
import re
from collections.abc import Sequence

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
    elif normalized in ["empty", "unknown", "structure", "none", "stairs", "vent", "road", "pillar", "ground", "wall", "ceiling", "concrete", "parking lot", "fence"]:
        return constants.PCB_CLASSIFICATION_STRUCTURE
    else:
        return constants.PCB_CLASSIFICATION_OTHER_OBJECTS

def _convert_to_json_string(raw_response: str) -> dict:
    """
    Extracts and loads the first JSON block for label/description using regex from the response.
    """

    # Match the first JSON-like object in the string
    match = re.search(
        r"\{[\s\S]*?\"label\"\s*:\s*\"[^\"]*\"[\s\S]*?\"description\"\s*:\s*\"[^\"]*\"[\s\S]*?\}",
        str(raw_response)
    )
    json_block = None
    if match:
        json_block = match.group(0)
    else:
        # fallback: match any JSON block
        match = re.search(r"\{[\s\S]*?\}", str(raw_response))
        if match:
            json_block = match.group(0)
        else:
            json_block = str(raw_response).strip()

    try:
        result = json.loads(json_block)
    except Exception as ex:
        raise ValueError(f"raw_response does not contain a valid JSON-like block: {raw_response}") from ex

    if "label" not in result:
        log.warning("raw_response is missing label: %s", raw_response)
        result["label"] = "unknown"
    if "description" not in result:
        log.warning("raw_response is missing description: %s", raw_response)
        result["description"] = ""

    return result

def identify_object_from_views(
    image_paths: Sequence[str],
    *,
    cluster_id: int,
    chunk_index: int,
    ollama_model: str = "gemma3:4b",
    runtime_report: RuntimeReport | None = None,
) -> dict:
    """
    Send multiple isometric PNGs (rotating views) to Ollama and return a record for the YAML registry.
    """
    paths = list(image_paths)
    if not paths:
        raise ValueError("identify_object_from_views requires at least one image path.")

    prompt = (
        "You see several isometric views around a LiDAR point cloud region (same scene from different angles) "
        "with a light background. "
        "The scene is for a parking lot area with only the object we want to identify. "
        "Reply with a single JSON object only, no markdown, in this exact form:\n"
        '{"label": "<short English object type name>", "description": "<short English description of max 30 words of what you see in the photo>"}\n'
        'Examples are: \n'
        '- {label: "chair", description: "a black chair that fell over"}\n'
        '- {label: "table", description: "a brown table with some dishes on it"}\n'
        "Some guidelines:\n"
        '1. Some photos could be a piece of a wall, ground, pillars, or no actually identifiable object, name those "empty"\n'
        '2. Some photos could be a cloud of random points that are not part of an object, name those "empty"\n'
        '3. Only focus on the single object that is at the center of the photo, not the edges\n'
        '4. If you are not sure what the object is, name it "empty"\n'
        '5. The identified object should be standing upright EX: if you see a pole, but it is side ways, name it "empty"\n'
        '6. The colors of the identified object should make sense to its natural colors, EX: a "tree" without brown or green color should not be identified as a tree, instead put it as "unknown" '
        '7. The identified object should make sense in the context of the place, EX: a "candy bar" does not belong in a parking lot, so it should be identified as "unknown"\n'
    )

    images_b64 = [_encode_image_b64(p) for p in paths]

    t0 = time.perf_counter()
    try:
        response = chat(
            model=ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
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

    json_dict = _convert_to_json_string(raw)
    ai_label = json_dict["label"]
    description = json_dict["description"]
    normalized_ai_label = str(ai_label).strip().lower()

    return {
        "chunk_index": int(chunk_index),
        "cluster_id": int(cluster_id),
        "classification": _map_ai_label_to_classification(normalized_ai_label),
        "ai_label": normalized_ai_label,
        "description": description,
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
