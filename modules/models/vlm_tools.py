import base64
import os
import re
import json
import time
from collections.abc import Sequence

import yaml

from modules.models.vlm_providers import VLMFatalError, get_vlm_provider
from modules.utils import constants
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

CLUSTER_DATA_FILENAME = "data.yml"


def cluster_data_yaml_path(trace_dir: str) -> str:
    return os.path.join(trace_dir, CLUSTER_DATA_FILENAME)


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _normalize_vlm_class(raw_class) -> str:
    normalized = str(raw_class).strip().upper()
    if normalized == "UKNOWN":
        return constants.VLM_CLASS_UNKNOWN
    if normalized not in constants.VLM_CLASS_TO_CLASSIFICATION:
        log.warning("Unexpected VLM class %r; using %s", raw_class, constants.VLM_CLASS_UNKNOWN)
        return constants.VLM_CLASS_UNKNOWN
    return normalized

def _raw_response_to_json_dict(raw_response: str) -> dict:
    """
    Extracts and loads the first JSON block for class/label/description using regex from the response.
    """

    # Match the first JSON-like object containing a label (keys may come in any order)
    match = re.search(
        r"\{[^{}]*\"label\"\s*:\s*\"[^\"]*\"[^{}]*\}",
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
    if "class" not in result:
        log.warning("raw_response is missing class: %s", raw_response)
        result["class"] = constants.VLM_CLASS_UNKNOWN
    result["class"] = _normalize_vlm_class(result["class"])

    return result

def identify_object_from_views(
    image_paths: Sequence[str],
    *,
    cluster_id: int,
    chunk_index: int,
) -> dict:
    """
    Send multiple isometric PNGs (rotating views) to the configured VLM provider (``vlm.provider``)
    and return a record for the cluster's data.yml.

    Raises ``VLMFatalError`` without retrying when the provider can't work at all (bad key, unknown model).
    """
    paths = list(image_paths)
    if not paths:
        raise ValueError("identify_object_from_views requires at least one image path.")

    provider = get_vlm_provider()

    prompt = (
        "You see several isometric views around a LiDAR point cloud region (same scene from different angles) "
        "with a light blue background. "
        "The scence is for a real world object, with the photos rotating around it. "
        "You must reply with a single JSON object ONLY !!, in this exact form:\n"
        '{"class": "<one of the classes defined blow>", "label": "<single word object name>", "description": "<description of 30 words max of what you see in the photo>"}\n'
        'the possible classes are:\n'
        '- UNKNOWN: for unclear objects\n'
        '- BUILDING: for walls and roofs\n'
        '- VEHICLE: for cars, bike, and motorcylces\n'
        '- PERSON: for people and animales\n'
        '- VEGETATION: for plants\n'
        '- OTHER: for everything else\n'
        'Examples are: \n'
        '- {label: "van", class:"VEHICLE", description: "a brown van parked in the lot"}\n'
        '- {label: "chair", class:"OTHER", description: "a black chair that fell over"}\n'
        '- {label: "tree", class:"VEGETATION", description: "A brown tree with yellow leafs"}\n'
        "Some guidelines:\n"
        '1. Some photos could be a cloud of random points that are not part of an object, name those "empty" and class "UNKNOWN"\n'
        '2. Only focus on the single object that is at the center of the photo, not the edges\n'
        '3. The identified object should be standing upright EX: if you see a pole, but it is side ways, than it is not a pole, name it "empty" and class "UNKNOWN"\n'
        '4. The colors of the identified object should make sense to its natural colors, EX: a "tree" without brown or green color should not be identified as a tree\n'
        '5. The photos are not too far away or too close to the object, it is close enough to see the object clearly and fully, so an answer like "birds from far away" is not valid one. Name those "empty" and class "UNKNOWN"\n'
    )

    images_b64 = [_encode_image_b64(p) for p in paths]

    _MAX_RETRIES = 3
    _RETRY_DELAY_SECONDS = 2

    vlm_class = constants.VLM_CLASS_UNKNOWN
    name = "empty"
    description = "error from VLM, see logs for details"

    for attempt in range(_MAX_RETRIES):
        try:
            raw = provider.generate(prompt, images_b64)
            json_dict = _raw_response_to_json_dict(raw)
            vlm_class = json_dict["class"]
            name = json_dict["label"]
            description = json_dict["description"]
            break

        except VLMFatalError:
            raise
        except Exception as ex:
            log.warning(
                "identify_object_from_views: %s attempt %d/%d failed (cluster=%d): %s",
                provider.name, attempt + 1, _MAX_RETRIES, cluster_id, ex,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_SECONDS)
    else:
        log.error(
            "identify_object_from_views: all %d attempts failed for cluster=%d, defaulting to empty",
            _MAX_RETRIES, cluster_id,
        )

    return make_cluster_record(
        cluster_id=cluster_id,
        chunk_index=chunk_index,
        vlm_class=vlm_class,
        name=name,
        description=description,
    )


def make_cluster_record(
    *,
    cluster_id: int,
    chunk_index: int = 0,
    vlm_class: str = constants.VLM_CLASS_UNKNOWN,
    name: str = "empty",
    description: str = "",
) -> dict:
    vlm_class = _normalize_vlm_class(vlm_class)
    return {
        "chunk_index": int(chunk_index),
        "cluster_id": int(cluster_id),
        "vlm_class": vlm_class,
        "name": str(name).strip().lower(),
        "description": str(description),
        "classification": constants.VLM_CLASS_TO_CLASSIFICATION[vlm_class],
    }


def build_cluster_data(record: dict) -> dict:
    """Content of a cluster's ``data.yml``; ``EXTRA`` is reserved for future keys."""
    return {
        "CLUSTER_ID": int(record["cluster_id"]),
        "CLASS": str(record["vlm_class"]),
        "NAME": str(record["name"]),
        "DESCRIPTION": str(record["description"]),
        "EXTRA": {},
    }


def write_cluster_data_yaml(trace_dir: str, record: dict) -> str:
    path = cluster_data_yaml_path(trace_dir)
    os.makedirs(trace_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(build_cluster_data(record), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    log.debug("Wrote %s", path)
    return path
