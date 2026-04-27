"""
End-to-end tests for PostProcessingController.confirm_post_processing.

Golden LAS files under ``tests/fixtures/post_process`` were produced with fixed
DBSCAN parameters and (for the mocked-VLM case) patched screenshot + VLM hooks.
"""

from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np
import pytest
import yaml

from modules.controllers.post_process.main import PostProcessingController
from modules.models.vlm_tools import identified_objects_yaml_path
from modules.utils import constants as C
from modules.utils.config import SegmentorConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "post_process"
PIPELINE_FIXTURE = FIXTURES_DIR / "2_pipeline.las"
ORIGINAL_FIXTURE = FIXTURES_DIR / "original_input.las"
EXPECTED_NO_VLM_LAS = FIXTURES_DIR / "expected_post_process_no_vlm.las"
EXPECTED_MOCK_VLM_LAS = FIXTURES_DIR / "expected_post_process_mock_vlm.las"

# Frozen parameters aligned with golden generation.
EPS_NO_CLUSTER = 0.15
MIN_POINTS_NO_CLUSTER = 500

EPS_CLUSTER = 0.1
MIN_POINTS_CLUSTER = 5

# Matches INIT dense-patch point count in ``tests/fixtures/post_process/2_pipeline.las``.
_CLUSTERED_INIT_POINT_COUNT = 20
_TEST_VLM_VIEWS_COUNT = 6

MINIMAL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def _reset_segmentor_singleton():
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None
    yield
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None


def _write_post_process_config(config_path: Path, original_input: Path, output_dir: Path) -> None:
    payload = {
        "input_file": str(original_input),
        "chunk_size": 1_000_000,
        "collect_runtime_report": False,
        "result": {
            "display_result": False,
            "output_folder_path": str(output_dir),
            "overwrite_output_folder": False,
        },
        "objects_segmentation": {
            "ollama_model": "gemma3:4b",
            "vlm_traces_subdir": "vlm_object_traces",
        },
        "vlm": {
            "views_count": _TEST_VLM_VIEWS_COUNT,
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _seed_post_process_fixtures(output_dir: Path, original_dest: Path) -> None:
    shutil.copy(PIPELINE_FIXTURE, output_dir / "2_pipeline.las")
    shutil.copy(ORIGINAL_FIXTURE, original_dest)


def _xyz(las: laspy.LasData) -> np.ndarray:
    return np.column_stack((las.x, las.y, las.z)).astype(np.float64)


def _assert_las_numeric_core(actual_path: Path, expected_path: Path) -> None:
    actual = laspy.read(actual_path)
    expected = laspy.read(expected_path)

    actual_xyz = _xyz(actual)
    expected_xyz = _xyz(expected)
    assert actual_xyz.shape == expected_xyz.shape
    assert np.allclose(actual_xyz, expected_xyz, atol=1e-8)

    preferred_dims = (
        "classification",
        "intensity",
        "return_number",
        "number_of_returns",
        "Clusters",
    )
    for dim in preferred_dims:
        if hasattr(actual, dim) and hasattr(expected, dim):
            assert np.array_equal(getattr(actual, dim), getattr(expected, dim))


def _stub_rich_media_screenshots(pcd, dir_name: str, number_of_views: int = _TEST_VLM_VIEWS_COUNT) -> list[str]:
    """Avoid Open3D OffscreenRenderer (can segfault headless); matches rich_media PNG names."""
    os.makedirs(dir_name, exist_ok=True)
    paths: list[str] = []
    for i in range(int(number_of_views)):
        path = os.path.join(dir_name, f"{i + 1:02d}.png")
        Path(path).write_bytes(MINIMAL_PNG_BYTES)
        paths.append(path)
    return paths


def _fake_identify_object_from_views(
    image_paths,
    *,
    cluster_id: int,
    chunk_index: int = 0,
    ollama_model: str = "gemma3:4b",
    runtime_report=None,
):
    """Synthetic VLM record: structural validity only (no assertion on semantic label text)."""
    rec = {
        "chunk_index": int(chunk_index),
        "cluster_id": int(cluster_id),
        "classification": int(C.PCB_CLASSIFICATION_OTHER_OBJECTS),
        "ai_label": "stub",
        "raw_response": '{"label":"stub"}',
    }
    assert isinstance(rec["raw_response"], str) and rec["raw_response"].strip()
    return rec


def _assert_valid_vlm_response_dict(d: dict) -> None:
    assert isinstance(d, dict)
    assert "classification" in d and "ai_label" in d and "raw_response" in d
    assert isinstance(d["raw_response"], str) and len(d["raw_response"].strip()) > 0


def test_post_process_confirm_no_clusters(tmp_path: Path):
    """DBSCAN yields no clusters; VLM skipped; deterministic LAS vs golden."""
    output_dir = tmp_path / "result"
    output_dir.mkdir(parents=True)
    original_input = tmp_path / "original_input.las"
    _seed_post_process_fixtures(output_dir, original_input)

    cfg = tmp_path / "config.yml"
    _write_post_process_config(cfg, original_input, output_dir)
    SegmentorConfig.initialize(str(cfg))

    ctrl = PostProcessingController()
    n_pts, out_las, yml_path = ctrl.confirm_post_processing(
        eps=EPS_NO_CLUSTER,
        min_points=MIN_POINTS_NO_CLUSTER,
    )

    assert n_pts > 0
    assert Path(out_las).exists()
    assert Path(out_las).name == "3_post_process.las"
    _assert_las_numeric_core(Path(out_las), EXPECTED_NO_VLM_LAS)

    assert Path(yml_path).exists()
    assert Path(yml_path) == Path(identified_objects_yaml_path(str(output_dir)))
    registry = yaml.safe_load(Path(yml_path).read_text(encoding="utf-8"))
    assert registry["cars"]["total"] == 0
    assert registry["other_objects"]["total"] == 0
    assert registry["cars"]["items"] == []
    assert registry["other_objects"]["items"] == []


def test_post_process_confirm_mocked_vlm_response_valid(tmp_path: Path):
    """One DBSCAN cluster; screenshots and VLM stubbed — assert integration + valid dict shape only."""
    output_dir = tmp_path / "result"
    output_dir.mkdir(parents=True)
    original_input = tmp_path / "original_input.las"
    _seed_post_process_fixtures(output_dir, original_input)

    cfg = tmp_path / "config.yml"
    _write_post_process_config(cfg, original_input, output_dir)
    SegmentorConfig.initialize(str(cfg))

    with patch(
        "modules.models.post_processing_tools.take_rich_media_screenshots",
        side_effect=_stub_rich_media_screenshots,
    ):
        with patch(
            "modules.models.post_processing_tools.identify_object_from_views",
            side_effect=_fake_identify_object_from_views,
        ) as mock_identify:
            ctrl = PostProcessingController()
            n_pts, out_las, yml_path = ctrl.confirm_post_processing(
                eps=EPS_CLUSTER,
                min_points=MIN_POINTS_CLUSTER,
            )

    assert mock_identify.call_count >= 1
    for call in mock_identify.call_args_list:
        resp = _fake_identify_object_from_views(*call.args, **call.kwargs)
        _assert_valid_vlm_response_dict(resp)

    assert n_pts > 0
    assert Path(out_las).exists()
    _assert_las_numeric_core(Path(out_las), EXPECTED_MOCK_VLM_LAS)

    out = laspy.read(out_las)
    assert hasattr(out, "Clusters"), "3_post_process.las must include Extra Bytes `Clusters` (DBSCAN ids)"
    cl = np.asarray(out.Clusters, dtype=np.int32)
    assert cl.shape[0] == n_pts
    assert set(np.unique(cl).tolist()) == {0, 1}
    assert int(np.count_nonzero(cl)) == _CLUSTERED_INIT_POINT_COUNT
    assert int(np.max(cl)) == 1
    cls = np.asarray(out.classification, dtype=np.int32)
    object_mask = cls == C.PCB_CLASSIFICATION_OBJECTS
    assert bool(np.all(object_mask == (cl > 0))), "Clusters>0 must match OBJECTS classification from DBSCAN"

    registry = yaml.safe_load(Path(yml_path).read_text(encoding="utf-8"))
    assert "cars" in registry and "other_objects" in registry
    assert registry["cars"]["total"] == 0
    assert registry["other_objects"]["total"] == 1
    assert len(registry["other_objects"]["items"]) == 1
    item = registry["other_objects"]["items"][0]
    assert "cluster_id" in item and "bounds" in item
    assert isinstance(item["bounds"], dict)
    assert "min" in item["bounds"] and "max" in item["bounds"]
