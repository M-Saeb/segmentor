"""
End-to-end tests for PipelineController full-chunk processing (RANSAC segmentation).

Golden LAS files were produced with the bundled fixtures under
``tests/fixtures/pipeline_ransac`` and fixed YAML parameters in this module.
Open3D ``segment_plane`` is randomized; outputs were verified repeatable on the
developer machine—if tests flap across Open3D versions, narrow assertions or
refresh the golden files after a single local run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import laspy
import numpy as np
import pytest
import yaml

from modules.controllers.pipeline.main import PipelineController
from modules.utils.config import SegmentorConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pipeline_ransac"
INPUT_LAS_FIXTURE = FIXTURES_DIR / "1_pre_processing.las"
INPUT_INDEX_FIXTURE = FIXTURES_DIR / "1_pre_processing_index.json"

EXPECTED_GROUND = FIXTURES_DIR / "expected_pipeline_ground.las"
EXPECTED_CEILING = FIXTURES_DIR / "expected_pipeline_ceiling.las"
EXPECTED_WALL = FIXTURES_DIR / "expected_pipeline_wall.las"

# Frozen parameters used when generating expected_pipeline_*.las (keep in sync).
GROUND_RANSAC_PARAMS = {"distance_threshold": 0.05, "num_iterations": 50}
CEILING_PARAMS = {"facing_up_threshold": 0.95, "min_inlier_density": 15}
WALL_PARAMS = {"rotation_threshold": 0.9}


@pytest.fixture(autouse=True)
def _reset_segmentor_singleton():
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None
    yield
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None


def _write_pipeline_config(config_path: Path, input_file: Path, output_dir: Path, extra: dict) -> None:
    payload = {
        "input_file": str(input_file),
        "chunk_size": 5_000_000,
        "reset_classification": True,
        "result": {
            "display_result": False,
            "output_folder_path": str(output_dir),
            "overwrite_output_folder": False,
        },
        **extra,
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _seed_result_folder(output_dir: Path) -> None:
    shutil.copy(INPUT_LAS_FIXTURE, output_dir / "1_pre_processing.las")
    shutil.copy(INPUT_INDEX_FIXTURE, output_dir / "1_pre_processing_index.json")


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
    )
    for dim in preferred_dims:
        if hasattr(actual, dim) and hasattr(expected, dim):
            assert np.array_equal(getattr(actual, dim), getattr(expected, dim))


def _run_pipeline_case(tmp_path: Path, extra_config: dict, expected_las: Path) -> None:
    output_dir = tmp_path / "result"
    output_dir.mkdir(parents=True)
    _seed_result_folder(output_dir)

    input_las = output_dir / "1_pre_processing.las"
    config_path = tmp_path / "config.yml"
    _write_pipeline_config(config_path, input_las, output_dir, extra_config)

    SegmentorConfig.initialize(str(config_path))
    controller = PipelineController()
    assert len(controller.setup_pipeline.stages) == 1
    controller.setup_pipeline.stages[0].enabled = True
    controller.build_process_pipeline_from_setup()
    controller.run_process_pipeline_on_all_chunks()

    actual_path = Path(controller.full_las_path)
    assert actual_path.exists()
    assert actual_path.name == "2_pipeline.las"
    _assert_las_numeric_core(actual_path, expected_las)


def test_pipeline_ground_ransac(tmp_path: Path):
    _run_pipeline_case(
        tmp_path,
        {
            "ground_segmentation": {
                "enable": True,
                "method": "ransac",
                "ransac_params": dict(GROUND_RANSAC_PARAMS),
            },
            "ceiling_segmentation": {"enable": False},
            "wall_segmentation": {"enable": False},
        },
        EXPECTED_GROUND,
    )


def test_pipeline_ceiling_ransac(tmp_path: Path):
    _run_pipeline_case(
        tmp_path,
        {
            "ground_segmentation": {"enable": False},
            "ceiling_segmentation": {"enable": True, **CEILING_PARAMS},
            "wall_segmentation": {"enable": False},
        },
        EXPECTED_CEILING,
    )


def test_pipeline_wall_ransac(tmp_path: Path):
    _run_pipeline_case(
        tmp_path,
        {
            "ground_segmentation": {"enable": False},
            "ceiling_segmentation": {"enable": False},
            "wall_segmentation": {"enable": True, **WALL_PARAMS},
        },
        EXPECTED_WALL,
    )
