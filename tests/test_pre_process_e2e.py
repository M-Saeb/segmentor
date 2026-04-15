import json
from pathlib import Path

import laspy
import numpy as np
import pytest
import yaml

from modules.controllers.pre_process.main import PreProcessingController
from modules.utils.config import SegmentorConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pre_process"
INPUT_FIXTURE = FIXTURES_DIR / "input_minimal.las"
EXPECTED_OUTPUT_FIXTURE = FIXTURES_DIR / "expected_preprocess_output.las"
EXPECTED_INDEX_FIXTURE = FIXTURES_DIR / "expected_preprocess_index.json"

RUN_PARAMS = {
    "input_chunk_size": 3,
    "output_chunk_size": 4,
    "filter_noise": {"nb_neighbors": 3, "std_ratio": 2.0},
    "downsample": {"voxel_size": 0.6},
}


@pytest.fixture(autouse=True)
def _reset_segmentor_singleton():
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None
    yield
    SegmentorConfig._instance = None
    SegmentorConfig._config_path = None


def _write_test_config(config_path: Path, input_file: Path, output_dir: Path) -> None:
    payload = {
        "input_file": str(input_file),
        "result": {
            "display_result": False,
            "output_folder_path": str(output_dir),
            "overwrite_output_folder": False,
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


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


def _assert_index_payload(actual_path: Path, expected_path: Path) -> None:
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    assert set(actual) == {"target_points_per_tile", "num_seeds", "chunks"}
    assert actual["target_points_per_tile"] == expected["target_points_per_tile"]
    assert actual["num_seeds"] == expected["num_seeds"]
    assert actual["chunks"] == expected["chunks"]


def test_preprocess_from_run_function(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    output_dir = tmp_path / "result"
    _write_test_config(config_path, INPUT_FIXTURE, output_dir)
    SegmentorConfig.initialize(str(config_path))

    result_folder = Path(PreProcessingController().run(**RUN_PARAMS))
    output_las = result_folder / "1_pre_processing.las"
    output_index = result_folder / "1_pre_processing_index.json"

    assert output_las.exists()
    assert output_index.exists()
    _assert_las_numeric_core(output_las, EXPECTED_OUTPUT_FIXTURE)
    _assert_index_payload(output_index, EXPECTED_INDEX_FIXTURE)


def test_preprocess_run_from_user_input(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    output_dir = tmp_path / "result"
    _write_test_config(config_path, INPUT_FIXTURE, output_dir)
    SegmentorConfig.initialize(str(config_path))

    user_input = {
        "input_chunk_size": str(RUN_PARAMS["input_chunk_size"]),
        "output_chunk_size": str(RUN_PARAMS["output_chunk_size"]),
        "filter_noise": RUN_PARAMS["filter_noise"],
        "downsample": RUN_PARAMS["downsample"],
    }
    result_folder = Path(PreProcessingController().run_from_user_input(user_input))
    output_las = result_folder / "1_pre_processing.las"
    output_index = result_folder / "1_pre_processing_index.json"

    assert output_las.exists()
    assert output_index.exists()
    _assert_las_numeric_core(output_las, EXPECTED_OUTPUT_FIXTURE)
    _assert_index_payload(output_index, EXPECTED_INDEX_FIXTURE)
