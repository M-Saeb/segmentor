"""Regression tests for LasWrapper public API (see .cursor/skills/test-las-wrapper2-public-api/SKILL.md)."""

from __future__ import annotations

import math
import os

import laspy
import numpy as np
import pytest

from modules.models import LasWrapper
from modules.utils.config import SegmentorConfig
from modules.utils import constants


def _write_tiny_las(path: str, n_points: int = 100) -> None:
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = np.linspace(0, 1, n_points)
    las.y = np.linspace(0, 1, n_points)
    las.z = np.linspace(0, 1, n_points)
    # Point format 3 packs classification in 5 bits (max 31).
    las.classification = (np.arange(n_points) % 32).astype(np.uint8)
    las.write(path)


def _write_minimal_config(tmp_path, input_las, chunk_size: int, reset_classification: bool) -> str:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        f"input_file: {input_las.as_posix()!r}\n"
        f"chunk_size: {chunk_size}\n"
        f"reset_classification: {reset_classification}\n",
        encoding="utf-8",
    )
    return str(cfg_path)


@pytest.fixture
def tiny_las_path(tmp_path):
    path = tmp_path / "in.las"
    _write_tiny_las(str(path), n_points=100)
    return path


def _make_wrapper(tmp_path, tiny_las_path, chunk_size: int = 30, reset_classification: bool = True):
    cfg_path = _write_minimal_config(tmp_path, tiny_las_path, chunk_size, reset_classification)
    return LasWrapper(SegmentorConfig(cfg_path))


def test_init_sets_config_and_chunk_state(tmp_path, tiny_las_path):
    w = _make_wrapper(tmp_path, tiny_las_path)
    assert w.input_file == str(tiny_las_path)
    assert w.current_chunk is None
    assert w._las_iterator._current_chunk_index == 0
    assert w._las_iterator.get_total_items_number() == 4

def test_iterate_chunk_zero_loads_first_chunk(tmp_path, tiny_las_path):
    w = _make_wrapper(tmp_path, tiny_las_path, chunk_size=30, reset_classification=True)
    assert w.iterate_chunk(0) is True
    pcd, attrs = w.get_current_chunk()
    n = len(pcd.points)
    assert n == 30
    assert "classification" in attrs
    assert attrs["classification"].shape == (n,)


def test_iterate_chunk_boundaries(tmp_path, tiny_las_path):
    chunk_size = 30
    n = 100
    expected = math.ceil(n / chunk_size)
    w = _make_wrapper(tmp_path, tiny_las_path, chunk_size=chunk_size)
    assert w.iterate_chunk(0) is True
    for _ in range(expected - 1):
        assert w.iterate_chunk(1) is True
    assert w.iterate_chunk(1) is False
    assert w.iterate_chunk(-(expected - 1)) is True
    assert w._las_iterator._current_chunk_index == 0
    assert w.iterate_chunk(-1) is False


def test_get_current_chunk_raises_before_navigation(tmp_path, tiny_las_path):
    w = _make_wrapper(tmp_path, tiny_las_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        w.get_current_chunk()


def test_set_current_chunk_round_trip(tmp_path, tiny_las_path):
    w = _make_wrapper(tmp_path, tiny_las_path)
    w.iterate_chunk(0)
    pcd, attrs = w.get_current_chunk()
    w.set_current_chunk(pcd, attrs)
    pcd2, attrs2 = w.get_current_chunk()
    assert len(pcd2.points) == len(pcd.points)
    assert np.array_equal(np.asarray(pcd2.points), np.asarray(pcd.points))
    assert np.array_equal(attrs2["classification"], attrs["classification"])


def test_save_get_restore_delete_snapshot(tmp_path, tiny_las_path):
    w = _make_wrapper(tmp_path, tiny_las_path, chunk_size=100)
    snapshot_path = w.snapshot.path
    assert os.path.basename(snapshot_path).startswith("snapshot_")
    assert snapshot_path.endswith(f"_{w.snapshot.id}.las")

    w.iterate_chunk(0)
    pcd0, attrs0 = w.get_current_chunk()

    w.save_snapshot()
    assert os.path.isfile(snapshot_path)

    pcd_s, attrs_s = w.get_snapshot()
    assert len(pcd_s.points) == len(pcd0.points)
    assert np.allclose(np.asarray(pcd_s.points), np.asarray(pcd0.points))
    assert np.array_equal(attrs_s["classification"], attrs0["classification"])

    w.set_current_chunk(
        type(pcd0)(),
        {k: v for k, v in attrs0.items()},
    )
    w.restore_snapshot()
    pcd_r, attrs_r = w.get_current_chunk()
    assert np.allclose(np.asarray(pcd_r.points), np.asarray(pcd0.points))
    assert np.array_equal(attrs_r["classification"], attrs0["classification"])

    w.delete_snapshot()
    assert not os.path.exists(snapshot_path)


def test_iter_input_chunks_matches_expected_count(tmp_path, tiny_las_path):
    chunk_size = 30
    n = 100
    expected = math.ceil(n / chunk_size)
    w = _make_wrapper(tmp_path, tiny_las_path, chunk_size=chunk_size)
    chunks = list(w.iter_input_chunks())
    assert len(chunks) == expected
    for i, (idx, pcd, attrs) in enumerate(chunks):
        assert idx == i
        assert len(pcd.points) == (chunk_size if i < expected - 1 else n % chunk_size or chunk_size)


def test_initialize_output_and_write_pcb_chunk_writes_object_cluster_id(tmp_path, tiny_las_path):
    out = tmp_path / "out.las"
    w = _make_wrapper(tmp_path, tiny_las_path, chunk_size=100)
    w.iterate_chunk(0)
    pcd, attrs = w.get_current_chunk()
    attrs = dict(attrs)
    expected_ocid = np.arange(len(pcd.points), dtype=np.int32)
    attrs["object_cluster_id"] = expected_ocid

    w.initialize_output_file(str(out))
    w.write_pcb_chunk_to_file(0, pcd, attrs, str(out))

    with laspy.open(str(out)) as lasf:
        data = lasf.read()
    assert len(data.x) == len(pcd.points)
    names = [n.lower() for n in data.point_format.dimension_names]
    assert "object_cluster_id" in names
    assert np.array_equal(np.asarray(data.object_cluster_id), expected_ocid)


def test_empty_file_no_chunks(tmp_path):
    empty = tmp_path / "empty.las"
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = np.array([], dtype=np.float64)
    las.y = np.array([], dtype=np.float64)
    las.z = np.array([], dtype=np.float64)
    las.write(str(empty))
    cfg = _write_minimal_config(tmp_path, empty, 10, True)
    w = LasWrapper(SegmentorConfig(cfg))
    assert w.iterate_chunk(0) is False
    assert list(w.iter_input_chunks()) == []
