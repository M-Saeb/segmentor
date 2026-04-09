"""Regression tests for global object cluster ID allocation."""

from __future__ import annotations

import numpy as np
import pytest

from modules.models.object_cluster_id_manager import ObjectClusterIdManager
from modules.models.pcd_tools import segment_objects_dbscan
from modules.utils import constants


class _FakePointCloud:
    def __init__(self, point_count: int, labels):
        self.points = [None] * point_count
        self._labels = np.asarray(labels, dtype=np.int32)

    def cluster_dbscan(self, eps, min_points, print_progress=False):  # noqa: ARG002
        return self._labels


@pytest.fixture(autouse=True)
def reset_cluster_id_manager():
    ObjectClusterIdManager.get_instance().reset(start=1)
    yield
    ObjectClusterIdManager.get_instance().reset(start=1)


def test_object_cluster_id_manager_singleton_and_monotonic_allocation():
    m1 = ObjectClusterIdManager.get_instance()
    m2 = ObjectClusterIdManager.get_instance()
    assert m1 is m2

    assert m1.reserve_next_ids(3) == [1, 2, 3]
    assert m2.reserve_next_ids(2) == [4, 5]


def test_segment_objects_dbscan_assigns_unique_ids_across_calls():
    pcd_1 = _FakePointCloud(point_count=5, labels=[0, 0, 1, 1, -1])
    attrs_1 = {
        "classification": np.full(5, constants.PCB_CLASSIFICATION_INIT, dtype=np.int32),
    }
    _, out_1 = segment_objects_dbscan(pcd_1, attrs_1, eps=0.1, min_points=2)
    ids_1 = out_1["object_cluster_id"]
    assert set(np.unique(ids_1[ids_1 > 0]).tolist()) == {1, 2}

    pcd_2 = _FakePointCloud(point_count=4, labels=[0, 0, 1, 1])
    attrs_2 = {
        "classification": np.full(4, constants.PCB_CLASSIFICATION_INIT, dtype=np.int32),
    }
    _, out_2 = segment_objects_dbscan(pcd_2, attrs_2, eps=0.1, min_points=2)
    ids_2 = out_2["object_cluster_id"]
    assert set(np.unique(ids_2[ids_2 > 0]).tolist()) == {3, 4}


def test_object_cluster_id_manager_reset_restores_start_index():
    manager = ObjectClusterIdManager.get_instance()
    manager.reset(start=10)
    assert manager.reserve_next_ids(2) == [10, 11]

    manager.reset(start=1)
    assert manager.reserve_next_ids(1) == [1]
