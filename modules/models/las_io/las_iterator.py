from abc import ABC, abstractmethod
import json
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d


class LasIteratorAbstract(ABC):
    def __init__(self, file_path: str, chunk_size: int):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.current_chunk = None
        self._current_chunk_index = 0

    @abstractmethod
    def iterate_chunk(self, delta: int) -> bool:
        pass

    @abstractmethod
    def get_current_chunk(self) -> tuple[o3d.geometry.PointCloud, dict]:
        pass


    def _read_file_in_chunks(self):
        """
        Read the LAS file in chunks and set up the Open3D point cloud and attributes.
        """
        with laspy.open(self.file_path) as f:
            for chunk in f.chunk_iterator(self.chunk_size):
                pcd = o3d.geometry.PointCloud()
                points_attributes = {}
                xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T
                pcd.points = o3d.utility.Vector3dVector(xyz)

                # Store all attributes except x, y, z
                for dim in chunk.point_format.dimension_names:
                    if dim not in ('x', 'y', 'z'):
                        if dim not in points_attributes:
                            points_attributes[dim] = []
                        points_attributes[dim].append(getattr(chunk, dim))

                for dim, arrays in points_attributes.items():
                    points_attributes[dim] = np.concatenate(arrays)
        
                yield pcd, points_attributes

    def get_total_items_number(self) -> int:
        og_header = None
        with laspy.open(self.file_path) as las:
            og_header = las.header
        n = og_header.point_count
        if self.chunk_size <= 0:
            return 1 if n > 0 else 0
        return (n + self.chunk_size - 1) // self.chunk_size

class LasIteratorBasic(LasIteratorAbstract):
    def iterate_chunk(self, delta: int) -> bool:
        new_index = self._current_chunk_index + delta
        for i, (pcd, points_attributes) in enumerate(
            self._read_file_in_chunks()
        ):
            if i == new_index:
                self._current_chunk_index = i
                self.current_chunk = pcd, points_attributes
                return True
        return False

    def get_current_chunk(self) -> tuple[o3d.geometry.PointCloud, dict]:
        if self.current_chunk is None:
            raise RuntimeError("Current chunk is not initialized. Call iterate_chunk(0) once before using get_current_chunk().")
        return self.current_chunk


class LasIteratorKDTree(LasIteratorAbstract):
    """
    Iterate logical point ranges from an index JSON (``chunks`` with ``start`` / ``end``).

    Each item is built by streaming the LAS file in ``chunk_size``-sized reads and merging
    slices whose global indices overlap the logical range (same idea as ``kd_tree()`` in
    sandbox). Non-XYZ dimensions are concatenated into ``points_attributes`` like
    :class:`LasIteratorBasic`.
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int,
        index_json_path: str | None = None,
    ):
        super().__init__(file_path, chunk_size)
        idx_path = index_json_path or str(Path(file_path).parent / "index.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks_data = data["chunks"]

    def get_total_items_number(self) -> int:
        return len(self._chunks_data)

    def _build_logical_chunk(self, chunk_idx: int) -> tuple[o3d.geometry.PointCloud, dict]:
        chunk_spec = self._chunks_data[chunk_idx]
        start = int(chunk_spec["start"])
        end = int(chunk_spec["end"])
        logical_pcd = o3d.geometry.PointCloud()
        attr_parts: dict[str, list[np.ndarray]] = {}
        file_base = 0
        with laspy.open(self.file_path) as f:
            for las_chunk in f.chunk_iterator(self.chunk_size):
                n = len(las_chunk.x)
                chunk_end = file_base + n
                overlap_lo = max(start, file_base)
                overlap_hi = min(end, chunk_end)
                if overlap_lo < overlap_hi:
                    i0 = overlap_lo - file_base
                    i1 = overlap_hi - file_base
                    x = np.asarray(las_chunk.x[i0:i1], dtype=np.float64)
                    y = np.asarray(las_chunk.y[i0:i1], dtype=np.float64)
                    z = np.asarray(las_chunk.z[i0:i1], dtype=np.float64)
                    xyz = np.column_stack((x, y, z))
                    part = o3d.geometry.PointCloud()
                    part.points = o3d.utility.Vector3dVector(xyz)
                    logical_pcd += part
                    for dim in las_chunk.point_format.dimension_names:
                        if dim not in ("x", "y", "z"):
                            arr = np.asarray(getattr(las_chunk, dim)[i0:i1])
                            attr_parts.setdefault(dim, []).append(arr)
                file_base = chunk_end
                if file_base >= end:
                    break
        points_attributes = {
            dim: np.concatenate(parts) for dim, parts in attr_parts.items()
        }
        return logical_pcd, points_attributes

    def iterate_chunk(self, delta: int) -> bool:
        new_index = self._current_chunk_index + delta
        if new_index < 0 or new_index >= len(self._chunks_data):
            return False
        self._current_chunk_index = new_index
        self.current_chunk = self._build_logical_chunk(new_index)
        return True

    def get_current_chunk(self) -> tuple[o3d.geometry.PointCloud, dict]:
        if self.current_chunk is None:
            raise RuntimeError("Current chunk is not initialized. Call iterate_chunk(0) once before using get_current_chunk().")
        return self.current_chunk