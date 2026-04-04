from abc import ABC, abstractmethod
import open3d as o3d
import laspy
import numpy as np
from modules.utils import constants

class LasIterator(ABC):
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

class LasIteratorBasic(LasIterator):
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