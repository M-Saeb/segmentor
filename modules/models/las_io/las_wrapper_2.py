import os
import hashlib
import laspy
import open3d as o3d
import numpy as np
from modules.utils import log_utils, config, constants
from .las_snapshot import LasSnapshot
from .las_iterator import LasIteratorBasic

log = log_utils.get_dual_logger(__name__)

class LasWrapper2:

    def __init__(self, config: config.SegmentorConfig):
        """
        High-level wrapper around LAS I/O and chunk iteration.

        Args:
            config: Parsed `SegmentorConfig` holding input path, chunk size.
        """
        self.input_file = config.input_file
        self.chunk_size = config.chunk_size
        self.reset_classification = config.reset_classification
        # Preview navigation state (loaded lazily via iterate_chunk()).
        self._las_iterator = LasIteratorBasic(self.input_file, self.chunk_size)
        self.current_chunk = None
        self.snapshot = LasSnapshot(self._get_original_las_header())
        log.debug("Chunk size set to: %d", self.chunk_size)

    def genrate_unique_id(self) -> str:
        """Read the meta data of the LAS to generate a unique id for the file.
        """
        header = self._get_original_las_header()
        items = []
        for key in header.__dict__.keys():
            items.append(f"{key}: {getattr(header, key)}")
        raw_string = "\n".join(items)
        return hashlib.sha256(raw_string.encode()).hexdigest()


    def _reset_classification_if_needed(self, points_attributes):
        if self.reset_classification:
            points_attributes['classification'] = np.full(
                len(points_attributes['classification']),
                constants.PCB_CLASSIFICATION_INIT,
                dtype=np.uint8
            )
        return points_attributes

    def iterate_chunk(self, delta: int) -> bool:
        """
        Move `current_chunk` forward or backward along input chunks.

        Args:
            delta: +1 for next chunk, -1 for previous (other values move by
                that many chunks).

        Returns:
            True if the current chunk changed, False if the target index is
            out of range (e.g. no next/previous chunk).
        """
        ok = self._las_iterator.iterate_chunk(delta)
        if not ok:
            return False
        pcd, points_attributes = self._las_iterator.get_current_chunk()
        points_attributes = self._reset_classification_if_needed(points_attributes)
        self.current_chunk = pcd, points_attributes
        return True

    def _get_original_las_header(self):
        with laspy.open(self.input_file) as las:
            return las.header

    def _create_new_las_filer_header(self):
        original_header = self._get_original_las_header()
        new_header = laspy.LasHeader(point_format=original_header.point_format, version=original_header.version)
        new_header.offsets = original_header.offsets
        new_header.scales = original_header.scales
        return new_header

    def _update_pcd_and_points_attributes(self, pcd, points_attributes, indices):
        """
        Update the Open3D point cloud and attributes based on given indices.
        Args:
            pcd: The Open3D point cloud to update
            points_attributes: The dictionary of point attributes to update
            indices: Indices of points to keep
        """
        pcd = pcd.select_by_index(indices)
        for dim, arr in points_attributes.items():
            points_attributes[dim] = arr[indices]
        return pcd, points_attributes

    def get_current_chunk(self):
        if self.current_chunk is None:
            raise RuntimeError(
                "LasWrapper current chunk is not initialized. "
                "Call iterate_chunk(0) once before using get_current_chunk()."
            )
        return self.current_chunk

    def set_current_chunk(self, pcd, points_attributes):
        self.current_chunk = pcd, points_attributes

    def save_snapshot(self):
        pcd, points_attributes = self.get_current_chunk()
        self.snapshot.save_snapshot(pcd, points_attributes)

    def restore_snapshot(self):
        pcd, points_attributes = self.snapshot.restore_snapshot()
        self.set_current_chunk(pcd, points_attributes)

    def get_snapshot(self):
        return self.snapshot.get_snapshot()

    def delete_snapshot(self):
        self.snapshot.delete_snapshot()

    def iter_input_chunks(self):
        """
        Public iterator over all input chunks as (index, pcd, points_attributes).
        """
        index = 0
        while self._las_iterator.iterate_chunk(delta=(0 if index == 0 else 1)):
            pcd, points_attributes = self._las_iterator.get_current_chunk()
            points_attributes = self._reset_classification_if_needed(points_attributes)
            yield index, pcd, points_attributes
            index += 1

    def initialize_output_file(self, full_path):
        """
        Create or truncate the output LAS file with a fresh header.
        Subsequent writes should use write_pcb_chunk_to_file in append mode.
        """
        with laspy.open(
            full_path,
            mode="w",
            header=self._create_new_las_filer_header(),
        ) as writer:
            # Header is written on open; no points yet.
            pass

    def write_pcb_chunk_to_file(self, index, pcd, points_attributes, full_path):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        log.info("vvv Writing chunk %d vvv", index)
        points = np.asarray(pcd.points)

        new_las_header = self._create_new_las_filer_header()
        with laspy.open(full_path, mode="a", header=new_las_header) as writer:
            las_chunk = laspy.LasData(new_las_header)
            las_chunk.x = points[:, 0]
            las_chunk.y = points[:, 1]
            las_chunk.z = points[:, 2]
            # Restore LAS dimensions only (skip internal keys like object_cluster_id)
            for dim, arr in points_attributes.items():
                if dim in ('object_cluster_id', ):
                    continue
                setattr(las_chunk, dim, arr)
            writer.append_points(las_chunk.points)
            log.info("^^^ Wrote chunk %d points ^^^", index)
        
