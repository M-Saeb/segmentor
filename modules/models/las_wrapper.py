import os
import hashlib
import laspy
import open3d as o3d
import numpy as np
from modules.utils import log_utils, config, constants

log = log_utils.get_dual_logger(__name__)
  
class LasWrapper:

    def __init__(self, config: config.SegmentorConfig):
        """
        High-level wrapper around LAS I/O and chunk iteration.

        Args:
            config: Parsed `SegmentorConfig` holding input path, chunk size, and
                behavior flags such as `reset_classification`.
        """
        self.config = config
        # Preview navigation state (loaded lazily via iterate_chunk()).
        self._chunk_index = 0
        self._total_chunks = None
        self.current_chunk = None
        log.debug("Chunk size set to: %d", self.config.chunk_size)
        log.debug("Reset classification is set to: %s", self.config.reset_classification)

    def genrate_unique_id(self) -> str:
        """Read the meta data of the LAS to generate a unique id for the file.
        """
        header = self._get_original_las_header()
        items = []
        for key in header.__dict__.keys():
            items.append(f"{key}: {getattr(header, key)}")
        raw_string = "\n".join(items)
        return hashlib.sha256(raw_string.encode()).hexdigest()

    def _compute_total_chunks(self) -> int:
        header = self._get_original_las_header()
        chunk_size = self.config.chunk_size
        n = header.point_count
        if chunk_size <= 0:
            return 1 if n > 0 else 0
        return (n + chunk_size - 1) // chunk_size

    def _load_chunk_at_index(self, index: int):
        """
        Load the i-th chunk from the input file (0-based). Reads sequentially
        from the start of the file until the requested index is reached.
        """
        for i, chunk in enumerate(self._read_input_file_in_chunks()):
            if i == index:
                return chunk
        raise IndexError(f"No chunk at index {index}")

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
        if self._total_chunks is None:
            self._total_chunks = self._compute_total_chunks()
        if self._total_chunks <= 0:
            return False
        new_index = self._chunk_index + delta
        if new_index < 0 or new_index >= self._total_chunks:
            return False
        self._chunk_index = new_index
        self.current_chunk = self._load_chunk_at_index(new_index)
        return True

    def _get_original_las_header(self):
        with laspy.open(self.config.input_file) as las:
            return las.header

    def _create_new_las_filer_header(self):
        original_header = self._get_original_las_header()
        new_header = laspy.LasHeader(point_format=original_header.point_format, version=original_header.version)
        new_header.offsets = original_header.offsets
        new_header.scales = original_header.scales
        return new_header

    def _read_input_file_in_chunks(self):
        """
        Read the LAS file in chunks and set up the Open3D point cloud and attributes.
        """
        for points in self._read_file_in_chunks():
            pcd = o3d.geometry.PointCloud()
            points_attributes = {}
            xyz = np.vstack((points.x, points.y, points.z)).T
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Store all attributes except x, y, z
            for dim in points.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    if dim not in points_attributes:
                        points_attributes[dim] = []
                    points_attributes[dim].append(getattr(points, dim))

            for dim, arrays in points_attributes.items():
                points_attributes[dim] = np.concatenate(arrays)
    
            # Initialize classification attribute for all points
            num_points = len(pcd.points)
            if self.config.reset_classification:
                classification = np.full(num_points, constants.PCB_CLASSIFICATION_INIT, dtype=np.uint8)
                points_attributes['classification'] = classification
            
            yield pcd, points_attributes

    def _read_file_in_chunks(self):
        orignal_header = self._get_original_las_header()
        chunk_size = self.config.chunk_size
        log.info("Reading LAS file in chunks of %d / %d...", orignal_header.point_count, chunk_size)
        expected_number_of_chunks = orignal_header.point_count // chunk_size
        log.debug("Expected number of chunks: %d", expected_number_of_chunks)
        with laspy.open(self.config.input_file) as f:
            for index, points in enumerate(f.chunk_iterator(chunk_size)):
                log.debug("Read chunk %d with %d points", index, len(points))
                yield points

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
        output_file_path = ".snapshot.las"
        points = np.asarray(pcd.points)

        new_las_header = self._create_new_las_filer_header()
        with laspy.open(output_file_path, mode="w", header=new_las_header) as writer:
            las_data = laspy.LasData(new_las_header)
            las_data.x = points[:, 0]
            las_data.y = points[:, 1]
            las_data.z = points[:, 2]
            # Restore all other attributes
            for dim, arr in points_attributes.items():
                setattr(las_data, dim, arr)
            writer.write_points(las_data.points)
            log.debug("Wrote snapshot")

    def restore_snapshot(self):
        """
        Restore the snapshot from the `.snapshot.las` file.
        """
        input_file_path = ".snapshot.las"
        if not os.path.exists(input_file_path):
            self.save_snapshot()

        with laspy.open(input_file_path) as reader:
            las_data = reader.read()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack((las_data.x, las_data.y, las_data.z)).T)
            points_attributes = {}
            for dim in las_data.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    points_attributes[dim] = getattr(las_data, dim)
            self.set_current_chunk(pcd, points_attributes)
            log.debug("Restored snapshot")

    def get_snapshot(self):
        input_file_path = ".snapshot.las"
        with laspy.open(input_file_path) as reader:
            las_data = reader.read()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack((las_data.x, las_data.y, las_data.z)).T)
            points_attributes = {}
            for dim in las_data.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    points_attributes[dim] = getattr(las_data, dim)
            return pcd, points_attributes

    def delete_snapshot(self):
        input_file_path = ".snapshot.las"
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
            log.info("Deleted snapshot")
        else:
            log.debug("Snapshot file does not exist")

    def iter_input_chunks(self):
        """
        Public iterator over all input chunks as (index, pcd, points_attributes).
        """
        for index, (pcd, points_attributes) in enumerate(self._read_input_file_in_chunks()):
            yield index, pcd, points_attributes

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
        
