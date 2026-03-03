import laspy
import open3d as o3d
import numpy as np
from modules.utils import log_utils, config, constants
from . import pcd_tools

log = log_utils.get_dual_logger(__name__)
  
class LasWrapper:

    def __init__(self, config: config.SegmentorConfig):
        """
        Wrapper to read LAS files either in chunks or bulk.
        Args:
            config_file_path: Path to the configuration YAML file
            chunk_size: Number of points to read per chunk (default 5 million)
        """
        self.config = config
        self.chunks_iter = iter(self._read_input_file_in_chunks())
        self.current_chunk = next(self.chunks_iter)
        log.debug("Chunk size set to: %d", self.config.chunk_size)
        log.debug("Reset classification is set to: %s", self.config.reset_classification)


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
        return self.current_chunk

    def set_current_chunk(self, pcd, points_attributes):
        self.current_chunk = pcd, points_attributes

    def iter_input_chunks(self):
        """
        Public iterator over all input chunks as (index, pcd, points_attributes).
        """
        for index, (pcd, points_attributes) in enumerate(self._read_input_file_in_chunks()):
            yield index, pcd, points_attributes

    def initialize_output_file(self):
        """
        Create or truncate the output LAS file with a fresh header.
        Subsequent writes should use write_pcb_chunk_to_file in append mode.
        """
        with laspy.open(
            self.config.result["write_to_file"],
            mode="w",
            header=self._create_new_las_filer_header(),
        ) as writer:
            # Header is written on open; no points yet.
            pass

    def write_pcb_chunk_to_file(self, index, pcd, points_attributes):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        output_file_path = self.config.result["write_to_file"]
        log.debug("Writing chunk %d", index)
        points = np.asarray(pcd.points)

        new_las_header = self._create_new_las_filer_header()
        with laspy.open(output_file_path, mode="a", header=new_las_header) as writer:
            las_chunk = laspy.LasData(new_las_header)
            las_chunk.x = points[:, 0]
            las_chunk.y = points[:, 1]
            las_chunk.z = points[:, 2]
            # Restore all other attributes
            for dim, arr in points_attributes.items():
                setattr(las_chunk, dim, arr)
            writer.append_points(las_chunk.points)
            log.debug("Wrote chunk %d points", index)
