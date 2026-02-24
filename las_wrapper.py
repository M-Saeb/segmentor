import laspy
import open3d as o3d
import numpy as np
from utils import log_utils, pcd_utils, config, classses

log = log_utils.get_dual_logger(__name__)
  
class LasWrapper:

    def __init__(self, config_file_path):
        """
        Wrapper to read LAS files either in chunks or bulk.
        Args:
            config_file_path: Path to the configuration YAML file
            chunk_size: Number of points to read per chunk (default 5 million)
        """
        self.config = config.SegmentorConfig(config_file_path)
        self._pcd = o3d.geometry.PointCloud()
        self._points_attributes = {}
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

    def get_chunks_loop(self):
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
                classification = np.full(num_points, classses.INIT_CLASSIFICATION, dtype=np.uint8)
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

    def filter_noise(self, index, pcd, points_attributes, nb_neighbors=20, std_ratio=2.0):
        """
        Remove statistical outliers from the Open3D point cloud and update attributes accordingly.
        Args:
            pcv: The Open3D point cloud to filter
            points_attributes: The dictionary of point attributes to filter
            nb_neighbors: Number of neighbors to analyze for each point.
            std_ratio: Standard deviation ratio threshold.
        """
        log.info("Chunk %d: Filtering noise", index)
        # Filter Open3D point cloud
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        new_pcd = pcd.select_by_index(ind)
        new_points_attributes = {}
        for dim, arr in points_attributes.items():
            new_points_attributes[dim] = arr[ind]

        return new_pcd, new_points_attributes


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

    def segment_ground(self, pcd, points_attributes):
        """
        Segment ground/non-ground by analyzing Z coordinate distribution and assign classification numbers.
        Steps:
        1. Export Z to ScalarField
        2. Find start/end of peak in Z histogram
        3. Split cloud at peak end
        4. Assign classification numbers
        """
        segementation = self.config.ground_segmentation
        log.info("Segmenting ground points using method: %s", segementation['method'])
        if segementation['method'] == "basic":
            points_attributes = pcd_utils.basic_ground_filter(
                np.asarray(pcd.points),
                points_attributes,
                z_threshold=None
            )
        elif segementation['method'] == "csf":
            threshold_factor=segementation['csf_params']['threshold_factor']
            resolution = segementation['csf_params']['resolution']
            rigidness = segementation['csf_params']['rigidness']
            max_iter = segementation['csf_params']['max_iter']
            points_attributes = pcd_utils.cloth_simulation_filter(
                np.asarray(pcd.points),
                points_attributes,
                threshold_factor=threshold_factor,
                resolution=resolution,
                rigidness=rigidness,
                max_iter=max_iter
            )
        else:
            log.error("Unknown ground filter method: %s", segementation['method'])
            raise ValueError(f"Unknown ground filter method: {segementation['method']}")
        return pcd, points_attributes
        
    def display_pcb(self, pcd, points_attributes, colors="classification"):
        """
        Display the point cloud with a chosen coloring scheme.

        Args:
            pcd: Open3D point cloud.
            points_attributes: dict of point attributes (e.g., 'classification', 'red', 'green', 'blue').
            colors: Either "classification" (default) or "rgb".
                - "classification": use current classification-based coloring.
                - "rgb": use the original RGB colors from the LAS attributes.
        """


        if colors == "rgb":
            pcd = pcd_utils.apply_rgb_colors(pcd, points_attributes)
        elif colors == "classification":
            pcd = pcd_utils.apply_classification_colors(pcd, points_attributes)
        else:
            log.warning("Unknown colors mode '%s'; defaulting to classification-based coloring.", colors)
            pcd = pcd_utils.apply_classification_colors(pcd, points_attributes)

        o3d.visualization.draw_geometries([pcd])

    def segment_vertical_surfaces(self):
        """
        Segment vertical surfaces using normal vectors on unclassified points only.
        """
        # Filter to only unclassified points
        classification = self._points_attributes['classification']
        unclassified_mask = classification == classses.INIT_CLASSIFICATION
        unclassified_indices = np.where(unclassified_mask)[0]
        
        if len(unclassified_indices) == 0:
            log.info("No unclassified points to segment for vertical surfaces")
            return
        
        # Extract unclassified points and their attributes
        points = np.asarray(self._pcd.points)
        unclassified_points = points[unclassified_indices]
        
        unclassified_attributes = {}
        for dim, arr in self._points_attributes.items():
            unclassified_attributes[dim] = arr[unclassified_indices]
        
        # Segment vertical surfaces on unclassified subset
        updated_attributes = pcd_utils.apply_ransac_wall_detection(
            unclassified_points,
            unclassified_attributes
        )
        
        # Update classification in original array
        self._points_attributes['classification'][unclassified_indices] = updated_attributes['classification']

    def run_steps(self):

        with laspy.open(self.config.result["write_to_file"], mode="w", header=self._create_new_las_filer_header()) as writer:
            pass


        for index, [pcd, points_attributes] in enumerate(self.get_chunks_loop()):
            if self.config.filter_noise:
                pcd, points_attributes = self.filter_noise(index, pcd, points_attributes)

            if self.config.ground_segmentation["enable"]:
                pcd, points_attributes = self.segment_ground(pcd, points_attributes)

            if self.config.wall_segmentation["enable"]:
                self.segment_vertical_surfaces()

            if self.config.result["display_result"]:
                self.display_pcb(pcd, points_attributes, colors="rgb")

            if self.config.result["write_to_file"]:
                self.write_pcb_chunk_to_file(index, pcd, points_attributes)
                log.info("Chunk %d: Point cloud written", index)
