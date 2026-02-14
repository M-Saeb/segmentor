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
        self._setup_o3d_points_and_attributes()

    def _get_original_las_header(self):
        with laspy.open(self.config.input_file) as las:
            return las.header

    def _create_new_las_filer_header(self):
        original_header = self._get_original_las_header()
        new_header = laspy.LasHeader(point_format=original_header.point_format, version=original_header.version)
        new_header.offsets = original_header.offsets
        new_header.scales = original_header.scales
        return new_header

    def _setup_o3d_points_and_attributes(self):
        pcd = o3d.geometry.PointCloud()
        all_points = []
        points_attributes = {}
        for points in self._read_file_in_chunks():
            xyz = np.vstack((points.x, points.y, points.z)).T
            all_points.append(xyz)
            # Store all attributes except x, y, z
            for dim in points.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    if dim not in points_attributes:
                        points_attributes[dim] = []
                    points_attributes[dim].append(getattr(points, dim))
        if all_points:
            all_xyz = np.vstack(all_points)
            pcd.points = o3d.utility.Vector3dVector(all_xyz)
            # Concatenate all attribute arrays
            for dim, arrays in points_attributes.items():
                points_attributes[dim] = np.concatenate(arrays)

        self._pcd = pcd
        self._points_attributes = points_attributes
    
        # Initialize classification attribute for all points
        num_points = len(pcd.points)
        classification = np.full(num_points, classses.INIT_CLASSIFICATION, dtype=np.uint8)
        self._points_attributes['classification'] = classification
        log.info("Set up Open3D point cloud with %d points", len(self._pcd.points))

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

    def _update_pcd_and_points_attributes(self, indices):
        """
        Update the Open3D point cloud and attributes based on given indices.
        Args:
            indices: Indices of points to keep
        """
        self._pcd = self._pcd.select_by_index(indices)
        for dim, arr in self._points_attributes.items():
            self._points_attributes[dim] = arr[indices]

    def filter_noise(self, nb_neighbors=20, std_ratio=2.0):
        """
        Remove statistical outliers from the Open3D point cloud and update attributes accordingly.
        Args:
            nb_neighbors: Number of neighbors to analyze for each point.
            std_ratio: Standard deviation ratio threshold.
        """
        log.info("Filtering noise using statistical outlier removal (nb_neighbors=%d, std_ratio=%f)...", nb_neighbors, std_ratio)
        # Filter Open3D point cloud
        _, ind = self._pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        self._update_pcd_and_points_attributes(ind)
        log.info("Noise filtering complete. Remaining points: %d", len(self._pcd.points))

    def write_to_file(self):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        output_file_path = self.config.result["write_to_file"]
        log.info("Writing Open3D point cloud to %s in chunks...", output_file_path)
        points = np.asarray(self._pcd.points)

        num_points = points.shape[0]
        chunk_size = self.config.chunk_size
        new_las_header = self._create_new_las_filer_header()
        with laspy.open(output_file_path, mode="w", header=new_las_header) as writer:
            for index in range(0, num_points, chunk_size):
                chunk = points[index:index+chunk_size]
                las_chunk = laspy.LasData(new_las_header)
                las_chunk.x = chunk[:, 0]
                las_chunk.y = chunk[:, 1]
                las_chunk.z = chunk[:, 2]
                # Restore all other attributes
                for dim, arr in self._points_attributes.items():
                    setattr(las_chunk, dim, arr[index:index+chunk_size])
                writer.write_points(las_chunk.points)
                log.debug("Wrote chunk %d points", index//chunk_size)
        log.info("Finished writing Open3D point cloud to LAS file")

    def segment_ground(self):
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
            self._points_attributes = pcd_utils.basic_ground_filter(
                np.asarray(self._pcd.points),
                self._points_attributes,
                z_threshold=None
            )
        elif segementation['method'] == "csf":
            threshold_factor=segementation['csf_params']['threshold_factor']
            resolution = segementation['csf_params']['resolution']
            rigidness = segementation['csf_params']['rigidness']
            max_iter = segementation['csf_params']['max_iter']
            self._points_attributes = pcd_utils.cloth_simulation_filter(
                np.asarray(self._pcd.points),
                self._points_attributes,
                threshold_factor=threshold_factor,
                resolution=resolution,
                rigidness=rigidness,
                max_iter=max_iter
            )
        else:
            log.error("Unknown ground filter method: %s", segementation['method'])
            raise ValueError(f"Unknown ground filter method: {segementation['method']}")
        
    def display_pcb(self):
        log.info("Displaying point cloud with classification-based coloring...")
        # Try to color points using the 'classification' attribute if available
        cls = self._points_attributes.get('classification', None)
        if cls is None or len(cls) != len(self._pcd.points):
            o3d.visualization.draw_geometries([self._pcd])
            return

        cls = np.asarray(cls, dtype=int)
        # predefined colors for known classes
        color_map = {
            classses.INIT_CLASSIFICATION: np.array([0.5, 0.5, 0.5]),  # unclassified -> gray
            classses.GROUND_CLASSIFICATION: np.array([0.0, 1.0, 0.0]),  # ground -> green
        }

        colors = np.empty((len(cls), 3), dtype=float)
        for i, c in enumerate(cls):
            if c in color_map:
                colors[i] = color_map[c]
            else:
                # deterministic fallback color per class id
                ci = int(c)
                colors[i] = np.array([((ci * 37) % 255) / 255.0,
                                      ((ci * 61) % 255) / 255.0,
                                      ((ci * 97) % 255) / 255.0])

        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self._pcd])

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
        if self.config.filter_noise:
            self.filter_noise()

        if self.config.ground_segmentation["enable"]:
            self.segment_ground()

        if self.config.wall_segmentation["enable"]:
            self.segment_vertical_surfaces()

        if self.config.result["write_to_file"]:
            self.write_to_file()

        if self.config.result["display_result"]:
            self.display_pcb()