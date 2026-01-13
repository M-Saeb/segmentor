import laspy
import open3d as o3d
import numpy as np
from  utils import log_utils, pcd_utils

log = log_utils.get_dual_logger(__name__)
  
class LasWrapper:

    INIT_CLASSIFICATION = 1  # Unclassified
    GROUND_CLASSIFICATION = 2

    def __init__(self, las_path, chunk_size=5_000_000):
        """
        Wrapper to read LAS files either in chunks or bulk.
        Args:
            las_path: Path to the .las file
            chunk_size: Number of points to read per chunk (default 5 million)
        """
        self._las_path = las_path
        self._pcd = o3d.geometry.PointCloud()
        self._points_attributes = {}
        self._chunk_size = chunk_size
        self._setup_o3d_points_and_attributes()

    def _get_original_las_header(self):
        with laspy.open(self._las_path) as las:
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
        classification = np.full(num_points, self.INIT_CLASSIFICATION, dtype=np.uint8)
        self._points_attributes['classification'] = classification
        log.info("Set up Open3D point cloud with %d points", len(self._pcd.points))

    def _read_file_in_chunks(self):
        orignal_header = self._get_original_las_header()
        log.info("Reading LAS file in chunks of %d / %d...", orignal_header.point_count, self._chunk_size)
        expected_number_of_chunks = orignal_header.point_count // self._chunk_size
        log.info("Expected number of chunks: %d", expected_number_of_chunks)
        with laspy.open(self._las_path) as f:
            for index, points in enumerate(f.chunk_iterator(self._chunk_size)):
                log.info("Read chunk %d with %d points", index, len(points))
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

    def write_to_file(self, output_path):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        log.info("Writing Open3D point cloud to %s in chunks...", output_path)
        points = np.asarray(self._pcd.points)

        num_points = points.shape[0]
        chunk_size = self._chunk_size
        new_las_header = self._create_new_las_filer_header()
        with laspy.open(output_path, mode="w", header=new_las_header) as writer:
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
                log.info("Wrote chunk %d points to %s", index//chunk_size, output_path)
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
        log.info("Filtering ground using Z coordinate histogram...")
        points = np.asarray(self._pcd.points)
        z_values = points[:, 2]
        # Step 1: Create Z histogram (to make it usable)
        x, y = pcd_utils.create_histogram(z_values)
        # Step 2: Find peak in Z histogram
        peak_idx, left_idx, right_idx = pcd_utils.find_peak_and_drops(x, y)
        log.info("Peak at Z=%.2f, threshold at Z=%.2f", x[peak_idx], x[right_idx])
        # Step 3: Split cloud
        ground_mask = z_values <= x[right_idx]
        # Step 4: Update classification using previous values
        classification = self._points_attributes['classification']
        classification[ground_mask] = self.GROUND_CLASSIFICATION
        self._points_attributes['classification'] = classification
        log.info("Assigned classification: ground=%d, non-ground=%d", np.sum(ground_mask), np.sum(~ground_mask))