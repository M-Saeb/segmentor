import laspy
import open3d as o3d
import numpy as np
from models.utils import log_utils, pcd_utils, config, constants

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



    def filter_noise_for_none_classified_points(self, index, pcd, points_attributes, nb_neighbors=20, std_ratio=2.0):
        """
        Apply statistical noise filtering only to points that are still in the
        initial (unclassified) state, keeping all classified points untouched.

        Args:
            index: Chunk index (for logging only).
            pcd: Open3D point cloud to filter.
            points_attributes: Dictionary of point attributes.
            nb_neighbors: Number of neighbors for statistical outlier removal.
            std_ratio: Standard deviation ratio threshold.
        """
        classification = points_attributes.get("classification", None)
        if classification is None or len(classification) != len(pcd.points):
            log.warning(
                "filter_noise_for_none_classified_points called but 'classification' "
                "is missing or has mismatched length; falling back to full-cloud noise filter."
            )
            return self.filter_noise(index, pcd, points_attributes, nb_neighbors, std_ratio)

        classification = np.asarray(classification)
        unclassified_mask = classification == constants.PCB_CLASSIFICATION_INIT
        if not np.any(unclassified_mask):
            log.info("No unclassified points to filter; returning original point cloud.")
            return pcd, points_attributes

        log.info("Chunk %d: Filtering noise on %d unclassified points", index, np.sum(unclassified_mask))

        # Work on the unclassified subset only
        unclassified_indices = np.where(unclassified_mask)[0]
        subset_pcd = pcd.select_by_index(unclassified_indices)

        # Run the same statistical outlier removal as in filter_noise, but on the subset
        _, subset_keep_idx = subset_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        # Map kept indices back to original indices
        kept_unclassified_indices = unclassified_indices[subset_keep_idx]

        # Build a global "keep" mask: all classified points are kept,
        # unclassified points are kept only if they survived the filter.
        num_points = len(pcd.points)
        keep_mask = np.ones(num_points, dtype=bool)
        # Start by dropping all unclassified points
        keep_mask[unclassified_indices] = False
        # Re-enable only the filtered unclassified points
        keep_mask[kept_unclassified_indices] = True

        final_indices = np.where(keep_mask)[0]
        new_pcd = pcd.select_by_index(final_indices)

        new_points_attributes = {}
        for dim, arr in points_attributes.items():
            arr_np = np.asarray(arr)
            if len(arr_np) == num_points:
                new_points_attributes[dim] = arr_np[final_indices]
            else:
                # Attributes not per-point (e.g., metadata) are kept as-is
                new_points_attributes[dim] = arr

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
        
    def display_pcb(self, pcd, points_attributes, colors="classification", only_none_classified=False):
        """
        Display the point cloud with a chosen coloring scheme.

        Args:
            pcd: Open3D point cloud.
            points_attributes: dict of point attributes (e.g., 'classification', 'red', 'green', 'blue').
            colors: Either "classification" (default) or "rgb".
                - "classification": use current classification-based coloring.
                - "rgb": use the original RGB colors from the LAS attributes.
            only_none_classified: If True, only visualize points whose classification
                is still INIT_CLASSIFICATION (i.e., not clearly classified yet).
        """

        # Optionally restrict to points that are not yet clearly classified.
        display_pcd = pcd
        display_attrs = points_attributes
        if only_none_classified:
            cls = points_attributes.get("classification", None)
            cls_arr = np.asarray(cls)
            mask = cls_arr == constants.PCB_CLASSIFICATION_INIT
            if not np.any(mask):
                log.info("No unclassified points to display; skipping visualization.")
                return

            indices = np.where(mask)[0]
            display_pcd = pcd.select_by_index(indices)
            # Filter all attributes consistently
            filtered_attrs = {}
            for dim, arr in points_attributes.items():
                arr_np = np.asarray(arr)
                if len(arr_np) == len(pcd.points):
                    filtered_attrs[dim] = arr_np[indices]
                else:
                    # Keep attributes that don't match point count as-is
                    filtered_attrs[dim] = arr
            display_attrs = filtered_attrs

        # Apply coloring scheme to the (possibly filtered) subset.
        if colors == "rgb":
            display_pcd = pcd_utils.apply_rgb_colors(display_pcd, display_attrs)
        elif colors == "classification":
            display_pcd = pcd_utils.apply_classification_colors(display_pcd, display_attrs)
        else:
            log.warning("Unknown colors mode '%s'; defaulting to classification-based coloring.", colors)
            display_pcd = pcd_utils.apply_classification_colors(display_pcd, display_attrs)

        o3d.visualization.draw_geometries([display_pcd])

    def segment_vertical_surfaces(self, pcd, points_attributes):
        """
        Segment vertical surfaces using normal vectors on unclassified points only.
        """
        # Filter to only unclassified points
        classification = points_attributes['classification']
        unclassified_mask = classification == constants.PCB_CLASSIFICATION_INIT
        unclassified_indices = np.where(unclassified_mask)[0]
        
        if len(unclassified_indices) == 0:
            log.info("No unclassified points to segment for vertical surfaces")
            return
        
        # Extract unclassified points and their attributes
        points = np.asarray(pcd.points)
        unclassified_points = points[unclassified_indices]
        
        unclassified_attributes = {}
        for dim, arr in points_attributes.items():
            unclassified_attributes[dim] = arr[unclassified_indices]
        
        # Segment vertical surfaces on unclassified subset
        updated_attributes = pcd_utils.apply_ransac_wall_detection(
            unclassified_points,
            unclassified_attributes
        )
        
        # Update classification in original array
        points_attributes['classification'][unclassified_indices] = updated_attributes['classification']

        return pcd, points_attributes

    def run_steps(self):

        with laspy.open(self.config.result["write_to_file"], mode="w", header=self._create_new_las_filer_header()) as writer:
            pass


        for index, [pcd, points_attributes] in enumerate(self._read_input_file_in_chunks()):
            if self.config.filter_noise:
                pcd, points_attributes = self.filter_noise(index, pcd, points_attributes)

            if self.config.ground_segmentation["enable"]:
                pcd, points_attributes = self.segment_ground(pcd, points_attributes)

            if self.config.wall_segmentation["enable"]:
                pcd, points_attributes = self.segment_vertical_surfaces(pcd, points_attributes)

                if self.config.wall_segmentation["filter_noise_after_segmenetation"]:
                    pcd, points_attributes  = self.filter_noise_for_none_classified_points(index, pcd, points_attributes )


            if self.config.result["display_result"]:
                self.display_pcb(pcd, points_attributes, colors="classification")
                # self.display_pcb(pcd, points_attributes, colors="rgb", only_none_classified=True)

            if self.config.result["write_to_file"]:
                self.write_pcb_chunk_to_file(index, pcd, points_attributes)
                log.info("Chunk %d: Point cloud written", index)

            break