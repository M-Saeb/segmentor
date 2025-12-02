import laspy
import open3d as o3d
import numpy as np

class LasWrapper:
    def __init__(self, las_path, chunk_size=5_000_000):
        """
        Wrapper to read LAS files either in chunks or bulk.
        Args:
            las_path: Path to the .las file
            chunk_size: Number of points to read per chunk (default 5 million)
        """
        self._las_path = las_path
        self._attributes = {}
        self._o3d_pcb = o3d.geometry.PointCloud()
        self._chunk_size = chunk_size
        self._las_header = self._create_new_las_filer_header()
        self._set_attributes_and_o3d_points()

    def _get_original_las_header(self):
        with laspy.open(self._las_path) as las:
            return las.header

    def _create_new_las_filer_header(self):
        with laspy.open(self._las_path) as las:
            original_header = las.header
            header = laspy.LasHeader(point_format=original_header.point_format, version=original_header.version)
            header.offsets = original_header.offsets
            header.scales = original_header.scales
            return header

    def _set_attributes_and_o3d_points(self):
        pcd = o3d.geometry.PointCloud()
        all_points = []
        attributes = {}
        for points in self._read_file_in_chunks():
            xyz = np.vstack((points.x, points.y, points.z)).T
            all_points.append(xyz)
            # Store all attributes except x, y, z
            for dim in points.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    if dim not in attributes:
                        attributes[dim] = []
                    attributes[dim].append(getattr(points, dim))
        if all_points:
            all_xyz = np.vstack(all_points)
            pcd.points = o3d.utility.Vector3dVector(all_xyz)
            # Concatenate all attribute arrays
            for dim, arrays in attributes.items():
                attributes[dim] = np.concatenate(arrays)

        self._attributes = attributes
        self._o3d_pcb = pcd
        print ("Sat up Open3D point cloud with", len(self._o3d_pcb.points), "points")

    def _read_file_in_chunks(self):
        orignal_header = self._get_original_las_header()
        print(f"Reading LAS file in chunks of {orignal_header.point_count} / {self._chunk_size}...")
        expected_number_of_chunks = self._las_header.point_count // self._chunk_size
        print(f"Expected number of chunks: {expected_number_of_chunks}")
        with laspy.open(self._las_path) as f:
            for index, points in enumerate(f.chunk_iterator(self._chunk_size)):
                print(f"Read chunk {index} with {len(points)} points")
                yield points

    def _update_pcb_and_attributes(self, indices):
        """
        Update the Open3D point cloud and attributes based on given indices.
        Args:
            indices: Indices of points to keep
        """
        self._o3d_pcb = self._o3d_pcb.select_by_index(indices)
        for dim, arr in self._attributes.items():
            self._attributes[dim] = arr[indices]

    def filter_noise(self, nb_neighbors=20, std_ratio=2.0):
        """
        Remove statistical outliers from the Open3D point cloud and update attributes accordingly.
        Args:
            nb_neighbors: Number of neighbors to analyze for each point.
            std_ratio: Standard deviation ratio threshold.
        """
        print(f"Filtering noise using statistical outlier removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        pcd = self._o3d_pcb
        # Get indices of inliers
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        # Filter Open3D point cloud
        self._update_pcb_and_attributes(ind)
        print(f"Noise filtering complete. Remaining points: {len(self._o3d_pcb.points)}")

    def write_to_file(self, output_path):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        print(f"Writing Open3D point cloud to {output_path} in chunks...")
        points = np.asarray(self._o3d_pcb.points)

        num_points = points.shape[0]
        chunk_size = self._chunk_size
        with laspy.open(output_path, mode="w", header=self._las_header) as writer:
            for index in range(0, num_points, chunk_size):
                chunk = points[index:index+chunk_size]
                las_chunk = laspy.LasData(self._las_header)
                las_chunk.x = chunk[:, 0]
                las_chunk.y = chunk[:, 1]
                las_chunk.z = chunk[:, 2]
                # Restore all other attributes
                for dim, arr in self._attributes.items():
                    setattr(las_chunk, dim, arr[index:index+chunk_size])
                writer.write_points(las_chunk.points)
                print(f"Wrote chunk {index//chunk_size} points to {output_path}")
        print(f"Finished writing Open3D point cloud to LAS file")