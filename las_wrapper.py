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
        self._chunk_size = chunk_size
        self._header = self._get_header_from_las_file(las_path)
        self._attributes = {}
        self._o3d_pcb = self._create_o3d_point_cloud(las_path)

    def _read_file_in_chunks(self):
        print(f"Reading LAS file in chunks {self._chunk_size} / {self._header.point_count}...")
        number_of_chunks = self._header.point_count // self._chunk_size
        print(f"Expected number of chunks: {number_of_chunks}")
        with laspy.open(self._las_path) as f:
            for index, points in enumerate(f.chunk_iterator(self._chunk_size)):
                print(f"Read chunk {index} with {len(points)} points")
                yield points

    def _get_header_from_las_file(self, las_path):
        with laspy.open(las_path) as las:
            return las.header

    def _create_o3d_point_cloud(self, las_path):
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
        return pcd

    def write_to_file(self, output_path):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        print(f"Writing Open3D point cloud to {output_path} in chunks, preserving all attributes...")
        points = np.asarray(self._o3d_pcb.points)
        header = laspy.LasHeader(point_format=self._header.point_format, version=self._header.version)
        header.offsets = self._header.offsets
        header.scales = self._header.scales

        num_points = points.shape[0]
        chunk_size = self._chunk_size
        with laspy.open(output_path, mode="w", header=header) as writer:
            for index in range(0, num_points, chunk_size):
                chunk = points[index:index+chunk_size]
                las_chunk = laspy.LasData(header)
                las_chunk.x = chunk[:, 0]
                las_chunk.y = chunk[:, 1]
                las_chunk.z = chunk[:, 2]
                # Restore all other attributes
                for dim, arr in self._attributes.items():
                    setattr(las_chunk, dim, arr[index:index+chunk_size])
                writer.write_points(las_chunk.points)
                print(f"Wrote chunk {index//chunk_size} points to {output_path}")
        print(f"Finished writing Open3D point cloud to LAS file")