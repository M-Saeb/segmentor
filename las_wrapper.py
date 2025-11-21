import laspy

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
        self._header = self._get_las_file_header(las_path)

    def _read_file_in_chunks(self):
        print(f"Reading LAS file in chunks {self._chunk_size} / {self._header.point_count}...")
        number_of_chunks = self._header.point_count // self._chunk_size
        print(f"Expected number of chunks: {number_of_chunks}")
        with laspy.open(self._las_path) as f:
            for index, points in enumerate(f.chunk_iterator(self._chunk_size)):
                print(f"Read chunk {index} with {len(points)} points")
                yield points

    def _get_las_file_header(self, las_path):
        with laspy.open(las_path) as las:
            return las.header

    def write_to_file(self, output_path):
        """
        Write the LAS data to a new file.
        Args:
            output_path: Path to save the new .las file
        """
        print(f"Writing LAS data to {output_path}...")
        with laspy.open(output_path, mode="w", header=self._header) as writer:
            for index, points in enumerate(self._read_file_in_chunks()):
                writer.write_points(points)
                print(f"Wrote chunk {index} points to {output_path}")
        print(f"Finished writing LAS data")