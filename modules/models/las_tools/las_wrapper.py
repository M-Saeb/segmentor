import numpy as np
from modules.utils import log_utils, config, constants
from .las_snapshot import LasSnapshot
from .las_iterator import LasIteratorAbstract, LasIteratorBasic
from . import las_io

log = log_utils.get_dual_logger(__name__)

class LasWrapper:

    def __init__(
        self,
        cfg: config.SegmentorConfig | None = None,
        iterator_class: type[LasIteratorAbstract] = None,
    ):
        """
        High-level wrapper around LAS I/O and chunk iteration.

        Args:
            cfg: Parsed `SegmentorConfig` holding input path, chunk size.
        """
        if cfg is None:
            cfg = config.SegmentorConfig.get_instance()
        self.input_file = cfg.input_file
        self.chunk_size = cfg.chunk_size
        self.reset_classification = cfg.reset_classification
        # Preview navigation state (loaded lazily via iterate_chunk()).
        if iterator_class is None:
            self._las_iterator = LasIteratorBasic(self.input_file, self.chunk_size)
        else:
            self._las_iterator = iterator_class(self.input_file, self.chunk_size)
        self.current_chunk = None
        self.snapshot = LasSnapshot(self.input_file)
        log.debug("Chunk size set to: %d", self.chunk_size)

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

    def _create_new_las_filer_header(self):
        original_header = self._get_original_las_header()
        new_header = las_io.create_new_las_header(original_header)
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
        self.snapshot.save(pcd, points_attributes)

    def restore_snapshot(self):
        try:
            pcd, points_attributes = self.snapshot.get()
        except FileNotFoundError:
            log.warning("Snapshot file not found. Saving new snapshot.")
            self.save_snapshot()
            pcd, points_attributes = self.snapshot.get()
        self.set_current_chunk(pcd, points_attributes)

    def get_snapshot(self):
        return self.snapshot.get()

    def delete_snapshot(self):
        self.snapshot.delete()

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
        las_io.init_las_file(
            full_path,
            las_io.create_pipeline_output_las_header(
                las_io.get_original_las_header(self.input_file)
            )
        )

    def write_pcb_chunk_to_file(self, index, pcd, points_attributes, full_path):
        """
        Write the Open3D point cloud and all original attributes to a new LAS file.
        Args:
            output_path: Path to save the new .las file
        """
        log.info("vvv Writing chunk %d vvv", index)
        las_io.write_chunk_las_file(
            pcd,
            points_attributes,
            full_path,
            las_io.create_pipeline_output_las_header(
                las_io.get_original_las_header(self.input_file)
            )
        )
        log.info("^^^ Wrote chunk %d points ^^^", index)
