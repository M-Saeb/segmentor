import tempfile
import random
import os
from modules.utils import log_utils
from . import las_io

log = log_utils.get_dual_logger(__name__)

class LasSnapshot:
    def __init__(self, file_path: str):
        self.id = random.randint(0, 10000)
        self.path = tempfile.mktemp(prefix="snapshot_", suffix=f"_{self.id}.las")
        og_header = las_io.get_original_las_header(file_path)
        self.las_header = las_io.create_new_las_header(og_header)

    def get(self):
        return las_io.read_full_las_file(self.path)

    def save(self, pcd, points_attributes):
        las_io.write_full_las_file(pcd, points_attributes, self.path, self.las_header)
        log.debug("Wrote snapshot to %s", self.path)

    def delete(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            log.info("Deleted snapshot %s", self.path)
        else:
            log.warning("Snapshot file does not exist")