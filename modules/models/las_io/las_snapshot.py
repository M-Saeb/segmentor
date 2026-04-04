import tempfile
import random
import os
import numpy as np
import laspy
import open3d as o3d
from modules.utils import log_utils

log = log_utils.get_dual_logger(__name__)

class LasSnapshot:
    def __init__(self, original_header: laspy.LasHeader):
        self.id = random.randint(0, 10000)
        self.path = tempfile.mktemp(prefix="snapshot_", suffix=f"_{self.id}.las")
        self.las_header = self._create_new_las_header(original_header)


    def _create_new_las_header(self, original_header):
        new_header = laspy.LasHeader(
            point_format=original_header.point_format,
            version=original_header.version
        )
        new_header.offsets = original_header.offsets
        new_header.scales = original_header.scales
        return new_header

    def get_snapshot(self):
        with laspy.open(self.path) as reader:
            las_data = reader.read()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack((las_data.x, las_data.y, las_data.z)).T)
            points_attributes = {}
            for dim in las_data.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    points_attributes[dim] = getattr(las_data, dim)
            return pcd, points_attributes

    def save_snapshot(self, pcd, points_attributes):
        points = np.asarray(pcd.points)
    
        with laspy.open(self.path, mode="w", header=self.las_header) as writer:
            las_data = laspy.LasData(self.las_header)
            las_data.x = points[:, 0]
            las_data.y = points[:, 1]
            las_data.z = points[:, 2]

            for dim, arr in points_attributes.items():
                setattr(las_data, dim, arr)

            writer.write_points(las_data.points)
            log.debug("Wrote snapshot to %s", self.path)

    def restore_snapshot(self):
        with laspy.open(self.path) as reader:
            las_data = reader.read()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack((las_data.x, las_data.y, las_data.z)).T)
            points_attributes = {}
            for dim in las_data.point_format.dimension_names:
                if dim not in ('x', 'y', 'z'):
                    points_attributes[dim] = getattr(las_data, dim)
            return pcd, points_attributes

    def delete_snapshot(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            log.info("Deleted snapshot %s", self.path)
        else:
            log.warning("Snapshot file does not exist")