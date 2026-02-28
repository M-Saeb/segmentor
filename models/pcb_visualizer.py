import numpy as np
import open3d as o3d
from .utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

class PCBVisualizer:
    def __init__(self, pcd, points_attributes):
        self.pcd = pcd
        self.points_attributes = points_attributes

    
    def apply_rgb_colors(self):
        log.info("Displaying point cloud with RGB-based coloring...")
        red = self.points_attributes.get('red', None)
        green = self.points_attributes.get('green', None)
        blue = self.points_attributes.get('blue', None)

        red = np.asarray(red)
        green = np.asarray(green)
        blue = np.asarray(blue)

        if not (len(red) == len(green) == len(blue) == len(self.pcd.points)):
            raise ValueError("RGB attribute lengths do not match point count; falling back to classification coloring.")

        rgb = np.vstack((red, green, blue)).T.astype(float)
        max_val = np.max(rgb)
        if max_val > 0:
            rgb /= max_val  # normalize to [0,1]

        self.pcd.colors = o3d.utility.Vector3dVector(rgb)
        