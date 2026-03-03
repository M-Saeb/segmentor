import numpy as np
import open3d as o3d
from modules.utils import constants
from modules.utils.log_utils import get_dual_logger

log = get_dual_logger(__name__)

class PCDPreviewer:
    def __init__(self, pcd, points_attributes):
        self.pcd = pcd
        self.points_attributes = points_attributes

    def apply_compress(self, max_points=100_000):
        num_points = len(self.pcd.points)
        if num_points == 0:
            log.info("apply_compress: point cloud is empty; nothing to compress.")
            return

        # Target maximum number of points for interactive viewing.
        if num_points <= max_points:
            log.info(
                "apply_compress: %d points <= max_points (%d); skipping compression.",
                num_points,
                max_points,
            )
            return

        log.info(
            "apply_compress: downsampling from %d to %d points for display.",
            num_points,
            max_points,
        )

        indices = np.random.choice(num_points, size=max_points, replace=False)
        indices.sort()

        # Update point cloud geometry
        self.pcd = self.pcd.select_by_index(indices)

        # Update per-point attributes to stay in sync with geometry
        updated_attrs = {}
        for dim, arr in self.points_attributes.items():
            arr_np = np.asarray(arr)
            if len(arr_np) == num_points:
                updated_attrs[dim] = arr_np[indices]
            else:
                # Keep non per-point attributes as-is
                updated_attrs[dim] = arr

        self.points_attributes = updated_attrs
    
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
        
    def apply_classfication_colors(self):
        log.info("Displaying point cloud with classification-based coloring...")
        cls = self.points_attributes.get('classification', None)
        if cls is None or len(cls) != len(self.pcd.points):
            raise ValueError("Classification attribute lengths do not match point count")

        cls_arr = np.asarray(cls, dtype=int)
        color_map = {
            constants.PCB_CLASSIFICATION_INIT: np.array([0.5, 0.5, 0.5]),  # unclassified -> gray
            constants.PCB_CLASSIFICATION_GROUND: np.array([0.0, 1.0, 0.0]),  # ground -> green
            constants.PCB_CLASSIFICATION_WALL: np.array([0.0, 0.0, 1.0]),  # wall -> blue
        }

        cls_colors = np.empty((len(cls_arr), 3), dtype=float)
        for i, c in enumerate(cls_arr):
            if c in color_map:
                cls_colors[i] = color_map[c]
            else:
                # deterministic fallback color per class id
                ci = int(c)
                cls_colors[i] = np.array([
                    ((ci * 37) % 255) / 255.0,
                    ((ci * 61) % 255) / 255.0,
                    ((ci * 97) % 255) / 255.0,
                ])

        self.pcd.colors = o3d.utility.Vector3dVector(cls_colors)
