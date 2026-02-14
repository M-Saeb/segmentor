import numpy as np
import open3d as o3d
from  utils import log_utils
from .classses import INIT_CLASSIFICATION, GROUND_CLASSIFICATION, WALL_CLASSIFICATION

log = log_utils.get_dual_logger(__name__)

def create_histogram(array, num_bins=256):
    counts, bin_edges = np.histogram(array, bins=num_bins)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
    y = counts
    return x, y



def find_peak_and_drops(x, y, drop_threshold=0.10):
    peak_idx = np.argmax(y)

    # Find left drop
    left_idx = 0
    for i in range(0, peak_idx):
        if y[i] < y[peak_idx] * drop_threshold:
            left_idx = i
            break

    # Find right drop
    right_idx = peak_idx
    for i in range(peak_idx, len(y)):
        if y[i] < y[peak_idx] * drop_threshold:
            right_idx = i
            break

    return peak_idx, left_idx, right_idx

def basic_ground_filter(points: np.ndarray, points_attributes, z_threshold):
        """
        Segment ground/non-ground by analyzing Z coordinate distribution and assign classification numbers.
        Steps:
        1. Export Z to ScalarField
        2. Find start/end of peak in Z histogram
        3. Split cloud at peak end
        4. Assign classification numbers
        """
        log.debug("Filtering ground using basic ground filter...")
        z_values = points[:, 2]
        # Step 1: Create Z histogram (to make it usable)
        x, y = create_histogram(z_values)
        # Step 2: Find peak in Z histogram
        peak_idx, left_idx, right_idx = find_peak_and_drops(x, y)
        log.info("Peak at Z=%.2f, threshold at Z=%.2f", x[peak_idx], x[right_idx])
        # Step 3: Split cloud
        ground_mask = z_values <= x[right_idx]
        # Step 4: Update classification using previous values
        classification = points_attributes['classification']
        classification[ground_mask] = GROUND_CLASSIFICATION
        points_attributes['classification'] = classification
        log.debug("Assigned classification: ground=%d, non-ground=%d", np.sum(ground_mask), np.sum(~ground_mask))
        return points_attributes

def cloth_simulation_filter(points: np.ndarray, points_attributes, threshold_factor=0.1, 
                            resolution=1.0, rigidness=3, max_iter=500):
    """Segment ground points from a point cloud using a simplified Cloth Simulation Filter and assign classification.

    Args:
        points (np.ndarray): Nx3 array of points (x, y, z).
        points_attributes (dict): Dictionary of point attributes including 'classification'.
        threshold_factor (float): factor to determine ground threshold distance from cloth.
        resolution (float): grid cell size in same units as `points`.
        rigidness (int): controls cloth stiffness (higher -> stiffer -> less smoothing).
        max_iter (int): maximum number of simulation iterations.

    Returns:
        dict: Updated points_attributes with classification assigned.
    """
    log.debug(
        "Filtering ground using Cloth Simulation Filter ( threshold_factor=%.2f, resolution=%.2f, rigidness=%d, max_iter=%d)...",
        threshold_factor, resolution, rigidness, max_iter
    )
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy.ndarray of shape (N,3)")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N,3)")

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    nx = int(np.ceil((x_max - x_min) / resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / resolution)) + 1

    # Map points to grid cells
    ix = np.floor((xs - x_min) / resolution).astype(int)
    iy = np.floor((ys - y_min) / resolution).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    lin_idx = iy * nx + ix

    # Build DEM: minimum z per cell
    cell_count = nx * ny
    dem = np.full(cell_count, np.inf, dtype=float)
    np.minimum.at(dem, lin_idx, zs)
    # replace empty cells with a large value (max z)
    max_z = float(zs.max())
    min_z = float(zs.min())
    dem[np.isinf(dem)] = max_z
    dem = dem.reshape((ny, nx))

    # Initialize cloth above highest point
    height_range = max_z - min_z
    cloth = np.full((ny, nx), max_z + max(1.0, height_range), dtype=float)

    # Prepare neighbor averaging helpers (4-neighbors)
    def neighbor_average(arr):
        # pad and compute neighbor average over up/down/left/right
        padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        return (up + down + left + right) / 4.0

    # Simulation loop
    for i in range(max_iter):
        prev = cloth.copy()

        # smoothing strength: smaller when rigidness is larger
        weight = 1.0 / max(1.0, float(rigidness))
        neigh = neighbor_average(cloth)
        cloth = cloth * (1.0 - weight) + neigh * weight

        # gravity: move cloth toward the terrain (DEM); move fraction of distance each iter
        gap = cloth - dem
        # ensure gap is non-negative before using it
        gap = np.maximum(gap, 0.0)
        cloth = cloth - 0.5 * gap

        # cloth must not go below the terrain
        cloth = np.maximum(cloth, dem)

        max_change = np.max(np.abs(cloth - prev))
        if max_change < 1e-3:
            break

    # classify ground: points close to cloth surface are ground
    cloth_lin = cloth.reshape(-1)
    point_cloth_z = cloth_lin[lin_idx]

    # threshold: points within a small height from cloth are ground
    threshold = max(0.3, resolution * threshold_factor)
    is_ground = (zs - point_cloth_z) <= threshold

    # Update classification using previous values
    classification = points_attributes['classification']
    classification[is_ground] = GROUND_CLASSIFICATION
    points_attributes['classification'] = classification
    log.debug("Assigned classification: ground=%d, non-ground=%d", np.sum(is_ground), np.sum(~is_ground))
    
    return points_attributes

def apply_ransac_wall_detection(
    points: np.ndarray,
    points_attributes,
    epsilon: float = 0.001,
    min_support_points: int = 20000,
    max_normal_dev: float = 10.0,
    max_iterations: int = 1000,
    min_plane_size_ratio: float = 0.01
):
    """
    Apply RANSAC plane detection for wall segmentation on a point cloud.
    
    Parameters:
    -----------
    points : np.ndarray
        3D numpy array of shape (N, 3) containing point cloud coordinates
    points_attributes : dict-like
        Must contain a `classification` array-like with length N
    epsilon : float
        Maximum distance from point to plane to be considered an inlier (default: 0.001)
    min_support_points : int
        Minimum number of points required to form a valid plane (default: 20000)
    max_normal_dev : float
        Maximum allowed deviation from vertical in degrees for wall classification (default: 10.0)
    max_iterations : int
        Maximum number of RANSAC iterations per plane detection (default: 1000)
    min_plane_size_ratio : float
        Minimum ratio of remaining points to continue detection (default: 0.01)

    Returns:
    --------
    dict
        The updated `points_attributes` with `WALL_CLASSIFICATION` assigned where applicable.
    """

    # input validation
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy.ndarray of shape (N,3)")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N,3)")
    if 'classification' not in points_attributes:
        raise KeyError("points_attributes must contain 'classification'")

    classification = np.asarray(points_attributes['classification'])
    if classification.shape[0] != points.shape[0]:
        raise ValueError("length of points_attributes['classification'] does not match number of points")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Results storage (kept for logging/debugging)
    detected_planes = []
    plane_points_list = []
    plane_indices_list = []
    wall_indices = []

    # Keep track of remaining points
    remaining_indices = np.arange(len(points))
    remaining_pcd = pcd

    plane_id = 0

    log.info("Starting RANSAC detection on %d points...", len(points))

    while len(remaining_indices) > len(points) * min_plane_size_ratio:
        # Apply RANSAC to detect one plane
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=epsilon,
            ransac_n=3,
            num_iterations=max_iterations
        )

        # Check if enough inliers found
        if len(inliers) < min_support_points:
            log.info("Stopping: Found only %d inliers (minimum: %d)", len(inliers), min_support_points)
            break

        # Extract plane parameters [a, b, c, d]
        a, b, c, d = plane_model

        # Normalize the normal vector
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Check if this is a wall (vertical plane)
        # A vertical plane has a normal vector mostly parallel to XY plane
        # i.e., small Z component
        angle_from_vertical = np.abs(np.arcsin(normal[2])) * 180 / np.pi
        is_wall = angle_from_vertical <= max_normal_dev

        # Get actual point indices in original cloud
        actual_indices = remaining_indices[inliers]

        # Store results
        detected_planes.append(plane_model)
        plane_points_list.append(points[actual_indices])
        plane_indices_list.append(actual_indices)

        if is_wall:
            wall_indices.append(plane_id)

        log.debug("Plane %d: %d points, angle from vertical: %.2f°, is_wall: %s", 
                  plane_id, len(inliers), angle_from_vertical, is_wall)

        # Remove inliers from remaining points
        remaining_indices = np.delete(remaining_indices, inliers)

        if len(remaining_indices) == 0:
            break

        # Create new point cloud with remaining points
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(points[remaining_indices])

        plane_id += 1

    # Assign WALL_CLASSIFICATION to points that belong to detected wall planes
    assigned_count = 0
    for pid in wall_indices:
        actual_indices = plane_indices_list[pid]
        # only overwrite points that are still in INIT_CLASSIFICATION
        mask_unclassified = classification[actual_indices] == INIT_CLASSIFICATION
        if np.any(mask_unclassified):
            idx_to_set = actual_indices[mask_unclassified]
            classification[idx_to_set] = WALL_CLASSIFICATION
            assigned_count += idx_to_set.size

    points_attributes['classification'] = classification

    log.info("RANSAC complete: %d planes found, %d walls detected, %d points labeled as WALL", 
             len(detected_planes), len(wall_indices), assigned_count)

    return points_attributes