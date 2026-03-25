import os
import numpy as np
import open3d as o3d
from modules.models.ai_object_detector import (
    identify_object_from_views,
    merge_identified_objects_record,
)
from modules.utils.log_utils import get_dual_logger
from modules.utils import constants

log = get_dual_logger(__name__)

def create_histogram(array, num_bins=256):
    counts, bin_edges = np.histogram(array, bins=num_bins)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
    y = counts
    return x, y

def filter_noise(pcd, points_attributes, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from the Open3D point cloud and update attributes accordingly.
    Args:
        pcv: The Open3D point cloud to filter
        points_attributes: The dictionary of point attributes to filter
        nb_neighbors: Number of neighbors to analyze for each point.
        std_ratio: Standard deviation ratio threshold.
    """
    # Filter Open3D point cloud
    log.info("Apply filtering noise")
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    new_pcd = pcd.select_by_index(ind)
    new_points_attributes = {}
    for dim, arr in points_attributes.items():
        new_points_attributes[dim] = arr[ind]

    return new_pcd, new_points_attributes


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
        classification[ground_mask] = constants.PCB_CLASSIFICATION_GROUND
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

        if i == (max_iter - 1):
            log.warning("Cloth filter reached the maximum number of iterations: %d", max_iter)

    # classify ground: points close to cloth surface are ground
    cloth_lin = cloth.reshape(-1)
    point_cloth_z = cloth_lin[lin_idx]

    # threshold: points within a small height from cloth are ground
    threshold = max(0.3, resolution * threshold_factor)
    is_ground = (zs - point_cloth_z) <= threshold

    # Update classification using previous values
    classification = points_attributes['classification']
    classification[is_ground] = constants.PCB_CLASSIFICATION_GROUND
    points_attributes['classification'] = classification
    log.debug("Assigned classification: ground=%d, non-ground=%d", np.sum(is_ground), np.sum(~is_ground))
    
    return points_attributes


# Max deviation from horizontal (Z-up) for accepting the RANSAC plane as floor; not exposed as API.
_RANSAC_FLOOR_MAX_TILT_FROM_HORIZONTAL_DEG = 10.0


def ransac_floor_segmentation(
    points: np.ndarray,
    points_attributes,
    distance_threshold: float,
    num_iterations: int,
):
    """
    Segment ground by fitting one dominant plane with RANSAC. If the plane is
    approximately horizontal, inliers are classified as ground.

    Args:
        points: (N, 3) point coordinates.
        points_attributes: Must include ``classification`` aligned with ``points``.
        distance_threshold: Inlier distance to plane (Open3D ``segment_plane``).
        num_iterations: RANSAC iterations (``ransac_n`` is fixed at 3).

    Returns:
        Updated ``points_attributes``. Classification is unchanged if the plane
        is not horizontal enough.
    """
    log.debug(
        "RANSAC floor segmentation (distance_threshold=%.4f, num_iterations=%d)...",
        distance_threshold,
        num_iterations,
    )
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy.ndarray of shape (N,3)")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N,3)")
    if "classification" not in points_attributes:
        raise KeyError("points_attributes must contain 'classification'")

    classification = np.asarray(points_attributes["classification"])
    if classification.shape[0] != points.shape[0]:
        raise ValueError(
            "length of points_attributes['classification'] does not match number of points"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        log.warning("RANSAC floor: degenerate plane normal; ground classification unchanged.")
        return points_attributes
    normal = normal / norm_len

    angle_from_horizontal = float(
        np.degrees(np.arccos(np.clip(np.abs(normal[2]), -1.0, 1.0)))
    )
    if angle_from_horizontal > _RANSAC_FLOOR_MAX_TILT_FROM_HORIZONTAL_DEG:
        log.warning(
            "RANSAC floor: dominant plane is not horizontal enough (%.2f° > %.1f°); "
            "ground classification unchanged.",
            angle_from_horizontal,
            _RANSAC_FLOOR_MAX_TILT_FROM_HORIZONTAL_DEG,
        )
        return points_attributes

    inliers_arr = np.asarray(inliers, dtype=np.int64)
    classification = classification.copy()
    classification[inliers_arr] = constants.PCB_CLASSIFICATION_GROUND
    points_attributes["classification"] = classification
    log.info(
        "RANSAC floor: labeled %d points as ground (angle from horizontal: %.2f°).",
        len(inliers_arr),
        angle_from_horizontal,
    )
    return points_attributes


def segment_ground(
    pcd,
    points_attributes,
    method=None,
    threshold_factor=None,
    resolution=None,
    rigidness=None,
    max_iter=None,
    distance_threshold=None,
    num_iterations=None,
):
    """
    Segment ground/non-ground with optional parameter overrides.
    """
    log.info("Segmenting ground points using method: %s", method)

    if method == "basic":
        points_attributes = basic_ground_filter(
            np.asarray(pcd.points),
            points_attributes,
            z_threshold=None,
        )
    elif method == "csf":
        points_attributes = cloth_simulation_filter(
            np.asarray(pcd.points),
            points_attributes,
            threshold_factor=threshold_factor,
            resolution=resolution,
            rigidness=rigidness,
            max_iter=max_iter,
        )
    elif method == "ransac":
        if distance_threshold is None:
            distance_threshold = 0.05
        if num_iterations is None:
            num_iterations = 1000
        points_attributes = ransac_floor_segmentation(
            np.asarray(pcd.points),
            points_attributes,
            distance_threshold=distance_threshold,
            num_iterations=num_iterations,
        )
    else:
        log.error("Unknown ground filter method: %s", method)
        raise ValueError(f"Unknown ground filter method: {method}")

    return pcd, points_attributes


def apply_ransac_wall_detection(
    points: np.ndarray,
    points_attributes,
    epsilon: float = 0.01,
    min_support_points: int = 2000,
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

    log.info("Starting RANSAC Wall Segmentation on %d points...", len(points))

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
        mask_unclassified = classification[actual_indices] == constants.PCB_CLASSIFICATION_INIT
        if np.any(mask_unclassified):
            idx_to_set = actual_indices[mask_unclassified]
            classification[idx_to_set] = constants.PCB_CLASSIFICATION_WALL
            assigned_count += idx_to_set.size

    points_attributes['classification'] = classification

    log.info("Walls RANSAC Segmentation complete: %d planes found, %d walls detected, %d points labeled as WALL", 
             len(detected_planes), len(wall_indices), assigned_count)

    return points_attributes


def apply_ransac_ceiling_detection(
    points: np.ndarray,
    points_attributes,
    epsilon: float = 0.01,
    min_support_points: int = 2000,
    max_tilt_from_horizontal_deg: float = 10.0,
    max_iterations: int = 1000,
    min_plane_size_ratio: float = 0.01,
    min_height_above_ground: float = 2.0,
    max_height_above_ground: float = 10.0,
    ground_height: float | None = None,
):
    """
    Apply RANSAC plane detection for ceiling segmentation on a point cloud.

    A ceiling is modeled as a (near) horizontal plane above the ground level.
    Multiple ceiling planes can be detected (e.g., multi-floor garages).

    Parameters
    ----------
    points : np.ndarray
        3D numpy array of shape (N, 3) containing point cloud coordinates.
    points_attributes : dict-like
        Must contain a `classification` array-like with length N.
    epsilon : float
        Maximum distance from point to plane to be considered an inlier.
    min_support_points : int
        Minimum number of points required to form a valid plane.
    max_tilt_from_horizontal_deg : float
        Maximum allowed deviation from horizontal (Z-up) in degrees for ceiling classification.
    max_iterations : int
        Maximum number of RANSAC iterations per plane detection.
    min_plane_size_ratio : float
        Minimum ratio of remaining points to continue detection.
    min_height_above_ground : float
        Minimum allowed height above ground (in Z-units) for a ceiling plane.
    max_height_above_ground : float
        Maximum allowed height above ground (in Z-units) for a ceiling plane.
    ground_height : float | None
        Optional pre-computed ground reference height. If None, it will be
        inferred from ground-classified points when available, otherwise from
        the global minimum Z.
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

    zs = points[:, 2]

    # Pre-compute ground height information. When multiple ground levels exist
    # (e.g. multi-floor garages), we later choose a local ground height per
    # detected plane instead of a single global value.
    ground_z_values = None
    global_min_z = float(zs.min())
    if ground_height is None:
        ground_mask = classification == constants.PCB_CLASSIFICATION_GROUND
        if np.any(ground_mask):
            ground_z_values = zs[ground_mask].astype(float)
            log.debug(
                "Found %d ground-classified points for per-plane ground estimation.",
                ground_z_values.size,
            )
        else:
            log.debug(
                "No ground-classified points found; falling back to global minimum Z "
                "for per-plane ground estimation: %.3f",
                global_min_z,
            )

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    detected_planes = []
    plane_indices_list = []
    ceiling_plane_ids = []

    remaining_indices = np.arange(len(points))
    remaining_pcd = pcd

    plane_id = 0

    log.info("Starting RANSAC ceiling detection on %d points...", len(points))

    while len(remaining_indices) > len(points) * min_plane_size_ratio:
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=epsilon,
            ransac_n=3,
            num_iterations=max_iterations,
        )

        if len(inliers) < min_support_points:
            log.info(
                "Stopping ceiling detection: Found only %d inliers (minimum: %d)",
                len(inliers),
                min_support_points,
            )
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal = normal / np.linalg.norm(normal)

        # Horizontal planes have normals mostly aligned with Z axis
        angle_from_horizontal = np.degrees(np.arccos(np.clip(np.abs(normal[2]), -1.0, 1.0)))

        actual_indices = remaining_indices[inliers]
        plane_z_values = zs[actual_indices]
        mean_z = float(plane_z_values.mean())

        # Choose a local ground height for this plane:
        # - If an explicit ground_height is given, use it directly.
        # - Otherwise, if we have ground_z_values, pick the highest ground
        #   level not above the plane (or the closest overall if all are above).
        # - As a final fallback, use the global minimum Z.
        if ground_height is not None:
            local_ground_height = float(ground_height)
        elif ground_z_values is not None and ground_z_values.size > 0:
            below_or_equal = ground_z_values[ground_z_values <= mean_z]
            if below_or_equal.size > 0:
                local_ground_height = float(below_or_equal.max())
            else:
                # All ground points are above the plane; choose the closest one.
                idx_closest = np.argmin(np.abs(ground_z_values - mean_z))
                local_ground_height = float(ground_z_values[idx_closest])
        else:
            local_ground_height = global_min_z

        height_above_ground = mean_z - local_ground_height

        is_ceiling = (
            angle_from_horizontal <= max_tilt_from_horizontal_deg
            and
            min_height_above_ground <= height_above_ground <= max_height_above_ground
        )

        detected_planes.append(plane_model)
        plane_indices_list.append(actual_indices)
        if is_ceiling:
            ceiling_plane_ids.append(plane_id)

        log.debug(
            "Plane %d: %d points, angle from horizontal: %.2f°, "
            "mean_z: %.3f, height_above_ground: %.3f, is_ceiling: %s",
            plane_id,
            len(inliers),
            angle_from_horizontal,
            mean_z,
            height_above_ground,
            is_ceiling,
        )

        # Remove inliers from remaining points
        remaining_indices = np.delete(remaining_indices, inliers)
        if len(remaining_indices) == 0:
            break

        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(points[remaining_indices])

        plane_id += 1

    # Assign CEILING classification to points that belong to detected ceiling planes
    assigned_count = 0
    for pid in ceiling_plane_ids:
        actual_indices = plane_indices_list[pid]
        # only overwrite points that are still in INIT classification
        mask_unclassified = classification[actual_indices] == constants.PCB_CLASSIFICATION_INIT
        if np.any(mask_unclassified):
            idx_to_set = actual_indices[mask_unclassified]
            classification[idx_to_set] = constants.PCB_CLASSIFICATION_CEILING
            assigned_count += idx_to_set.size

    points_attributes['classification'] = classification

    log.info(
        "Ceiling RANSAC complete: %d planes found, %d ceilings detected, %d points labeled as CEILING",
        len(detected_planes),
        len(ceiling_plane_ids),
        assigned_count,
    )

    return points_attributes

def apply_wall_segmentation(
    pcd,
    points_attributes,
    epsilon: float = 0.01,
    min_support_points: int = 2000,
    max_normal_dev: float = 10.0,
    max_iterations: int = 1000,
    min_plane_size_ratio: float = 0.01,
):
    """
    Wrapper around apply_ransac_wall_detection that works directly with an
    Open3D PointCloud, similar to how segment_ground wraps cloth_simulation_filter().

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to segment.
    points_attributes : dict-like
        Per-point attributes dictionary.
    Other parameters are forwarded to apply_ransac_wall_detection.

    Returns
    -------
    (pcd, points_attributes) : tuple
        The original point cloud and updated attributes.
    """
    points = np.asarray(pcd.points)
    points_attributes = apply_ransac_wall_detection(
        points,
        points_attributes,
        epsilon=epsilon,
        min_support_points=min_support_points,
        max_normal_dev=max_normal_dev,
        max_iterations=max_iterations,
        min_plane_size_ratio=min_plane_size_ratio,
    )
    return pcd, points_attributes


def apply_ceiling_segmentation(
    pcd,
    points_attributes,
    epsilon: float = 0.01,
    min_support_points: int = 2000,
    max_tilt_from_horizontal_deg: float = 10.0,
    max_iterations: int = 1000,
    min_plane_size_ratio: float = 0.01,
    min_height_above_ground: float = 2.0,
    max_height_above_ground: float = 10.0,
):
    """
    Wrapper around apply_ransac_ceiling_detection that works directly with an
    Open3D PointCloud, similar to how segment_ground and apply_wall_segmentation
    operate.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to segment.
    points_attributes : dict-like
        Per-point attributes dictionary.
    Other parameters are forwarded to apply_ransac_ceiling_detection.

    Returns
    -------
    (pcd, points_attributes) : tuple
        The original point cloud and updated attributes.
    """
    points = np.asarray(pcd.points)
    points_attributes = apply_ransac_ceiling_detection(
        points,
        points_attributes,
        epsilon=epsilon,
        min_support_points=min_support_points,
        max_tilt_from_horizontal_deg=max_tilt_from_horizontal_deg,
        max_iterations=max_iterations,
        min_plane_size_ratio=min_plane_size_ratio,
        min_height_above_ground=min_height_above_ground,
        max_height_above_ground=max_height_above_ground,
        ground_height=None,
    )
    return pcd, points_attributes


def filter_noise_for_none_classified_points(pcd, points_attributes, nb_neighbors=20, std_ratio=2.0):
    """
    Apply statistical noise filtering only to points that are still in the
    initial (unclassified) state, keeping all classified points untouched.

    Args:
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
        return filter_noise(pcd, points_attributes, nb_neighbors, std_ratio)

    classification = np.asarray(classification)
    unclassified_mask = classification == constants.PCB_CLASSIFICATION_INIT
    if not np.any(unclassified_mask):
        log.info("No unclassified points to filter; returning original point cloud.")
        return pcd, points_attributes

    log.info("Filtering noise on %d unclassified points", np.sum(unclassified_mask))

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


def _dbscan_cluster_mask(labels: np.ndarray) -> np.ndarray:
    """True where Open3D DBSCAN assigned a cluster id (>= 0); False for noise (-1)."""
    return np.asarray(labels, dtype=np.int64) >= 0


def segment_objects_dbscan(pcd, points_attributes, eps, min_points):
    """
    Run DBSCAN on points still classified as INIT only; assign PCB_CLASSIFICATION_OBJECTS
    to all points that belong to a cluster. Noise points (-1) stay INIT.

    Neighbors are computed only within the unclassified subset so clusters cannot
    bridge through ground, walls, or ceiling.

    Sets object_cluster_id (1..K per cluster) when clusters exist; removed by segment_objects
    before downstream use. Cleared at start if present from a previous run.
    """
    points_attributes.pop("object_cluster_id", None)
    classification = points_attributes.get("classification", None)
    if classification is None:
        log.warning(
            "segment_objects_dbscan: missing 'classification'; skipping object segmentation."
        )
        return pcd, points_attributes

    classification = np.asarray(classification)
    num_points = len(pcd.points)
    if classification.shape[0] != num_points:
        log.warning(
            "segment_objects_dbscan: classification length %d != num points %d; skipping.",
            classification.shape[0],
            num_points,
        )
        return pcd, points_attributes

    unclassified_mask = classification == constants.PCB_CLASSIFICATION_INIT
    n_unclassified = int(np.sum(unclassified_mask))
    if n_unclassified == 0:
        log.info("No unclassified points for DBSCAN object segmentation; skipping.")
        return pcd, points_attributes

    unclassified_indices = np.where(unclassified_mask)[0]
    subset_pcd = pcd.select_by_index(unclassified_indices)
    labels = np.asarray(
        subset_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    clustered_mask = _dbscan_cluster_mask(labels)
    n_clustered = int(np.sum(clustered_mask))
    n_noise = int(np.sum(~clustered_mask))
    log.info(
        "DBSCAN on %d unclassified points: %d clustered, %d noise (-1).",
        n_unclassified,
        n_clustered,
        n_noise,
    )
    if n_clustered == 0:
        return pcd, points_attributes

    global_object_indices = unclassified_indices[clustered_mask]
    classification = classification.copy()
    classification[global_object_indices] = constants.PCB_CLASSIFICATION_OBJECTS

    # Sequential cluster ids 1..K for each DBSCAN cluster (Open3D labels may be non-contiguous).
    object_cluster_id = np.zeros(num_points, dtype=np.int32)
    raw_to_seq = {}
    seq = 1
    for raw in sorted(np.unique(labels[labels >= 0])):
        raw_to_seq[int(raw)] = seq
        seq += 1
    for i, gidx in enumerate(unclassified_indices):
        lab = int(labels[i])
        if lab >= 0:
            object_cluster_id[gidx] = raw_to_seq[lab]

    points_attributes["classification"] = classification
    points_attributes["object_cluster_id"] = object_cluster_id
    return pcd, points_attributes


def take_isometric_screenshots(pcd, dir_name, file_prefix=""):
    """
    Save front and back isometric PNGs of the point cloud (RGB via pcd.colors).

    If file_prefix is empty, writes:
        {dir_name}/front_isometric.png, {dir_name}/back_isometric.png
    Otherwise:
        {dir_name}/{file_prefix}_front_isometric.png, etc.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    width, height = 640, 480
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("cloud", pcd, mat)

    center = np.asarray(pcd.get_center(), dtype=float)
    bounds = pcd.get_axis_aligned_bounding_box()
    diag = float((bounds.get_max_bound() - bounds.get_min_bound()).max())

    radius = max(1e-6, diag * 1.0)
    up = [0.0, 0.0, 1.0]

    elev_deg = 35.0
    elev = np.deg2rad(elev_deg)
    z_off = radius * np.sin(elev)
    horiz_r = radius * np.cos(elev)

    views = {
        "front_isometric": 225.0,
        "back_isometric": 45.0,
    }

    for name, az_deg in views.items():
        theta = np.deg2rad(az_deg)
        x_off = horiz_r * np.cos(theta)
        y_off = horiz_r * np.sin(theta)
        eye = center + np.array([x_off, y_off, z_off], dtype=float)

        renderer.setup_camera(60.0, center, eye, up)
        img = renderer.render_to_image()
        if file_prefix:
            fname = f"{file_prefix}_{name}.png"
        else:
            fname = f"{name}.png"
        o3d.io.write_image(os.path.join(dir_name, fname), img)
        log.debug("Saved %s/%s", dir_name, fname)


def take_screenshot(pcd, dir_name):
    """
    Save two screenshots of the point cloud, approximating
    CloudCompare's "front isometric" and "back isometric" views.

    Images are saved as:
        {dir_name}/front_isometric.png
        {dir_name}/back_isometric.png
    """
    take_isometric_screenshots(pcd, dir_name, file_prefix="")


def _apply_rgb_colors_to_pcd(pcd, points_attributes):
    """Set pcd.colors from LAS red/green/blue (same normalization as PCDPreviewer)."""
    red = np.asarray(points_attributes.get("red"))
    green = np.asarray(points_attributes.get("green"))
    blue = np.asarray(points_attributes.get("blue"))
    n = len(pcd.points)
    if not (len(red) == len(green) == len(blue) == n):
        raise ValueError("RGB attribute lengths do not match point count")
    rgb = np.vstack((red, green, blue)).T.astype(float)
    max_val = np.max(rgb)
    if max_val > 0:
        rgb /= max_val
    pcd.colors = o3d.utility.Vector3dVector(rgb)


def mask_indices_for_object_vlm_view(classification, object_cluster_id, cluster_id):
    """
    Ground + target cluster + unclassified (INIT) points only (excludes other clusters, walls, ceiling).
    """
    classification = np.asarray(classification)
    object_cluster_id = np.asarray(object_cluster_id, dtype=np.int32)
    ground = classification == constants.PCB_CLASSIFICATION_GROUND
    # init_pts = classification == constants.PCB_CLASSIFICATION_INIT
    this_object = object_cluster_id == cluster_id
    # combined = ground | init_pts | this_object
    combined = ground | this_object
    return np.where(combined)[0]


def segment_objects(
    pcd,
    points_attributes,
    eps,
    min_points,
    *,
    output_folder_path,
    chunk_index=0,
    ollama_model="gemma3:4b",
    vlm_traces_subdir="vlm_object_traces",
    enable_vlm=True,
    **kwargs,
):
    """
    DBSCAN object segmentation, optionally followed by per-cluster RGB screenshots and VLM labeling.

    When enable_vlm is False (e.g. setup GUI preview), only DBSCAN runs; no screenshots or Ollama.
    Removes internal object_cluster_id before return (classification unchanged for objects).
    """
    pcd, points_attributes = segment_objects_dbscan(
        pcd, points_attributes, eps, min_points
    )

    oc_id = points_attributes.get("object_cluster_id")
    if not enable_vlm:
        points_attributes.pop("object_cluster_id", None)
        return pcd, points_attributes

    traces_dir = os.path.join(output_folder_path, vlm_traces_subdir)
    os.makedirs(traces_dir, exist_ok=True)

    if oc_id is None:
        return pcd, points_attributes

    oc_id = np.asarray(oc_id, dtype=np.int32)
    unique_clusters = sorted(int(c) for c in np.unique(oc_id) if c > 0)
    if not unique_clusters:
        points_attributes.pop("object_cluster_id", None)
        return pcd, points_attributes

    cls_arr = np.asarray(points_attributes["classification"])

    for cid in unique_clusters:
        try:
            indices = mask_indices_for_object_vlm_view(cls_arr, oc_id, cid)
            if len(indices) == 0:
                log.warning("segment_objects: empty view for cluster %s; skipping", cid)
                continue

            view_pcd = pcd.select_by_index(indices)
            view_attrs = {}
            npt = len(cls_arr)
            for k, v in points_attributes.items():
                arr = np.asarray(v)
                if arr.shape[0] == npt:
                    view_attrs[k] = arr[indices]
                else:
                    view_attrs[k] = v

            try:
                _apply_rgb_colors_to_pcd(view_pcd, view_attrs)
            except ValueError as e:
                log.warning(
                    "segment_objects: RGB missing for cluster %s (%s); skipping VLM.",
                    cid,
                    e,
                )
                continue

            prefix = f"chunk{chunk_index:05d}_object{cid}"
            take_isometric_screenshots(view_pcd, traces_dir, file_prefix=prefix)
            front_name = f"{prefix}_front_isometric.png"
            back_name = f"{prefix}_back_isometric.png"
            front_path = os.path.join(traces_dir, front_name)
            back_path = os.path.join(traces_dir, back_name)

            try:
                record = identify_object_from_views(
                    front_path,
                    back_path,
                    cluster_id=cid,
                    chunk_index=chunk_index,
                    ollama_model=ollama_model,
                )
                merge_identified_objects_record(output_folder_path, record)
            except Exception as ex:
                log.warning(
                    "segment_objects: VLM/YAML failed for cluster %s: %s",
                    cid,
                    ex,
                    exc_info=True,
                )
        except Exception as ex:
            log.warning(
                "segment_objects: failed processing cluster %s: %s",
                cid,
                ex,
                exc_info=True,
            )

    points_attributes.pop("object_cluster_id", None)
    return pcd, points_attributes