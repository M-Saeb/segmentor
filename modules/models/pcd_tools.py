import os
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from modules.models.clusters_manager import ClustersManager
from modules.utils.log_utils import get_dual_logger
from modules.utils import constants

log = get_dual_logger(__name__)

def get_inlier_density(point_array, inlier_indices, plane_model):
    a, b, c, d = plane_model

    # Get the inlier points as a numpy array
    inlier_points = point_array[inlier_indices]

    # --- Project points onto the plane's local 2D coordinate system ---
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Build two orthogonal axes on the plane surface
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project each point onto (u, v) axes → 2D coordinates
    points_2d = np.column_stack([
        inlier_points @ u,
        inlier_points @ v
    ])

    # --- Compute convex hull area ---
    hull = ConvexHull(points_2d)
    area = hull.volume  # in 2D, .volume gives the area

    density = len(inlier_indices) / area

    log.debug(
        "get_inlier_density() => density: %s, area: %s, inlier_indices: %s",
        density,
        area,
        len(inlier_indices),
    )
    return density

def create_histogram(array, num_bins=256):
    counts, bin_edges = np.histogram(array, bins=num_bins)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
    y = counts
    return x, y

def is_plane_facing_up(plane, threshold=0.95):
    a, b, c, d = plane
    normal = np.array([a, b, c])
    
    normal_unit = normal / np.linalg.norm(normal)

    z_component = normal_unit[2]
    result= abs(z_component) > threshold
    log.debug(
        "is_plane_facing_up() => z_component: %s, threshold: %s, result: %s",
        z_component,
        threshold,
        result,
    )
    return result

def is_plane_facing_side(plane, threshold=0.9):
    """True if the plane normal is nearly horizontal in XY (vertical wall), matching sandbox ``is_facing_side``."""
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=float)
    normal = normal / np.linalg.norm(normal)
    return abs(normal[2]) < (1.0 - threshold)


def planes_footprint_overlap_xy(pcd_a, pcd_b, padding=0.0, min_overlap_area_m2=0.0):
    pts_a = np.asarray(pcd_a.points)
    pts_b = np.asarray(pcd_b.points)
    if pts_a.size == 0 or pts_b.size == 0:
        return False
    xy_a = pts_a[:, :2]
    xy_b = pts_b[:, :2]
    min_a, max_a = xy_a.min(axis=0), xy_a.max(axis=0)
    min_b, max_b = xy_b.min(axis=0), xy_b.max(axis=0)
    min_a = min_a - padding
    max_a = max_a + padding
    min_b = min_b - padding
    max_b = max_b + padding
    x0 = max(min_a[0], min_b[0])
    x1 = min(max_a[0], max_b[0])
    y0 = max(min_a[1], min_b[1])
    y1 = min(max_a[1], max_b[1])
    if x0 > x1 or y0 > y1:
        return False
    area = (x1 - x0) * (y1 - y0)
    result = area >= min_overlap_area_m2
    log.debug(
        "planes_footprint_overlap_xy() => area: %s, min_overlap_area_m2: %s, result: %s",
        area,
        min_overlap_area_m2,
        result,
    )
    return result

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

def _cloth_simulation_filter(points: np.ndarray, points_attributes, threshold_factor=0.1, 
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


def _ransac_floor_segmentation(
    pcd,
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

    def is_floor_check(plane_model, all_floor_planes, z_histogram):
        current_plane_z_intercept = -plane_model[3] / plane_model[2]   # -d1 / c1
        if not is_plane_facing_up(plane_model):
            return False
        # check if the plane is within the first half of the z_histogram
        is_within_first_half = current_plane_z_intercept < z_histogram[0][len(z_histogram[0])//2]
        if not is_within_first_half:
            return False

        for item in all_floor_planes:
            floor_plane_normal = item["plane_model"]
            floor_plane_z_intercept = -floor_plane_normal[3] / floor_plane_normal[2]   # -d1 / c1
            z_difference = abs(current_plane_z_intercept - floor_plane_z_intercept)
            is_above = z_difference > 0.2
            if not is_above:
                continue
            is_colliding = planes_footprint_overlap_xy(
                pcd, item["pcd"],
                min_overlap_area_m2=1
            )
            if is_colliding:
                return False
        return True

    floor_planes = []
    remain_pcd = pcd
    classification = points_attributes["classification"]
    n = len(np.asarray(pcd.points))
    orig_indices = np.arange(n, dtype=np.int64)
    z_values = np.asarray(pcd.points)[:, 2]
    z_histogram = create_histogram(z_values)
    for i in range(num_iterations):
        if len(remain_pcd.points) <= 3:
            break

        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )

        inliers_arr = np.asarray(inliers, dtype=np.int64)
        is_floor = is_floor_check(plane_model, floor_planes, z_histogram)

        if is_floor:
            classification[orig_indices[inliers_arr]] = constants.PCB_CLASSIFICATION_GROUND
            floor_planes.append({
                "plane_model": plane_model,
                "pcd": remain_pcd.select_by_index(inliers),
            })

        if len(remain_pcd.points) <= 3:
            log.debug("RANSAC floor segmentation: no more points to segment")
            break

        keep_mask = np.ones(len(orig_indices), dtype=bool)
        keep_mask[inliers_arr] = False
        orig_indices = orig_indices[keep_mask]
        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)

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

    if method == "csf":
        points_attributes = _cloth_simulation_filter(
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
        points_attributes = _ransac_floor_segmentation(
            pcd,
            points_attributes,
            distance_threshold=distance_threshold,
            num_iterations=num_iterations,
        )
    else:
        log.error("Unknown ground filter method: %s", method)
        raise ValueError(f"Unknown ground filter method: {method}")

    return pcd, points_attributes


def _apply_ransac_wall_detection(
    points: np.ndarray,
    points_attributes,
    rotation_threshold: float = 0.9,
):
    """
    Segment walls by repeatedly fitting dominant planes with RANSAC (same loop
    structure as ``ransac_floor_segmentation`` and sandbox ``segment_walls``).
    Inliers on nearly-vertical planes label INIT points as wall.

    ``distance_threshold`` (0.1) and inner ``segment_plane`` iteration count
    (5000) match sandbox ``segment_walls``. Wall acceptance uses
    ``is_plane_facing_side`` with ``threshold=rotation_threshold`` (sandbox
    ``is_facing_side``).

    Args:
        points: (N, 3) point coordinates.
        points_attributes: Must include ``classification`` aligned with ``points``.
        rotation_threshold: Passed to ``is_plane_facing_side``; higher values
            require normals closer to the horizontal XY plane (stricter vertical wall).

    Returns:
        Updated ``points_attributes``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, 3)")
    if "classification" not in points_attributes:
        raise KeyError("points_attributes must contain 'classification'")

    classification = np.asarray(points_attributes["classification"])
    if classification.shape[0] != points.shape[0]:
        raise ValueError(
            "length of points_attributes['classification'] does not match number of points"
        )


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    remain_pcd = pcd
    orig_indices = np.arange(len(points), dtype=np.int64)
    wall_planes_found = 0
    assigned_count = 0

    for _ in range(100):
        if len(remain_pcd.points) <= 3:
            log.debug("RANSAC wall detection: no more points to segment")
            break

        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=5000,
        )
        inliers_arr = np.asarray(inliers, dtype=np.int64)

        is_wall = is_plane_facing_side(plane_model, threshold=rotation_threshold)

        if is_wall:
            wall_planes_found += 1
            global_idx = orig_indices[inliers_arr]
            mask_init = classification[global_idx] == constants.PCB_CLASSIFICATION_INIT
            if np.any(mask_init):
                idx_to_set = global_idx[mask_init]
                classification[idx_to_set] = constants.PCB_CLASSIFICATION_WALL
                assigned_count += int(idx_to_set.size)

        keep_mask = np.ones(len(orig_indices), dtype=bool)
        keep_mask[inliers_arr] = False
        orig_indices = orig_indices[keep_mask]
        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)

    points_attributes["classification"] = classification

    log.info(
        "Wall RANSAC complete: %d wall plane(s) accepted, %d points labeled as WALL",
        wall_planes_found,
        assigned_count,
    )

    return points_attributes


def _apply_ransac_ceiling_detection(
    points: np.ndarray,
    points_attributes,
    facing_up_threshold: float = 0.95,
    min_inlier_density: float = 15,
):
    """
    Segment ceiling by repeated RANSAC planes (sandbox ``cieling_segmentation``).

    Outer loop runs up to 100 times; each step uses ``segment_plane`` with
    ``distance_threshold=0.1``, ``ransac_n=3``, and inner ``num_iterations=1000``.
    A plane is accepted as ceiling when inlier density is at least
    ``min_inlier_density`` (sandbox ``is_cieling_check`` used 15) and
    ``is_plane_facing_up`` passes with ``threshold=facing_up_threshold``.

    Args:
        points: (N, 3) point coordinates.
        points_attributes: Must include ``classification`` aligned with ``points``.
        facing_up_threshold: Passed to ``is_plane_facing_up`` (sandbox ``is_facing_up``).
        min_inlier_density: Minimum ``get_inlier_density`` for the plane inliers.

    Returns:
        Updated ``points_attributes``.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, 3)")
    if "classification" not in points_attributes:
        raise KeyError("points_attributes must contain 'classification'")

    classification = np.asarray(points_attributes["classification"])
    if classification.shape[0] != points.shape[0]:
        raise ValueError(
            "length of points_attributes['classification'] does not match number of points"
        )

    log.debug(
        "RANSAC ceiling detection (facing_up_threshold=%.4f, min_inlier_density=%s)...",
        facing_up_threshold,
        min_inlier_density,
    )

    def is_ceiling_check(plane_model, inlier_points_xyz, z_histogram):
        current_plane_z_intercept = -plane_model[3] / plane_model[2]   # -d1 / c1
        point_array = np.asarray(inlier_points_xyz)
        if point_array.size == 0:
            return False
        # check if the plane is within the higher half of the z_histogram
        is_within_higher_half = current_plane_z_intercept > z_histogram[0][len(z_histogram[0])//2]
        if not is_within_higher_half:
            return False

        inlier_indices = np.arange(len(point_array))
        density = get_inlier_density(
            point_array, inlier_indices, plane_model
        )
        if density < min_inlier_density:
            return False
        return is_plane_facing_up(plane_model, threshold=facing_up_threshold)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    ceiling_planes_found = 0
    assigned_count = 0
    remain_pcd = pcd
    orig_indices = np.arange(len(points), dtype=np.int64)
    n = len(points)
    z_values = np.asarray(pcd.points)[:, 2]
    z_histogram = create_histogram(z_values)

    log.info("Starting RANSAC ceiling detection on %d points...", n)

    for _ in range(100):
        if len(remain_pcd.points) <= 3:
            break
        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=5000,
        )
        inliers_arr = np.asarray(inliers, dtype=np.int64)
        if len(inliers_arr) == 0:
            break

        inlier_points = np.asarray(remain_pcd.points)[inliers_arr]
        is_ceiling = is_ceiling_check(plane_model, inlier_points, z_histogram)

        if is_ceiling:
            ceiling_planes_found += 1
            global_idx = orig_indices[inliers_arr]
            mask_init = classification[global_idx] == constants.PCB_CLASSIFICATION_INIT
            if np.any(mask_init):
                idx_to_set = global_idx[mask_init]
                classification[idx_to_set] = constants.PCB_CLASSIFICATION_CEILING
                assigned_count += int(idx_to_set.size)

        keep_mask = np.ones(len(orig_indices), dtype=bool)
        keep_mask[inliers_arr] = False
        orig_indices = orig_indices[keep_mask]
        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)

        if len(remain_pcd.points) <= 3:
            break

    points_attributes["classification"] = classification

    log.info(
        "Ceiling RANSAC complete: %d ceiling plane(s) accepted, %d points labeled as CEILING",
        ceiling_planes_found,
        assigned_count,
    )

    return points_attributes

def apply_wall_segmentation(
    pcd,
    points_attributes,
    rotation_threshold: float = 0.9,
):
    """
    Wrapper around ``apply_ransac_wall_detection`` for an Open3D PointCloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to segment.
    points_attributes : dict-like
        Per-point attributes dictionary.
    rotation_threshold : float
        Wall orientation gate; forwarded to ``apply_ransac_wall_detection``.

    Returns
    -------
    (pcd, points_attributes) : tuple
        The original point cloud and updated attributes.
    """
    points = np.asarray(pcd.points)
    points_attributes = _apply_ransac_wall_detection(
        points,
        points_attributes,
        rotation_threshold=rotation_threshold,
    )
    return pcd, points_attributes


def apply_ceiling_segmentation(
    pcd,
    points_attributes,
    facing_up_threshold: float = 0.95,
    min_inlier_density: float = 15,
):
    """
    Wrapper around ``apply_ransac_ceiling_detection`` for an Open3D PointCloud.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to segment.
    points_attributes : dict-like
        Per-point attributes dictionary.
    facing_up_threshold : float
        Ceiling horizontal normal gate; forwarded to ``apply_ransac_ceiling_detection``.
    min_inlier_density : float
        Minimum inlier density gate; forwarded to ``apply_ransac_ceiling_detection``.

    Returns
    -------
    (pcd, points_attributes) : tuple
        The original point cloud and updated attributes.
    """
    points = np.asarray(pcd.points)
    points_attributes = _apply_ransac_ceiling_detection(
        points,
        points_attributes,
        facing_up_threshold=facing_up_threshold,
        min_inlier_density=min_inlier_density,
    )
    return pcd, points_attributes

def _dbscan_cluster_mask(labels: np.ndarray) -> np.ndarray:
    """True where Open3D DBSCAN assigned a cluster id (>= 0); False for noise (-1)."""
    return np.asarray(labels, dtype=np.int64) >= 0

def segment_objects_dbscan(pcd, points_attributes, eps, min_points):
    """
    Run DBSCAN on points still classified as INIT only; assign PCB_CLASSIFICATION_OBJECTS
    to all points that belong to a cluster. Noise points (-1) stay INIT.

    Neighbors are computed only within the unclassified subset so clusters cannot
    bridge through ground, walls, or ceiling.

    Sets Clusters (1..K per cluster) when clusters exist; segment_objects may strip it
    for setup preview. Cleared at start if present from a previous run.
    """
    points_attributes.pop("Clusters", None)
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

    init_mask = classification == constants.PCB_CLASSIFICATION_INIT
    init_indices = np.where(init_mask)[0]
    n_init = int(init_indices.size)
    if n_init == 0:
        log.info("DBSCAN skipped: no INIT points available for object segmentation.")
        return pcd, points_attributes

    init_pcd = pcd.select_by_index(init_indices.tolist())
    labels = np.asarray(
        init_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    if labels.size == 0:
        log.info("DBSCAN produced no labels on INIT subset; skipping object segmentation.")
        return pcd, points_attributes

    clustered_mask = _dbscan_cluster_mask(labels)
    n_clustered = int(np.sum(clustered_mask))
    n_noise = int(np.sum(~clustered_mask))
    log.info(
        "DBSCAN on %d INIT points: %d clustered, %d noise (-1).",
        n_init,
        n_clustered,
        n_noise,
    )
    if n_clustered == 0:
        return pcd, points_attributes

    classification = classification.copy()
    global_object_indices = init_indices[clustered_mask]
    classification[global_object_indices] = constants.PCB_CLASSIFICATION_OBJECTS

    # Global cluster ids are reserved from a process-level singleton allocator so
    # cluster IDs remain unique across chunk-by-chunk DBSCAN calls.
    object_clusters = np.zeros(num_points, dtype=np.int32)
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    cluster_id_manager = ClustersManager.get_instance()
    reserved_ids = cluster_id_manager.reserve_next_ids(len(unique_labels))
    raw_to_global_id = {
        int(raw): int(global_id)
        for raw, global_id in zip(unique_labels, reserved_ids)
    }
    for local_idx, lab in enumerate(labels):
        lab = int(lab)
        if lab >= 0:
            global_idx = int(init_indices[local_idx])
            object_clusters[global_idx] = raw_to_global_id[lab]

    points_attributes["classification"] = classification
    points_attributes["Clusters"] = object_clusters
    return pcd, points_attributes

def apply_rgb_colors_to_pcd(pcd, points_attributes):
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
