import numpy as np
from  utils import log_utils
from .classses import GROUND_CLASSIFICATION

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

def cloth_simulation_filter(pcd, resolution=1.0, rigidness=3, max_iter=500):
    """Segment ground points from a point cloud using a simplified Cloth Simulation Filter.

    Args:
        pcd (np.ndarray): Nx3 array of points (x, y, z).
        resolution (float): grid cell size in same units as `pcd`.
        rigidness (int): controls cloth stiffness (higher -> stiffer -> less smoothing).
        max_iter (int): maximum number of simulation iterations.

    Returns:
        np.ndarray: boolean mask of length N; True for ground points.
    """
    if not isinstance(pcd, np.ndarray):
        raise TypeError("pcd must be a numpy.ndarray of shape (N,3)")
    if pcd.ndim != 2 or pcd.shape[1] < 3:
        raise ValueError("pcd must have shape (N,3)")

    xs = pcd[:, 0]
    ys = pcd[:, 1]
    zs = pcd[:, 2]

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
    threshold = max(0.3, resolution * 0.5)
    is_ground = (zs - point_cloth_z) <= threshold

    return is_ground