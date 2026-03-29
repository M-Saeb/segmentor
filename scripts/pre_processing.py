import tempfile
import json
import os
import sys
from pathlib import Path

import laspy
from laspy.vlrs import VLR
import numpy as np
import open3d as o3d

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.models.las_wrapper import LasWrapper  # noqa: E402
from modules.models import pcd_tools  # noqa: E402
from modules.utils.config import SegmentorConfig  # noqa: E402
from modules.utils import log_utils  # noqa: E402

log = log_utils.get_dual_logger(__name__)

# Same skip list as LasWrapper.write_pcb_chunk_to_file for non-LAS dimensions.
_SKIP_DIMS = frozenset({"object_cluster_id"})


def _merge_chunks(wrap: LasWrapper) -> tuple[o3d.geometry.PointCloud, dict[str, np.ndarray]]:
    xyz_parts: list[np.ndarray] = []
    attr_parts: dict[str, list[np.ndarray]] = {}

    for _index, pcd, points_attributes in wrap.iter_input_chunks():
        log.info("Merging chunk %d", _index)
        xyz_parts.append(np.asarray(pcd.points, dtype=np.float64))
        for dim, arr in points_attributes.items():
            attr_parts.setdefault(dim, []).append(np.asarray(arr))

    if not xyz_parts:
        raise RuntimeError("No points were read from the input.")

    log.info("Merged %d chunks", len(xyz_parts))
    xyz = np.vstack(xyz_parts)
    merged_attrs: dict[str, np.ndarray] = {
        dim: np.concatenate(chunks, axis=0) for dim, chunks in attr_parts.items()
    }
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyz)
    return merged_pcd, merged_attrs

def _quadtree_leaves(
    xy: np.ndarray,
    indices: np.ndarray,
    target_points: int,
    max_depth: int,
    min_tile_meters: float,
    depth: int = 0,
) -> list[np.ndarray]:
    """
    Partition XY with a quadtree (four quadrants per split) until each leaf has at
    most ``target_points`` points (or limits apply). Returns a list of global index
    arrays, one per leaf, in DFS order (SW, SE, NW, NE).
    """
    n = len(indices)
    if n == 0:
        return []
    if n <= target_points or depth >= max_depth:
        return [indices]

    xs = xy[indices, 0]
    ys = xy[indices, 1]
    xmin = float(xs.min())
    xmax = float(xs.max())
    ymin = float(ys.min())
    ymax = float(ys.max())
    if (xmax - xmin) < 1e-12 and (ymax - ymin) < 1e-12:
        return [indices]
    if (
        min_tile_meters > 0
        and (xmax - xmin) <= min_tile_meters
        and (ymax - ymin) <= min_tile_meters
    ):
        return [indices]

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    x = xs
    y = ys
    sw = indices[(x <= cx) & (y <= cy)]
    se = indices[(x > cx) & (y <= cy)]
    nw = indices[(x <= cx) & (y > cy)]
    ne = indices[(x > cx) & (y > cy)]

    leaves: list[np.ndarray] = []
    for child in (sw, se, nw, ne):
        if len(child) == 0:
            continue
        leaves.extend(
            _quadtree_leaves(
                xy,
                child,
                target_points,
                max_depth,
                min_tile_meters,
                depth + 1,
            )
        )
    return leaves


def _voxel_downsample_preserve_attributes(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict[str, np.ndarray],
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, dict[str, np.ndarray]]:
    """Downsample `pcd` with Open3D voxel filter, preserving per-point attributes.

    Open3D's voxel downsampling produces new points (voxel centroids). We map each
    downsampled point to the nearest original point (KD-tree, k=1) and copy that
    point's LAS attributes.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    n_in = len(pcd.points)
    if n_in == 0:
        # Preserve attribute keys/empties consistently.
        empty_pcd = o3d.geometry.PointCloud()
        empty_pcd.points = o3d.utility.Vector3dVector(
            np.empty((0, 3), dtype=np.float64)
        )
        new_attrs = {
            dim: np.asarray(arr)[:0] for dim, arr in points_attributes.items()
        }
        return empty_pcd, new_attrs

    log.info("Voxel downsampling with voxel_size=%s", voxel_size)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    n_out = len(down_pcd.points)
    if n_out == 0:
        empty_pcd = o3d.geometry.PointCloud()
        empty_pcd.points = o3d.utility.Vector3dVector(
            np.empty((0, 3), dtype=np.float64)
        )
        new_attrs = {
            dim: np.asarray(arr)[:0] for dim, arr in points_attributes.items()
        }
        return empty_pcd, new_attrs

    # Map downsampled centroids to nearest original point indices.
    tree = o3d.geometry.KDTreeFlann(pcd)
    down_points = np.asarray(down_pcd.points)
    nn_indices = np.empty(n_out, dtype=np.int64)
    for i in range(n_out):
        pt = down_points[i]
        _, ind, _ = tree.search_knn_vector_3d(pt, 1)
        nn_indices[i] = int(ind[0])

    new_attrs: dict[str, np.ndarray] = {}
    for dim, arr in points_attributes.items():
        arr_np = np.asarray(arr)
        if arr_np.shape[0] == n_in:
            new_attrs[dim] = arr_np[nn_indices]
        else:
            # Attributes not per-point are kept as-is.
            new_attrs[dim] = arr

    return down_pcd, new_attrs


def _kdtree_seed_grid_3d(
    xmin: float,
    ymin: float,
    zmin: float,
    xmax: float,
    ymax: float,
    zmax: float,
    num_bins: int,
) -> tuple[np.ndarray, int, int, int]:
    """Build a 3D uniform grid of seed positions; return (seeds, nx, ny, nz)."""
    ex = max(float(xmax - xmin), 1e-12)
    ey = max(float(ymax - ymin), 1e-12)
    ez = max(float(zmax - zmin), 1e-12)

    num_bins = max(1, int(num_bins))
    s = (num_bins / (ex * ey * ez)) ** (1.0 / 3.0)
    nx = max(1, int(round(s * ex)))
    ny = max(1, int(round(s * ey)))
    nz = max(1, int(round(s * ez)))

    xs = np.linspace(xmin + ex / (2 * nx), xmax - ex / (2 * nx), nx)
    ys = np.linspace(ymin + ey / (2 * ny), ymax - ey / (2 * ny), ny)
    zs = np.linspace(zmin + ez / (2 * nz), zmax - ez / (2 * nz), nz)
    xi, yi, zi = np.meshgrid(xs, ys, zs, indexing="ij")
    seeds = np.column_stack((xi.ravel(), yi.ravel(), zi.ravel())).astype(
        np.float64, copy=False
    )
    return seeds, nx, ny, nz


def _order_kdtree_index(
    xyz: np.ndarray,
    target_points: int,
    max_depth: int,
) -> tuple[np.ndarray, list[dict[str, int]], int]:
    """Nearest-seed binning via Open3D ``KDTreeFlann``, then Z within each bin.

    Returns a permutation ``order`` such that ``xyz[order]`` groups points by
    nearest seed (bin), with Z sorted inside each bin. ``chunks`` lists non-empty
    bins in bin_id order with half-open ``[start, end)`` indices into that sorted
    array (``index.json`` uses the same ranges after reordering is applied).
    ``num_seeds`` is the number of seed points (spatial bins), including empty bins.
    """
    n = len(xyz)
    if n == 0:
        return np.arange(0, dtype=np.int64), [], 0

    xyz = np.asarray(xyz, dtype=np.float64)
    xmin, ymin, zmin = xyz.min(axis=0)
    xmax, ymax, zmax = xyz.max(axis=0)

    num_bins = max(1, int(np.ceil(n / float(target_points))))
    cap = 2 ** min(max(0, int(max_depth)), 24)
    num_bins = min(num_bins, cap)

    seeds, nx, ny, nz = _kdtree_seed_grid_3d(
        xmin, ymin, zmin, xmax, ymax, zmax, num_bins
    )
    num_seeds = int(len(seeds))
    if num_seeds == 0:
        seeds = np.array(
            [[0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]],
            dtype=np.float64,
        )
        num_seeds = 1

    log.info(
        "KD-tree seeds: %d points (%d x %d x %d grid), cap=%d",
        num_seeds,
        nx,
        ny,
        nz,
        cap,
    )

    seed_pcd = o3d.geometry.PointCloud()
    seed_pcd.points = o3d.utility.Vector3dVector(seeds)
    tree = o3d.geometry.KDTreeFlann(seed_pcd)

    bin_id = np.empty(n, dtype=np.int64)
    for i in range(n):
        pt = xyz[i]
        _, ind, _ = tree.search_knn_vector_3d(pt, 1)
        bin_id[i] = int(ind[0])

    z = xyz[:, 2]
    order = np.lexsort((z, bin_id))

    counts = np.bincount(bin_id, minlength=num_seeds)
    chunks: list[dict[str, int]] = []
    start = 0
    for b in range(num_seeds):
        c = int(counts[b])
        if c == 0:
            continue
        end = start + c
        chunks.append(
            {
                "start": start,
                "end": end,
                "bin_id": b,
                "point_count": c,
            }
        )
        start = end

    return order, chunks, num_seeds


def _write_las(
    wrap: LasWrapper,
    output_path: str,
    xyz: np.ndarray,
    merged_attrs: dict[str, np.ndarray],
) -> None:
    """
    Write LAS in slices of ``wrap.config.chunk_size``, matching
    ``LasWrapper.initialize_output_file`` + ``write_pcb_chunk_to_file`` (append).

    Optionally embeds a JSON VLR (user_id ``Segmentor``, record_id ``65001``) with
    ``segmentor_is_sorted`` and ``sort_mode`` to mark that this file was
    pre-processed by reordering/sorting.
    """
    n = len(xyz)
    chunk_size = int(wrap.config.chunk_size)
    if chunk_size <= 0:
        chunk_size = max(n, 1)

    base_header = wrap._create_new_las_filer_header()

    with laspy.open(output_path, mode="w", header=base_header) as _writer:
        pass

    log.info("Writing %d points to %s in chunks of %d", n, output_path, chunk_size)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        new_las_header = wrap._create_new_las_filer_header()
        with laspy.open(output_path, mode="a", header=new_las_header) as writer:
            las_chunk = laspy.LasData(new_las_header)
            las_chunk.x = xyz[start:end, 0]
            las_chunk.y = xyz[start:end, 1]
            las_chunk.z = xyz[start:end, 2]
            for dim, arr in merged_attrs.items():
                if dim in _SKIP_DIMS:
                    continue
                setattr(las_chunk, dim, arr[start:end])
            writer.append_points(las_chunk.points)
            log.debug("Appended points [%d, %d)", start, end)


def run_script(
    config: SegmentorConfig,
) -> int:

    wrap = LasWrapper(config, auto_read_input_file=False)

    pcd, merged_attrs = _merge_chunks(wrap)
    log.info("Initial point count: %d", len(pcd.points))

    # 2) Remove Statistical outliers (Open3D) while keeping attributes aligned.
    pcd, merged_attrs = pcd_tools.filter_noise(
        pcd,
        merged_attrs,
        nb_neighbors=config.filter_noise["nb_neighbors"],
        std_ratio=config.filter_noise["std_ratio"],
    )
    log.info("After statistical outlier removal: %d", len(pcd.points))

    # 3) Voxel downsampling (Open3D), preserving LAS attributes by nearest neighbor.
    pcd, merged_attrs = _voxel_downsample_preserve_attributes(
        pcd,
        merged_attrs,
        voxel_size=config.downsample["voxel_size"],
    )
    log.info("After voxel downsampling: %d", len(pcd.points))

    xyz = np.asarray(pcd.points)
    kd_num_seeds = 0
    kd_chunks: list[dict[str, int]] = []
    order, kd_chunks, kd_num_seeds = _order_kdtree_index(
        xyz,
        target_points=config.chunk_size,
        max_depth=64,
    )

    tmp_folder_path = tempfile.mkdtemp(prefix=f"segmentor_{wrap.genrate_unique_id()}")
    log.info(f"Storing pre-processed data in: {tmp_folder_path}")
    output_path = os.path.join(tmp_folder_path, "output.las")
    index_path = os.path.join(tmp_folder_path, "index.json")

    _write_las(
        wrap,
        output_path,
        xyz,
        merged_attrs,
    )
    index_payload = {
        "target_points_per_tile": config.chunk_size,
        "num_seeds": kd_num_seeds,
        "chunks": kd_chunks,
    }
    with open(index_path, "w") as f:
        json.dump(index_payload, f, indent=2)
    log.info("Wrote spatial bin index to %s", index_path)

    return tmp_folder_path

def main() -> int:
    return run_script(
        config=SegmentorConfig("config.yml"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
