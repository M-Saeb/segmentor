#!/usr/bin/env python3
"""
Pre-process LAS/LAZ data by reordering (sorting) points for better spatial locality
when read in fixed-size chunks.

Modes:
  - lex: lexicographic (X, then Y, then Z) — can look like long thin strips per chunk.
  - spatial-xy: adaptive XY quadtree leaves, then Z within each leaf.
  - uniform-xy: uniform XY grid sized from bbox and target points per cell.
  - kd-tree: Open3D `KDTreeFlann` nearest-seed binning, then Z within each bin (default).

Reads via LasWrapper chunk iteration, merges in memory, sorts once, writes using the
wrapper's header helper.

Outputs a Segmentor VLR (JSON: ``segmentor_is_sorted``, ``sort_mode``) unless
``--no-segmentor-vlr`` is set.

With ``--sort-mode kd-tree``, also writes ``index.json`` next to the output LAS:
half-open ``[start, end)`` ranges per spatial bin in sorted file order (for chunk
processing).

Run from the project root so `modules` imports resolve, e.g.:

    python scripts/pre_processing.py -i input.las -o preprocessed.las
    python scripts/pre_processing.py -i in.las -o out.las --sort-mode lex
"""

from __future__ import annotations

import argparse
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
from modules.utils import config as config_module  # noqa: E402
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


def _lexsort_xyz(xyz: np.ndarray) -> np.ndarray:
    """Return indices that sort points by X, then Y, then Z (np.lexsort key order)."""
    return np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))


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


def _order_spatial_xy_quadtree(
    xyz: np.ndarray,
    target_points: int,
    max_depth: int,
    min_tile_meters: float,
) -> np.ndarray:
    """Sort key: quadtree leaf id (DFS), then Z."""
    n = len(xyz)
    xy = xyz[:, :2]
    root_idx = np.arange(n, dtype=np.int64)
    leaves = _quadtree_leaves(
        xy, root_idx, target_points, max_depth, min_tile_meters, 0
    )
    leaf_id = np.empty(n, dtype=np.int64)
    for lid, leaf in enumerate(leaves):
        leaf_id[leaf] = lid
    z = xyz[:, 2]
    return np.lexsort((z, leaf_id))


def _order_uniform_xy(xyz: np.ndarray, target_points: int) -> np.ndarray:
    """
    Uniform XY grid: choose nx, ny so approximate cells ≈ N / target_points, then
    sort by cell_id, then Z.
    """
    log.info("Ordering uniform XY grid")
    n = len(xyz)
    if n == 0:
        return np.array([], dtype=np.int64)

    xy = xyz[:, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    ex = max(float(xmax - xmin), 1e-12)
    ey = max(float(ymax - ymin), 1e-12)

    num_tiles = max(1, int(np.ceil(n / float(target_points))))
    nx = max(1, int(round(np.sqrt(num_tiles * (ex / ey)))))
    ny = max(1, int(np.ceil(num_tiles / nx)))

    ix = np.floor((xy[:, 0] - xmin) / (ex / nx)).astype(np.int64)
    iy = np.floor((xy[:, 1] - ymin) / (ey / ny)).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    cell_id = ix + nx * iy
    z = xyz[:, 2]
    return np.lexsort((z, cell_id))


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
    min_tile_meters: float,
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

    if min_tile_meters > 0:
        while nx * ny * nz > 1:
            dx = ex / nx
            dy = ey / ny
            dz = ez / nz
            if min(dx, dy, dz) >= min_tile_meters - 1e-12:
                break
            # Coarsen the axis with the smallest cell edge (finest subdivision).
            edges = [(dx, "x"), (dy, "y"), (dz, "z")]
            edges.sort(key=lambda t: t[0])
            _, which = edges[0]
            if which == "x" and nx > 1:
                nx = max(1, nx // 2)
            elif which == "y" and ny > 1:
                ny = max(1, ny // 2)
            elif nz > 1:
                nz = max(1, nz // 2)
            else:
                break

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
    min_tile_meters: float,
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
        xmin, ymin, zmin, xmax, ymax, zmax, num_bins, min_tile_meters
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
    *,
    sort_mode: str,
    write_segmentor_vlr: bool = True,
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
    if write_segmentor_vlr:
        payload = json.dumps(
            {"segmentor_is_sorted": True, "sort_mode": sort_mode},
            separators=(",", ":"),
        ).encode("utf-8")
        base_header.vlrs.append(
            VLR(
                user_id="Segmentor",
                record_id=65001,
                description="segmentor_preprocess_metadata",
                record_data=payload,
            )
        )

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-processing of LAS file for better spatial locality."
    )
    parser.add_argument("--input", "-i", required=True, help="Input .las or .laz path")
    parser.add_argument("--output", "-o", required=True, help="Output .las path")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yml",
        help="YAML config path (chunk_size and LasWrapper settings come from here)",
    )
    parser.add_argument(
        "--sort-mode",
        choices=("lex", "spatial-xy", "uniform-xy", "kd-tree", "none"),
        default="none",
        help=(
            "lex=XYZ sweep; spatial-xy=adaptive quadtree on XY; "
            "uniform-xy=fixed grid; kd-tree=Open3D KDTreeFlann nearest-seed binning "
            "(writes index.json beside -o); none=no sorting (default: kd-tree)"
        ),
    )
    parser.add_argument(
        "--target-points-per-tile",
        type=int,
        default=25000,
        metavar="N",
        help="Target max points per XY tile (quadtree leaf or uniform cell). "
        "Defaults to chunk_size from config.",
    )
    parser.add_argument(
        "--min-tile-meters",
        type=float,
        default=0.0,
        metavar="M",
        help="Stop subdividing when tile edge lengths are at or below this (meters; spatial-xy and kd-tree; 0=disabled).",
    )
    parser.add_argument(
        "--filter-noise",
        action="store_true",
        default=False,
        help="Remove statistical outliers (Open3D) (default: True).",
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        metavar="K",
        help="Statistical outlier removal: nb_neighbors (default: 20).",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        metavar="R",
        help="Statistical outlier removal: std_ratio (default: 2.0).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.15,
        metavar="S",
        help="Voxel downsampling size passed to Open3D (default: 0.1).",
    )
    parser.add_argument(
        "--no-segmentor-vlr",
        action="store_true",
        help="Do not write the Segmentor preprocessing JSON metadata VLR (segmentor_is_sorted).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if not os.path.isfile(input_path):
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    cfg = config_module.SegmentorConfig(args.config)
    cfg.input_file = input_path
    cfg.reset_classification = False

    target = args.target_points_per_tile
    if target <= 0:
        log.error("--target-points-per-tile / chunk_size must be positive.")
        return 1

    wrap = LasWrapper(cfg, auto_read_input_file=False)

    pcd, merged_attrs = _merge_chunks(wrap)
    log.info("Initial point count: %d", len(pcd.points))

    # 2) Remove Statistical outliers (Open3D) while keeping attributes aligned.
    if args.filter_noise:
        pcd, merged_attrs = pcd_tools.filter_noise(
            pcd,
            merged_attrs,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )
        log.info("After statistical outlier removal: %d", len(pcd.points))
    else:
        log.info("Skipping statistical outlier removal")


    # 3) Voxel downsampling (Open3D), preserving LAS attributes by nearest neighbor.
    pcd, merged_attrs = _voxel_downsample_preserve_attributes(
        pcd,
        merged_attrs,
        voxel_size=args.voxel_size,
    )
    log.info("After voxel downsampling: %d", len(pcd.points))

    xyz = np.asarray(pcd.points)

    sort_mode = args.sort_mode
    kd_num_seeds = 0
    kd_chunks: list[dict[str, int]] = []
    if sort_mode == "lex":
        order = _lexsort_xyz(xyz)
        label = "XYZ lexicographic"
    elif sort_mode == "uniform-xy":
        order = _order_uniform_xy(xyz, target)
        label = f"uniform XY (target ~{target} pts/tile)"
    elif sort_mode == "spatial-xy":
        order = _order_spatial_xy_quadtree(
            xyz,
            target_points=target,
            max_depth=64,
            min_tile_meters=args.min_tile_meters,
        )
        label = f"spatial XY quadtree (target ~{target} pts/leaf)"
    elif sort_mode == "kd-tree":
        order, kd_chunks, kd_num_seeds = _order_kdtree_index(
            xyz,
            target_points=target,
            max_depth=64,
            min_tile_meters=args.min_tile_meters,
        )
        label = f"kd-tree indexing (target ~{target} pts/leaf)"
    else:
        order = np.arange(len(xyz), dtype=np.int64)
        label = "no sorting"

    if sort_mode != "none":
        log.debug("Sorting with %s", sort_mode)
        xyz = xyz[order]
        for dim in list(merged_attrs.keys()):
            merged_attrs[dim] = merged_attrs[dim][order]
    else:
        log.info("Skipping sorting")

    _write_las(
        wrap,
        output_path,
        xyz,
        merged_attrs,
        sort_mode=sort_mode,
        write_segmentor_vlr=not args.no_segmentor_vlr,
    )

    if sort_mode == "kd-tree":
        index_path = Path(output_path).parent / "index.json"
        index_payload = {
            "sort_mode": sort_mode,
            "target_points_per_tile": target,
            "num_seeds": kd_num_seeds,
            "chunks": kd_chunks,
        }
        index_path.write_text(
            json.dumps(index_payload, indent=2),
            encoding="utf-8",
        )
        log.info("Wrote spatial bin index to %s", index_path)

    print(f"Wrote {len(xyz):,} points ({label}) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
