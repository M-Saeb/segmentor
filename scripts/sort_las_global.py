#!/usr/bin/env python3
"""
Reorder LAS/LAZ points for better spatial locality when read in fixed-size chunks.

Modes:
  - lex: lexicographic (X, then Y, then Z) — can look like long thin strips per chunk.
  - spatial-xy: adaptive XY quadtree leaves, then Z within each leaf (default).
  - uniform-xy: uniform XY grid sized from bbox and target points per cell.

Reads via LasWrapper chunk iteration, merges in memory, sorts once, writes using the
wrapper's header helper.

Outputs a Segmentor VLR (JSON: ``segmentor_is_sorted``, ``sort_mode``) unless
``--no-segmentor-vlr`` is set.

Run from the project root so `modules` imports resolve, e.g.:

    python scripts/sort_las_global.py -i input.las -o sorted.las
    python scripts/sort_las_global.py -i in.las -o out.las --mode lex
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.models.las_wrapper import LasWrapper  # noqa: E402
from modules.utils import config as config_module  # noqa: E402
from modules.utils import log_utils  # noqa: E402

log = log_utils.get_dual_logger(__name__)

# Same skip list as LasWrapper.write_pcb_chunk_to_file for non-LAS dimensions.
_SKIP_DIMS = frozenset({"object_cluster_id"})


def _merge_chunks(wrap: LasWrapper) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    return xyz, merged_attrs


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
    ``segmentor_is_sorted`` and ``sort_mode`` for downstream detection.
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
                description="segmentor_sort_metadata",
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
        description="Reorder LAS points for chunk locality (spatial XY) or lexicographic XYZ."
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
        "--mode",
        choices=("lex", "spatial-xy", "uniform-xy"),
        default="spatial-xy",
        help="lex=XYZ sweep; spatial-xy=adaptive quadtree on XY; uniform-xy=fixed grid (default: spatial-xy)",
    )
    parser.add_argument(
        "--target-points-per-tile",
        type=int,
        default=None,
        metavar="N",
        help="Target max points per XY tile (quadtree leaf or uniform cell). "
        "Defaults to chunk_size from config.",
    )
    parser.add_argument(
        "--max-quadtree-depth",
        type=int,
        default=64,
        metavar="D",
        help="Safety cap on quadtree recursion depth (spatial-xy only).",
    )
    parser.add_argument(
        "--min-tile-meters",
        type=float,
        default=0.0,
        metavar="M",
        help="Stop subdividing when both XY edge lengths are at or below this (meters; spatial-xy only; 0=disabled).",
    )
    parser.add_argument(
        "--no-segmentor-vlr",
        action="store_true",
        help="Do not write the Segmentor JSON metadata VLR (segmentor_is_sorted).",
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

    target = (
        args.target_points_per_tile
        if args.target_points_per_tile is not None
        else int(cfg.chunk_size)
    )
    if target <= 0:
        print("--target-points-per-tile / chunk_size must be positive.", file=sys.stderr)
        return 1

    wrap = LasWrapper(cfg, auto_read_input_file=False)

    xyz, merged_attrs = _merge_chunks(wrap)

    if args.mode == "lex":
        order = _lexsort_xyz(xyz)
        label = "XYZ lexicographic"
    elif args.mode == "uniform-xy":
        order = _order_uniform_xy(xyz, target)
        label = f"uniform XY (target ~{target} pts/tile)"
    else:
        order = _order_spatial_xy_quadtree(
            xyz,
            target_points=target,
            max_depth=args.max_quadtree_depth,
            min_tile_meters=args.min_tile_meters,
        )
        label = f"spatial XY quadtree (target ~{target} pts/leaf)"

    xyz = xyz[order]
    for dim in list(merged_attrs.keys()):
        merged_attrs[dim] = merged_attrs[dim][order]

    _write_las(
        wrap,
        output_path,
        xyz,
        merged_attrs,
        sort_mode=args.mode,
        write_segmentor_vlr=not args.no_segmentor_vlr,
    )

    print(f"Wrote {len(xyz):,} points ({label}) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
