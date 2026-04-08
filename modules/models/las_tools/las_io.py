from typing import Iterator, Union
import hashlib
import uuid
import laspy
import numpy as np
import open3d as o3d

from modules.utils import log_utils

log = log_utils.get_dual_logger(__name__)
EXCLUDED_DIMENSIONS = ('x', 'y', 'z')

def get_original_las_header(file_path: str) -> laspy.LasHeader:
    """Get the original LAS header from the file."""
    header = None
    with laspy.open(file_path) as las:
        header = las.header
    return header

def create_new_las_header(original_header: laspy.LasHeader) -> laspy.LasHeader:
    result = laspy.LasHeader(point_format=original_header.point_format, version=original_header.version)
    result.offsets = original_header.offsets
    result.scales = original_header.scales
    return result

def get_las_file_uuid(file_path: str) -> Union[uuid.UUID, None]:    
    to_include_headers = ["_version", "creation_date", "point_count", "scales", "offsets", "maxs", "mins", "offset_to_point_data"]
    header = get_original_las_header(file_path)
    items = []
    for key in header.__dict__.keys():
        if key not in to_include_headers:
            continue
        items.append(f"{key}: {getattr(header, key)}")
    raw_string = "\n".join(items)
    return uuid.uuid5(uuid.NAMESPACE_URL, raw_string)

def read_full_las_file(file_path: str) -> tuple[o3d.geometry.PointCloud, dict]:
    with laspy.open(file_path) as reader:
        las_data = reader.read()
        pcd = o3d.geometry.PointCloud()
        xyz = np.vstack((las_data.x, las_data.y, las_data.z)).T
        pcd.points = o3d.utility.Vector3dVector(xyz)
        points_attributes = {}
        for dim in las_data.point_format.dimension_names:
            if dim not in EXCLUDED_DIMENSIONS:
                points_attributes[dim] = getattr(las_data, dim)
        return pcd, points_attributes

def read_chunk_las_file(
    file_path: str,
    chunk_size: int,
) -> Iterator[tuple[o3d.geometry.PointCloud, dict]]:
    with laspy.open(file_path) as f:
        for chunk in f.chunk_iterator(chunk_size):
            pcd = o3d.geometry.PointCloud()
            points_attributes = {}
            xyz = np.vstack((chunk.x, chunk.y, chunk.z)).T
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # Store all attributes except x, y, z
            for dim in chunk.point_format.dimension_names:
                if dim not in EXCLUDED_DIMENSIONS:
                    if dim not in points_attributes:
                        points_attributes[dim] = []
                    points_attributes[dim].append(getattr(chunk, dim))

            for dim, arrays in points_attributes.items():
                points_attributes[dim] = np.concatenate(arrays)
    
            yield pcd, points_attributes

def init_las_file(file_path: str, header: laspy.LasHeader) -> None:
    with laspy.open(
        file_path,
        mode="w",
        header=header,
    ) as writer:
        # Header is written on open; no points yet.
        pass

def write_chunk_las_file(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    full_path: str,
    header: laspy.LasHeader,
):
    """
    Write the points and all original attributes to a new LAS file.
    Args:
        points: Points to write
        points_attributes: Points attributes to write
        full_path: Path to save the new .las file
        header: Header to write
    """

    points = np.asarray(pcd.points)
    with laspy.open(full_path, mode="a", header=header) as writer:
        las_chunk = laspy.LasData(header)
        las_chunk.x = points[:, 0]
        las_chunk.y = points[:, 1]
        las_chunk.z = points[:, 2]
        for dim, arr in points_attributes.items():
            if dim in EXCLUDED_DIMENSIONS:
                continue
            setattr(las_chunk, dim, arr)
        writer.append_points(las_chunk.points)

def write_full_las_file(
    pcd: o3d.geometry.PointCloud,
    points_attributes: dict,
    full_path: str,
    header: laspy.LasHeader,
):
    points = np.asarray(pcd.points)
    with laspy.open(full_path, mode="w", header=header) as writer:
        las_data = laspy.LasData(header)
        las_data.x = points[:, 0]
        las_data.y = points[:, 1]
        las_data.z = points[:, 2]

        for dim, arr in points_attributes.items():
            setattr(las_data, dim, arr)

        writer.write_points(las_data.points)
