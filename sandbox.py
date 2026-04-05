import open3d as o3d
import json
import laspy
import numpy as np
from modules.utils import config
from toolbox.density import get_inlier_density
from toolbox.ai_tools import get_plane_id_from_ai
from toolbox.video_render import render_rotating_isometric_video
from toolbox.restore_origina_pcd import restore_original_points_in_inliers

LAS_FILE_PATH = "/tmp/segmentor_4b1faa02-e417-5d6a-8737-011b8966905e/output.las"
INDEXING_FILE_PATH = "/tmp/segmentor_4b1faa02-e417-5d6a-8737-011b8966905e/index.json"
# Points per read when streaming LAS for kd_tree (logical index.json ranges use global point indices).
_KD_TREE_LAS_READ_CHUNK_SIZE = 1_000_000


def display_pcd_in_rgb():
    las = laspy.read(LAS_FILE_PATH)
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(float)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    red = np.asarray(las.red, dtype=float)
    green = np.asarray(las.green, dtype=float)
    blue = np.asarray(las.blue, dtype=float)
    rgb = np.vstack((red, green, blue)).T
    max_val = np.max(rgb)
    if max_val > 0:
        rgb /= max_val  # normalize to [0, 1] like PCDPreviewer.apply_rgb_colors()
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    draw_geometries_with_light_blue_background([pcd])


def draw_geometries_with_light_blue_background(
    geometries,
    window_name: str = "Open3D",
):
    """
    Wrapper around Open3D's Visualizer that shows geometries on a light blue background.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.8, 0.9, 1.0])
    vis.run()
    vis.destroy_window()


def restore_inlier_rgb_from_las(pcd_inliers: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Given a point-cloud of inliers and the original LAS/LAZ file, return a new
    point-cloud with RGB colors restored from the source LAS.

    The mapping is done via nearest-neighbour in XYZ space: for each inlier
    point, we find the closest point in the original LAS and copy its RGB.
    """
    las = laspy.read(LAS_FILE_PATH)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    red = np.asarray(las.red, dtype=float)
    green = np.asarray(las.green, dtype=float)
    blue = np.asarray(las.blue, dtype=float)
    rgb = np.vstack((red, green, blue)).T.astype(np.float64)
    max_val = np.max(rgb)
    if max_val > 0:
        rgb /= max_val  # normalize to [0, 1] like display_pcd_in_rgb / PCDPreviewer

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(xyz)
    source_pcd.colors = o3d.utility.Vector3dVector(rgb)

    tree = o3d.geometry.KDTreeFlann(source_pcd)
    inlier_points = np.asarray(pcd_inliers.points, dtype=np.float64)
    if inlier_points.size == 0:
        return pcd_inliers

    source_colors = np.asarray(source_pcd.colors)
    restored_colors = np.empty((len(inlier_points), 3), dtype=np.float64)
    for i, pt in enumerate(inlier_points):
        _, idx, _ = tree.search_knn_vector_3d(pt, 1)
        restored_colors[i] = source_colors[idx[0]]

    out = o3d.geometry.PointCloud(pcd_inliers)
    out.colors = o3d.utility.Vector3dVector(restored_colors)
    return out


def write_las_file(file_path, points):
    with laspy.open(file_path, 'w') as las:
        las.points = points

def is_facing_up(plane, threshold=0.95, debug=False):
    a, b, c, d = plane
    normal = np.array([a, b, c])
    
    # Normalize the normal vector
    normal_unit = normal / np.linalg.norm(normal)
    
    # Check if Z component is close to 1 (facing up) or -1 (facing down)
    if debug:
        print(f"threshold: {threshold}")
        print(f"abs(normal_unit[2]): {abs(normal_unit[2])}")
    return abs(normal_unit[2]) > threshold

def is_facing_side(plane, threshold=0.9):
    a, b, c, d = plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    result = abs(normal[2]) < (1 - threshold )
    return result

def planes_footprint_overlap_xy(pcd_a, pcd_b, padding=0.0, min_overlap_area_m2=0.0, debug=False):
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
    if debug:
        print(f"Checking if planes footprint overlap xy")
        print(f"area: {area}")
        print(f"min_overlap_area_m2: {min_overlap_area_m2}")
        print(f"result: {result}")
    return result

def is_floor_check(plane_model, pcd, all_floor_plane_data, debug=False):
    if debug:
        print(f"=== starting floor check ===")
    current_plane_z_intercept = -plane_model[3] / plane_model[2]   # -d1 / c1
    if not is_facing_up(plane_model, debug=debug):
        return False
    for item in all_floor_plane_data:
        floor_plane_normal = item["plane_model"]
        floor_plane_z_intercept = -floor_plane_normal[3] / floor_plane_normal[2]   # -d1 / c1
        z_difference = abs(current_plane_z_intercept - floor_plane_z_intercept)
        is_above = z_difference > 0.2
        if debug:
            print(f"z_difference: {z_difference}")
            print(f"is_above: {is_above}")
            print(f"floor plane z_intercept: {floor_plane_z_intercept}")
            print(f"current plane z_intercept: {current_plane_z_intercept}")
        if not is_above:
            continue
        is_colliding = planes_footprint_overlap_xy(
            pcd, item["pcd"],
            min_overlap_area_m2=1,
            debug=debug
        )
        if is_colliding:
            return False
    return True

def is_wall_check(plane_model, pcd_inlier, all_wall_plane_data):
    point_array = np.asarray(pcd_inlier.points)
    inlier_indices = np.arange(len(point_array))
    density = get_inlier_density(point_array, inlier_indices, plane_model)
    # if density < 30:
    #     return False
    is_facing_correct = is_facing_side(plane_model)
    return is_facing_correct

def is_cieling_check(plane_model, pcd_inlier, all_cieling_plane_data, debug=False):
    point_array = np.asarray(pcd_inlier.points)
    inlier_indices = np.arange(len(point_array))
    density = get_inlier_density(point_array, inlier_indices, plane_model, debug=debug)
    if density < 15:
        return False
    is_facing_correct = is_facing_up(plane_model)
    return is_facing_correct

def is_other_structure_check(plane_model, pcd_inlier, debug=False):
    point_array = np.asarray(pcd_inlier.points)
    inlier_indices = np.arange(len(point_array))
    density = get_inlier_density(point_array, inlier_indices, plane_model, debug=debug)
    return density > 15

def get_red_colors(i):
    red_grade = i * 0.01
    return [ 1- red_grade, 0.0, 0.0]

def get_blue_colors(i):
    blue_grade = i * 0.01
    return [ 0.0, 0.0, 1 - blue_grade]

def get_green_colors(i):
    green_grade = i * 0.01
    return [ 0.0, 1 - green_grade, 0.0]

def get_yellow_colors(i):
    yellow_grade = i * 0.01
    return [ 1 - yellow_grade, 1 - yellow_grade, 0.0]

def get_orange_colors(i):
    orange_grade = i * 0.01
    return [ 1 - orange_grade, 0.5 - orange_grade, 0.0]

def get_purple_colors(i):
    purple_grade = i * 0.01
    return [ 1 - purple_grade, 0.0, 1 - purple_grade]

def get_cyan_colors(i):
    cyan_grade = i * 0.01
    return [ 0.0, 1 - cyan_grade, 1 - cyan_grade]

def get_pink_colors(i):
    pink_grade = i * 0.01
    return [ 1 - pink_grade, 0.0, 1 - pink_grade]

def get_brown_colors(i):
    brown_grade = i * 0.01
    return [ 0.5 - brown_grade, 0.25 - brown_grade, 0.0]


def segment_ground(pcd, debug=False):
    floor_plane_data = []
    other_planes_data = []
    remain_pcd = pcd
    for i in range(10):        
        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000,
        )

        pcd_inliers = remain_pcd.select_by_index(inliers)

        is_floor = is_floor_check(plane_model, pcd_inliers, floor_plane_data, debug=debug)

        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)
        if len(remain_pcd.points) <= 3:
            break
        
        if is_floor:
            pcd_color = get_red_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            floor_plane_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })
        else:
            pcd_color = get_blue_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            other_planes_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })

        if debug:
            color = get_green_colors(i) if is_floor else get_orange_colors(i)
            display_pcd = o3d.geometry.PointCloud(pcd_inliers.points)
            display_pcd.paint_uniform_color(color)
            o3d.visualization.draw_geometries(
                [p["pcd"] for p in floor_plane_data] + \
                [p["pcd"] for p in other_planes_data] + \
                [remain_pcd] + \
                [display_pcd]
            )

    for p in other_planes_data:
        remain_pcd += p["pcd"]	

    return floor_plane_data, remain_pcd

def segment_walls(pcd, debug=False):
    wall_plane_data = []
    other_planes_data = []
    remain_pcd = pcd
    for i in range(100):
        if len(remain_pcd.points) <= 3:
            print("no more points to segment walls")
            break

        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=5000,
        )
        pcd_inliers = remain_pcd.select_by_index(inliers)
        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)
        pcd_color = get_yellow_colors(i)
        pcd_inliers.paint_uniform_color(pcd_color)
        is_wall = is_wall_check(plane_model, pcd_inliers, wall_plane_data)

        if is_wall:
            pcd_color = get_green_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            wall_plane_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })
        else:
            pcd_color = get_blue_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            other_planes_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })

        if debug:
            display_pcd = o3d.geometry.PointCloud(pcd_inliers.points)
            color = get_orange_colors(i) if is_wall else get_purple_colors(i)
            display_pcd.paint_uniform_color(color)
            o3d.visualization.draw_geometries(
                [p["pcd"] for p in wall_plane_data] + \
                [p["pcd"] for p in other_planes_data] + \
                [remain_pcd] + \
                [display_pcd]
            )

    for p in other_planes_data:
        remain_pcd += p["pcd"]	

    return wall_plane_data, remain_pcd

def cieling_segmentation(pcd, debug=False):
    cieling_plane_data = []
    other_planes_data = []
    remain_pcd = pcd
    for i in range(100):        
        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000,
        )

        pcd_inliers = remain_pcd.select_by_index(inliers)
        is_cieling = is_cieling_check(plane_model, pcd_inliers, cieling_plane_data, debug=debug)
        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)
        if len(remain_pcd.points) <= 3:
            break
        
        if is_cieling:
            pcd_color = get_yellow_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            cieling_plane_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })
        else:
            pcd_color = get_blue_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            other_planes_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })

        if debug:
            display_pcd = o3d.geometry.PointCloud(pcd_inliers.points)
            color = get_orange_colors(i) if is_cieling else get_purple_colors(i)
            display_pcd.paint_uniform_color(color)
            o3d.visualization.draw_geometries(
                [p["pcd"] for p in cieling_plane_data] + \
                [p["pcd"] for p in other_planes_data] + \
                [remain_pcd] + \
                [display_pcd]
            )

    for p in other_planes_data:
        remain_pcd += p["pcd"]	

    return cieling_plane_data, remain_pcd

def other_structure_segmentation(pcd, debug=False):
    other_structure_plane_data = []
    other_planes_data = []
    remain_pcd = pcd
    for i in range(100):        
        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000,
        )

        pcd_inliers = remain_pcd.select_by_index(inliers)
        is_other_structure = is_other_structure_check(plane_model, pcd_inliers, debug=debug)
        remain_pcd = remain_pcd.select_by_index(inliers, invert=True)
        if len(remain_pcd.points) <= 3:
            break
        
        if is_other_structure:
            pcd_color = get_orange_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            other_structure_plane_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })
        else:
            pcd_color = get_blue_colors(i)
            pcd_inliers.paint_uniform_color(pcd_color)
            other_planes_data.append({
                "plane_model": plane_model,
                "pcd": pcd_inliers
            })

    for p in other_planes_data:
        remain_pcd += p["pcd"]	

    return other_structure_plane_data, remain_pcd

def segment_objects(pcd, debug=False):
    labels = np.asarray(pcd.cluster_dbscan(eps=0.35, min_points=15, print_progress=False))
    if labels.size == 0:
        return [], pcd

    labels_count = int(labels.max())
    objects_pcds = []
    all_cluster_indices = []
    for i in range(labels_count + 1):
        indices = np.where(labels == i)[0]
        if indices.size == 0:
            continue
        cluster_pcd = pcd.select_by_index(indices)
        cluster_pcd = restore_inlier_rgb_from_las(cluster_pcd)
        og_cluster_pcd = restore_original_points_in_inliers(
            cluster_pcd,
            "/home/mohammedalsalihi/LidarData/for_mohammed/fh_underground.las",
            voxel_size=0.15,
        )
        # draw_geometries_with_light_blue_background([cluster_pcd])
        to_render = "y" # input("to render? (y/n)")
        if to_render == "y":
            # render_rotating_isometric_video(cluster_pcd, "temp/video.mp4", fps=10)
            response = get_plane_id_from_ai(
                og_cluster_pcd,
                debug=debug
            )
            if response == "CAR":
                cluster_pcd.paint_uniform_color(get_cyan_colors(i % 100))
            elif response == "OTHER":
                cluster_pcd.paint_uniform_color(get_pink_colors(i % 100))
            else:
                cluster_pcd.paint_uniform_color(get_orange_colors(i % 100))


        if debug:
            display_pcd = o3d.geometry.PointCloud(pcd.points)
            display_pcd.paint_uniform_color([0.5, 0.5, 0.5])# grey
            o3d.visualization.draw_geometries(
                [display_pcd] + [cluster_pcd]
            )

        objects_pcds.append(cluster_pcd)
        all_cluster_indices.extend(indices.tolist())

    remain_pcd = pcd.select_by_index(all_cluster_indices, invert=True)
    return objects_pcds, remain_pcd

def get_chunk_sizes(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["chunks"]

def kd_tree():
    """
    Yield one Open3D point cloud per logical chunk from index.json.

    For each logical range [start, end) (global point indices), stream LAS_FILE_PATH
    in fixed-size file chunks and merge the overlapping slices so a logical chunk can
    span multiple on-disk read chunks.
    """
    chunks_data = get_chunk_sizes(INDEXING_FILE_PATH)
    for chunk_spec in chunks_data:
        start = int(chunk_spec["start"])
        end = int(chunk_spec["end"])
        logical_pcd = o3d.geometry.PointCloud()
        file_base = 0
        with laspy.open(LAS_FILE_PATH) as f:
            for las_chunk in f.chunk_iterator(_KD_TREE_LAS_READ_CHUNK_SIZE):
                n = len(las_chunk.x)
                chunk_end = file_base + n
                overlap_lo = max(start, file_base)
                overlap_hi = min(end, chunk_end)
                if overlap_lo < overlap_hi:
                    i0 = overlap_lo - file_base
                    i1 = overlap_hi - file_base
                    x = np.asarray(las_chunk.x[i0:i1], dtype=np.float64)
                    y = np.asarray(las_chunk.y[i0:i1], dtype=np.float64)
                    z = np.asarray(las_chunk.z[i0:i1], dtype=np.float64)
                    xyz = np.column_stack((x, y, z))
                    part = o3d.geometry.PointCloud()
                    part.points = o3d.utility.Vector3dVector(xyz)
                    logical_pcd += part
                file_base = chunk_end
                if file_base >= end:
                    break
        yield logical_pcd


config = config.SegmentorConfig("config.yml")

pcd_color = [0.5, 0.5, 0.5] # gray
all_new_pcd = o3d.geometry.PointCloud()
for index, chunk_pcd in enumerate(kd_tree()):
    chunk_pcd.paint_uniform_color(pcd_color)
    print(f"Processing chunk {index + 1}")
    new_chunk_pcd = o3d.geometry.PointCloud()
    floor_plane_data, remain_pcd = segment_ground(chunk_pcd, debug=False)
    wall_plane_data, remain_pcd = segment_walls(remain_pcd)
    cieling_plane_data, remain_pcd = cieling_segmentation(remain_pcd, debug=False)
    # other_structure_plane_data, remain_pcd = other_structure_segmentation(remain_pcd, debug=True)

    # objects_pcd, remain_pcd = segment_objects(remain_pcd, debug=False)
    new_chunk_pcd += remain_pcd
    for p in floor_plane_data:
        new_chunk_pcd += p["pcd"]
    for p in wall_plane_data:
        new_chunk_pcd += p["pcd"]
    for p in cieling_plane_data:
        new_chunk_pcd += p["pcd"]
    # for p in other_structure_plane_data:
    #     new_chunk_pcd += p["pcd"]
    # for p in objects_pcd:
    #     new_chunk_pcd += p
    all_new_pcd += new_chunk_pcd

    o3d.visualization.draw_geometries([new_chunk_pcd])


o3d.visualization.draw_geometries([all_new_pcd])