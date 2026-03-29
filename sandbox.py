import open3d as o3d
import matplotlib.pyplot as plt
import laspy
import numpy as np
from toolbox.density import get_inlier_density


def read_las_file(file_path):
    with laspy.open(file_path) as las:
        for point in las.chunk_iterator(100_000):
            xyz = np.vstack((point.x, point.y, point.z)).T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            return pcd

def write_las_file(file_path, points):
    with laspy.open(file_path, 'w') as las:
        las.points = points

def is_facing_up(plane, threshold=0.95):
    a, b, c, d = plane
    normal = np.array([a, b, c])
    
    # Normalize the normal vector
    normal_unit = normal / np.linalg.norm(normal)
    
    # Check if Z component is close to 1 (facing up) or -1 (facing down)
    return abs(normal_unit[2]) > threshold

def is_facing_side(plane, threshold=0.9):
    a, b, c, d = plane
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    result = abs(normal[2]) < (1 - threshold )
    return result

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
    if result:
        print("area", area)
        o3d.visualization.draw_geometries([pcd_a, pcd_b])
    return result

def is_floor_check(plane_model, pcd, all_floor_plane_data):
    current_plane_z_intercept = -plane_model[3] / plane_model[2]   # -d1 / c1
    if not is_facing_up(plane_model):
        return False
    for item in all_floor_plane_data:
        floor_plane_normal = item["plane_model"]
        floor_plane_z_intercept = -floor_plane_normal[3] / floor_plane_normal[2]   # -d1 / c1
        z_difference = current_plane_z_intercept - floor_plane_z_intercept
        is_above = z_difference > 0.3
        if not is_above:
            continue
        is_colliding = planes_footprint_overlap_xy(
            pcd, item["pcd"],
            min_overlap_area_m2=1
        )
        if is_colliding:
            return False
    return True

def is_wall_check(plane_model, pcd_inlier, all_wall_plane_data):
    point_array = np.asarray(pcd_inlier.points)
    inlier_indices = np.arange(len(point_array))
    density = get_inlier_density(point_array, inlier_indices, plane_model, visualize=False)
    # if density < 30:
    #     return False
    is_facing_correct = is_facing_side(plane_model)
    return is_facing_correct

def is_cieling_check(plane_model, pcd_inlier, all_cieling_plane_data):
    point_array = np.asarray(pcd_inlier.points)
    inlier_indices = np.arange(len(point_array))
    density = get_inlier_density(point_array, inlier_indices, plane_model, visualize=False)
    if density < 30:
        return False
    is_facing_correct = is_facing_up(plane_model)
    return is_facing_correct

def is_other_structure_check(plane_model, pcd_inlier, all_other_structure_plane_data):
    point_array = np.asarray(pcd_inlier.points)
    inlier_indices = np.arange(len(point_array))
    density = get_inlier_density(point_array, inlier_indices, plane_model, visualize=True)
    return density > 30

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

def segment_ground(pcd, visualize=False):
    floor_plane_data = []
    other_planes_data = []
    remain_pcd = pcd
    for i in range(100):        
        plane_model, inliers = remain_pcd.segment_plane(
            distance_threshold=0.1,
            ransac_n=3,
            num_iterations=1000,
        )

        pcd_inliers = remain_pcd.select_by_index(inliers)

        is_floor = is_floor_check(plane_model, pcd_inliers, floor_plane_data)

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

        if visualize:
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

def segment_walls(pcd):
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

        # o3d.visualization.draw_geometries(
        #     [pcd_inliers] + \
        #     [p["pcd"] for p in wall_plane_data] + \
        #     [p["pcd"] for p in other_planes_data] + \
        #     [remain_pcd]
        # )

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

    for p in other_planes_data:
        remain_pcd += p["pcd"]	

    return wall_plane_data, remain_pcd

def cieling_segmentation(pcd):
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
        is_cieling = is_cieling_check(plane_model, pcd_inliers, cieling_plane_data)
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

    for p in other_planes_data:
        remain_pcd += p["pcd"]	

    return cieling_plane_data, remain_pcd

def other_structure_segmentation(pcd):
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
        is_other_structure = is_other_structure_check(plane_model, pcd_inliers, other_structure_plane_data)
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

def segment_objects(pcd, visualize=False):
    labels = np.asarray(pcd.cluster_dbscan(eps=0.3, min_points=30, print_progress=False))
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
        color = get_purple_colors(i % 100)
        cluster_pcd.paint_uniform_color(color)

        if visualize:
            o3d.visualization.draw_geometries([cluster_pcd])

        objects_pcds.append(cluster_pcd)
        all_cluster_indices.extend(indices.tolist())

    remain_pcd = pcd.select_by_index(all_cluster_indices, invert=True)
    return objects_pcds, remain_pcd

pcd = read_las_file("/home/mohammedalsalihi/LidarData/for_mohammed/output.las")
pcd_color = [0.5, 0.5, 0.5] # gray
pcd.paint_uniform_color(pcd_color)
floor_plane_data, remain_pcd = segment_ground(pcd, visualize=False)
wall_plane_data, remain_pcd = segment_walls(remain_pcd)
cieling_plane_data, remain_pcd = cieling_segmentation(remain_pcd)
# other_structure_plane_data, remain_pcd = other_structure_segmentation(remain_pcd)
# objects_pcd, remain_pcd = segment_objects(remain_pcd, visualize=False)
o3d.visualization.draw_geometries(
    [p["pcd"] for p in floor_plane_data] + \
    [p["pcd"] for p in wall_plane_data] + \
    [p["pcd"] for p in cieling_plane_data] + \
    # objects_pcd + \
    [remain_pcd]
)
