

import os
from typing import Any
import numpy as np
import open3d as o3d
from modules.models.las_tools import las_io
from modules.utils import constants
from modules.models.cityjson_export import write_cityjson_mesh_part
from modules.models.mesh_tools.ball_neighbourhood_mesh_v2 import create_ball_neighbourhood_mesh as create_ball_neighbourhood_mesh_v2
from modules.models.mesh_tools.ball_neighbourhood_mesh import create_ball_neighbourhood_mesh as create_ball_neighbourhood_mesh_v1

TEST_CLUSTER_IDS = [4, 7, 10, 69, 73, 97]
MAIN_PCD, MAIN_PA = las_io.read_full_las_file("/home/mohammedalsalihi/LidarData/for_mohammed/fh_underground_result/3_post_process.las")
RESULT_FOLDER_PATH = "/home/mohammedalsalihi/LidarData/for_mohammed/fh_outdoor_result"
OUTPUT_JSON_PATH = "/home/mohammedalsalihi/LidarData/for_mohammed/fh_outdoor_result/under.citygml.city.json"

MAX_ITERATIONS = 5
BALL_RADIUES_ITERATIONS = [1, 3, 5, 6, 9]
BALL_MIN_NEIGHBOURS_ITERATIONS = [10, 7, 4, 2, 1]

if os.path.exists(OUTPUT_JSON_PATH):
    os.remove(OUTPUT_JSON_PATH)

def print_mesh_info(mesh: o3d.geometry.TriangleMesh, name: str):
    print(f"{name} has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    print(f"{name} has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")


def _draw_mesh_double_sided(meshes: list[o3d.geometry.TriangleMesh], window_name: str = "mesh") -> None:
    """
    Visualize mesh with back-face rendering enabled.

    Open3D's default draw path culls back faces. For open/non-watertight meshes
    (common with alpha-shape), this can make surfaces appear visible only from
    one side or only at certain camera angles.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for mesh in meshes:
        vis.add_geometry(mesh)
    render_opt = vis.get_render_option()
    render_opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

def from_pcd_to_mesh(pcd: o3d.geometry.PointCloud, iteration: int, debug: bool = False):
    if 1 > iteration or iteration > MAX_ITERATIONS:
        raise ValueError(f"Iteration must be between 1 and {MAX_ITERATIONS}")
    
    current_pcd = pcd
    for i in range(iteration):
        mesh = create_ball_neighbourhood_mesh_v2(
            current_pcd,
            ball_radius=BALL_RADIUES_ITERATIONS[i],
            min_neighbours=BALL_MIN_NEIGHBOURS_ITERATIONS[i],
            max_faces=10_000_000,
        )
        if debug:
            print(f"Mesh at iteration {i}")
            _draw_mesh_double_sided([mesh])
        to_compress_point = int(len(current_pcd.points) * 0.25)
        current_pcd = mesh.sample_points_uniformly(to_compress_point)
    return mesh



def create_ground_mesh(iteration: int):
    ground_pcd = MAIN_PCD.select_by_index(
        np.where(MAIN_PA["classification"] == constants.PCB_CLASSIFICATION_GROUND)[0]
    )
    mesh = from_pcd_to_mesh(ground_pcd, iteration=iteration)
    return mesh

def create_cieling_mesh(iteration: int):
    cieling_pcd = MAIN_PCD.select_by_index(
        np.where(MAIN_PA["classification"] == constants.PCB_CLASSIFICATION_CEILING)[0]
    )
    mesh = from_pcd_to_mesh(cieling_pcd, iteration=iteration)
    return mesh

def create_wall_meshes(iteration: int, debug: bool = False):
    wall_pcd = MAIN_PCD.select_by_index(
        np.where(MAIN_PA["classification"] == constants.PCB_CLASSIFICATION_WALL)[0]
    )
    point_labelels = wall_pcd.cluster_dbscan(eps=1, min_points=50)
    unique_labels = np.unique(point_labelels)
    if debug:
        print(f"Found {len(unique_labels)} labels")
    all_meshes = []
    for label in unique_labels:
        if label == -1:
            continue
        label_pcd = wall_pcd.select_by_index(np.where(point_labelels == label)[0])
        if debug:
            print(f"Processing label {label}")
            o3d.visualization.draw_geometries([label_pcd])
        try:
            mesh = from_pcd_to_mesh(label_pcd, iteration=iteration, debug=debug)
        except Exception:
            print(f"Error creating mesh for label {label}")
            continue
        mesh.paint_uniform_color(np.random.rand(3))
        if len(mesh.vertices) < 100 or len(mesh.triangles) < 100:
            continue
        all_meshes.append(mesh)
        print(f"Created mesh for label {len(all_meshes)}")
    return all_meshes


def _loop_car_clusters():
    cluster_vals = MAIN_PA["Clusters"]
    class_vals = MAIN_PA["classification"]
    unique_clusters = np.unique(cluster_vals)
    unique_clusters = unique_clusters[unique_clusters > 0]
    for cluster_id in unique_clusters:
        mask = cluster_vals == cluster_id
        if constants.PCB_CLASSIFICATION_CAR not in class_vals[mask]:
            continue
        cluster_pcd = MAIN_PCD.select_by_index(np.where(mask)[0])
        masked_points_attributes = {key: value[mask] for key, value in MAIN_PA.items()}
        yield cluster_id, cluster_pcd, masked_points_attributes

def create_car_meshes():
    all_meshes = []
    for cluster_id, cluster_pcd, _ in _loop_car_clusters():
        print("cluster_id", cluster_id)
        mesh = from_pcd_to_mesh(cluster_pcd, iteration=1)
        mesh.paint_uniform_color(np.random.rand(3))
        all_meshes.append(mesh)
    return all_meshes


ground_mesh = create_ground_mesh(iteration=2)
ground_mesh.paint_uniform_color(np.random.rand(3))
write_cityjson_mesh_part(
    mesh_id="groundSurface",
    mesh=ground_mesh,
    result_folder_path=RESULT_FOLDER_PATH,
    output_json_path=OUTPUT_JSON_PATH,
    append_to_existing=False,
)
print("Created ground mesh")
cieling_mesh = create_cieling_mesh(iteration=2)
cieling_mesh.paint_uniform_color(np.random.rand(3))
write_cityjson_mesh_part(
    mesh_id="cielingSurface",
    mesh=cieling_mesh,
    result_folder_path=RESULT_FOLDER_PATH,
    output_json_path=OUTPUT_JSON_PATH,
    append_to_existing=False,
)
print("Created cieling mesh")
wall_meshes = create_wall_meshes(iteration=1, debug=False)
for wall_mesh_id, wall_mesh in enumerate(wall_meshes):
    write_cityjson_mesh_part(
        mesh_id=f"wallSurface_{wall_mesh_id}",
        mesh=wall_mesh,
        result_folder_path=RESULT_FOLDER_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        append_to_existing=False,
    )
print("Created wall meshes")

car_meshes = create_car_meshes()
for car_mesh_id, car_mesh in enumerate(car_meshes):
    write_cityjson_mesh_part(
        mesh_id=f"carSurface_{car_mesh_id}",
        mesh=car_mesh,
        result_folder_path=RESULT_FOLDER_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        append_to_existing=False,
    )

_draw_mesh_double_sided([ground_mesh, *wall_meshes, *car_meshes])
