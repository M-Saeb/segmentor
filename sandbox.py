import open3d as o3d
import numpy as np
import os
from modules.models.las_tools import las_io
from modules.models.pcd_tools import is_plane_facing_side
from modules.controllers.pre_process.utils import _voxel_downsample_preserve_attributes
from modules.utils.config import SegmentorConfig
from modules.utils import constants
from toolbox.restore_origina_pcd import restore_original_points_in_inliers
from modules.models.mesh.ball_neighbourhood_mesh import reconstruct_mesh as reconstruct_mesh_custom
from modules.models.cityjson_export import write_cityjson_mesh_part


def create_mesh_1(pcd: o3d.geometry.PointCloud, alpha: float = 4) -> o3d.geometry.TriangleMesh:
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    mesh.paint_uniform_color([0.0, 0.0, 1.0])  # Add a red color to the mesh
    mesh.compute_vertex_normals()
    return mesh

def create_mesh_ball_pivoting(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    radii = o3d.utility.DoubleVector(list(i for i in np.arange(0.1, 10, 0.5)))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.paint_uniform_color([0.0, 0.0, 1.0])  # Add a red color to the mesh
    mesh.compute_vertex_normals()
    return mesh

def create_mesh_poisson(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, scale=0.5)
    mesh.paint_uniform_color([0.0, 0.0, 1.0])  # Add a red color to the mesh
    mesh.compute_vertex_normals()
    return mesh

def create_ground_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    shape_mesh = reconstruct_mesh_custom(pcd, ball_radius=5, max_faces=1_000_000)
    return shape_mesh
    # concave_hull = concave_hull_polygon_1(pcd.points)
    # shape_mesh = triangle_mesh_from_concave_hull_polygon(concave_hull)
    # return shape_mesh

def get_next_wall_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    remaining_pcd = o3d.geometry.PointCloud(pcd)
    yield remaining_pcd
    return
    for _ in range(100):
        plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)
        is_wall = is_plane_facing_side(plane_model, threshold=0.9)
        if not is_wall:
            print("Not a wall plane")
            continue


        pcd_inlier_layer_1 = remaining_pcd.select_by_index(inliers)
        labels = np.asarray(
            pcd_inlier_layer_1.cluster_dbscan(
                eps=4, min_points=5, print_progress=False
            )
        )
        if labels.size == 0 or np.all(labels == -1):
            print("No labels found")
            continue
        unique_labels = set(labels)
        unique_labels.discard(-1)
        print("unique_labels", len(unique_labels), unique_labels)
        if not unique_labels:
            print("No unique labels found")
            continue
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if cluster_indices.size < 4:
                print("Cluster indices size is less than 4")
                continue
            cluster_pcd = pcd_inlier_layer_1.select_by_index(cluster_indices.tolist())
            yield restore_original_points_in_inliers(cluster_pcd, "/home/mohammedalsalihi/LidarData/for_mohammed/fh_underground_result/1_pre_processing.las")
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    return

def create_wall_meshes(pcd: o3d.geometry.PointCloud) -> list[o3d.geometry.TriangleMesh]:
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # gray color
    shapes_meshes = []
    for wall_pcd in get_next_wall_pcd(pcd):
        wall_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # blue color
        wall_pcd.estimate_normals()
        # concave_hull = concave_hull_polygon_1(wall_pcd.points, plane="pca")
        # display_concave_hull_polygon(concave_hull, points=wall_pcd)
        # shape_mesh = triangle_mesh_from_concave_hull_polygon(concave_hull, plane="pca")
        shape_mesh = reconstruct_mesh_custom(wall_pcd, ball_radius=1, max_faces=1_000_000)
        shape_mesh.paint_uniform_color([0.0, 0.0, 1.0]) # blue color
        shapes_meshes.append(shape_mesh)
    return shapes_meshes


def create_ceiling_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    mesh = reconstruct_mesh_custom(pcd, ball_radius=2, max_faces=1_000_000)
    mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Add a red color to the mesh
    mesh.compute_vertex_normals()
    return mesh



config = SegmentorConfig.initialize("/home/mohammedalsalihi/Workspace/segmentor/config.yml")

pcd, attrs = las_io.read_full_las_file("/home/mohammedalsalihi/LidarData/for_mohammed/fh_underground_result/2_pipeline.las")
# Own LAS attribute memory (arrays can reference the closed reader / mmap).
attrs = {k: np.copy(np.asarray(v)) for k, v in attrs.items()}

classification = attrs["classification"]

classification_mask = np.isin(
    classification,
    [
        constants.PCB_CLASSIFICATION_WALL,
        constants.PCB_CLASSIFICATION_GROUND,
        constants.PCB_CLASSIFICATION_CEILING,
    ],
)
masked_indices = np.where(classification_mask)[0].tolist()
building_pcd = pcd.select_by_index(masked_indices)

n = len(pcd.points)
building_attrs = {
    k: (np.asarray(v)[classification_mask] if np.asarray(v).shape[0] == n else np.asarray(v))
    for k, v in attrs.items()
}

# building_pcd, building_attrs = _voxel_downsample_preserve_attributes(
#     building_pcd, building_attrs, 1
# )

classification = np.asarray(building_attrs["classification"])
ground_mask = np.isin(classification, [constants.PCB_CLASSIFICATION_GROUND])
ground_indices = np.where(ground_mask)[0].tolist()
ground_pcd = building_pcd.select_by_index(ground_indices)
ground_pcd.estimate_normals()
ground_mesh = create_ground_mesh(ground_pcd)
print("finished ground mesh")

wall_mask = np.isin(classification, [constants.PCB_CLASSIFICATION_WALL])
wall_indices = np.where(wall_mask)[0].tolist()
wall_pcd = building_pcd.select_by_index(wall_indices)
wall_meshes = create_wall_meshes(wall_pcd)
print("finished wall meshes")

ceiling_mask = np.isin(classification, [constants.PCB_CLASSIFICATION_CEILING])
ceiling_indices = np.where(ceiling_mask)[0].tolist()
ceiling_pcd = building_pcd.select_by_index(ceiling_indices)
ceiling_meshes = create_ceiling_mesh(ceiling_pcd)
print("finished ceiling meshes")

# output_json_path = "/home/mohammedalsalihi/LidarData/for_mohammed/fh_underground_result/citygml.city.json"
# if os.path.isfile(output_json_path):
#     os.remove(output_json_path)



# write_cityjson_mesh_part(
#     mesh_id="groundSurface",
#     mesh=ground_mesh,
#     result_folder_path="/home/mohammedalsalihi/Workspace/segmentor/results/fh_underground_result",
#     output_json_path=output_json_path,
# )


# for wall_mesh in wall_meshes:
#     write_cityjson_mesh_part(
#         mesh_id="wallSurface",
#         mesh=wall_mesh,
#         result_folder_path="/home/mohammedalsalihi/Workspace/segmentor/results/fh_underground_result",
#         output_json_path=output_json_path,
#         append_to_existing=True,
#     )
# write_cityjson_mesh_part(
#     mesh_id="ceilingSurface",
#     mesh=ceiling_meshes,
#     result_folder_path="/home/mohammedalsalihi/Workspace/segmentor/results/fh_underground_result",
#     output_json_path=output_json_path,
# )

o3d.visualization.draw_geometries([ground_mesh, *wall_meshes, ceiling_meshes], mesh_show_back_face=True)
# o3d.visualization.draw_geometries([ground_mesh], mesh_show_back_face=True)