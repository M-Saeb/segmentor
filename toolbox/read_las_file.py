import laspy
import open3d as o3d
import numpy as np

def read_whole_las_file(file_path):
    las = laspy.read(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((las.x, las.y, las.z)).T)
    return pcd