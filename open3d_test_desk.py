from math import pi
import open3d as o3d
import numpy as np
import transforms3d
import math

if __name__ == "__main__":
    # use this to visualize the pointcloud
    cloud = o3d.io.read_point_cloud("/media/ExtHDD/yuying/peract_real_test/kitchen_3_1_cabinet/real1/pcd58.ply")
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    print(points.shape, colors.shape)

    desk2camera = [[0.9999181075257777, 0.012558288952198796, 0.002463254079527643, -0.17409898059817605], [-0.0025892709454548683, 0.01002819281479538, 0.9999463640740138, 0.30000481310346677], [0.012532913389880709, -0.9998708540243897, 0.010059888393982353, 0.8866193118943917], [0.0, 0.0, 0.0, 1.0]]

    # RECALIBRATE
    adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])t 
    adjust_pos_mat = np.array([[1, 0, 0, -0.20], [0, 1, 0, 0.01], [0, 0, 1, 0], [0, 0, 0, 1]]) # manually adjust 
    base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
    cam2base = np.linalg.inv(base2camera)
    print(cam2base.reshape(-1).tolist())

    gl2cv = transforms3d.euler.euler2mat(np.pi, 0, 0)
    print(gl2cv)
    gl2cv_homo = np.eye(4)
    gl2cv_homo[:3, :3] = gl2cv
    cam2base = cam2base @ gl2cv_homo

    valid_bool = np.linalg.norm(points, axis=1) < 3.0
    points = points[valid_bool]
    colors = colors[valid_bool]

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_points = points @ cam2base[:3, :3].T + cam2base[:3, 3]
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pcd.colors = o3d.utility.Vector3dVector(colors)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    
    o3d.visualization.draw_geometries(geometry_list=[transformed_pcd, coordinate])


