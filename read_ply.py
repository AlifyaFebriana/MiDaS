import open3d as o3d

def visualize_point_cloud(ply_file):
    # load ply file
    point_cloud = o3d.io.read_point_cloud(ply_file)
    print(point_cloud)

    # visualize
    o3d.visualization.draw_geometries([point_cloud])

visualize_point_cloud("/home/junaid/alifya/Bean/3DBEAN/point/longberry (1191).ply")