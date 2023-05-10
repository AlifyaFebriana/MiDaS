import cv2
import os
import open3d as o3d
import numpy as np

def depth_map_to_point_cloud(depth_map_path, rgb_image_path, ply_path, fx, fy, cx, cy):

    #load the depth map
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)

    # convert the depth map to a float array
    depth_map = depth_map.astype(np.float32)

    # load the RGB image
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # generate the point cloud from the depth map and the camera intrinsics parameter
    # monocular images
    # a little bit complicated
    # http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
    height, width = depth_map.shape

    # create an aray for the x and y coordinates of each pixel
    x_range = np.linspace(0, width - 1, width)
    y_range = np.linspace(0, height - 1, height)

    # create meshgrid (2d arrays)
    x, y = np.meshgrid(x_range, y_range)

    # initialize the 3d array that used to store the 3d points generated from depth map
    # 3d array contain x, y, and z coordinates
    points_3D = np.zeros((height, width, 3), dtype=np.float32)
    points_3D[:, :, 0] = (x - cx) * depth_map / fx
    points_3D[:, :, 1] = (y - cy) * depth_map / fy
    points_3D[:, :, 2] = depth_map # z coordinate

    # convert the 3D points to an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3D.reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_image.reshape(-1, 3) / 255.0)

    # save the point cloud to a .ply file
    o3d.io.write_point_cloud(ply_path, point_cloud)


# call the function
depth_map_dir = "/home/junaid/alifya/Bean/3DBEAN/images"
rgb_image_dir = "/home/junaid/alifya/Bean/3DBEAN/depth"
output_ply_dir = "/home/junaid/alifya/Bean/dataset"

# camera intrinsics parameter
fx, fy, cx, cy = 5346.726457399102, 5334.765100671141, 128.0, 128.0

# process all the depth map in the train folder
for file in os.listdir(depth_map_dir):
    if file.endswith(".png"):
        depth_map_path = os.path.join(depth_map_dir, file)

        # get the base name file without the extension and remove the additional name in the depth map directory
        base_filename = os.path.splitext(file)[0]

        # construct the rgb image path with a correct extension
        rgb_image_path = os.path.join(rgb_image_dir, f"{base_filename}.jpg")

        # construct the ply file to each 3d point cloud
        ply_path = os.path.join(output_ply_dir, f"{base_filename}.ply")

        depth_map_to_point_cloud(depth_map_path, rgb_image_path, ply_path, fx, fy, cx, cy)




