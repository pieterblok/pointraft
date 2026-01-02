import os
from pathlib import Path
import cv2
import open3d as o3d
import numpy as np
import json


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def load_intrinsics(intrinsics_file):
    with open(intrinsics_file) as json_file:
        data = json.load(json_file)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(data['width'], data['height'], data['intrinsic_matrix'][0], data['intrinsic_matrix'][4], data['intrinsic_matrix'][6], data['intrinsic_matrix'][7])

    return intrinsics


def remove_distant_clusters(pcd, eps=0.05, min_points=100, distance_threshold=1.0):
    """
    Remove distant clusters of low-density points from the point cloud and retain color.
    
    Parameters:
    - pcd: The input Open3D point cloud.
    - eps: The maximum distance between two points to be considered as in the same neighborhood (DBSCAN parameter).
    - min_points: The minimum number of points required to form a cluster (DBSCAN parameter).
    - distance_threshold: The maximum distance from the origin to keep clusters.
    
    Returns:
    - Filtered point cloud after removing distant clusters, with original color values.
    """
    # Convert point cloud to numpy array of points and colors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # Identify the largest cluster and filter by distance from origin
    filtered_points = []
    filtered_colors = []
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # Skip noise points (label -1)
        
        # Extract points and corresponding colors belonging to the current cluster
        cluster_points = points[labels == cluster_id]
        cluster_colors = colors[labels == cluster_id]
        
        # Check if the cluster is within the specified distance threshold
        centroid = np.mean(cluster_points, axis=0)
        distance = np.linalg.norm(centroid)  # Calculate distance from origin (0, 0, 0)
        
        if distance <= distance_threshold:
            filtered_points.append(cluster_points)
            filtered_colors.append(cluster_colors)

    # Combine the filtered points and colors
    filtered_points = np.vstack(filtered_points)
    filtered_colors = np.vstack(filtered_colors)

    # Create a new point cloud from the filtered points and colors
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    return filtered_pcd


def save_pcd(rgb, depth, mask, intrinsics, file_name, write_folder, visualize=False):
    rgbmask = np.multiply(rgb, np.expand_dims(mask, axis=2))
    depthmask = np.multiply(depth, mask)

    rgb_mask = o3d.geometry.Image((rgbmask[:,:,::-1]).astype(np.uint8))
    depth_mask = o3d.geometry.Image(depthmask)
    rgbd_mask = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_mask, depth_mask, depth_scale=1000.0, depth_trunc=0.5, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_mask, intrinsics)
    pcd_filtered = remove_distant_clusters(pcd, eps=0.01, min_points=200, distance_threshold=1)
    pcd.paint_uniform_color([1, 0, 0])

    if visualize:
        o3d.visualization.draw_geometries([pcd, pcd_filtered], window_name="The red points are removed from the original point cloud. Press 'Q' to close screen.")

    subfolder = file_name.split("/")[-2]
    folder_name = os.path.join(write_folder, subfolder)
    check_direxcist(folder_name)
    basename = os.path.basename(file_name)
    fileext = os.path.splitext(basename)[-1]

    write_name = basename.replace("_rgb", "_pcd").replace(fileext, ".ply")
    o3d.io.write_point_cloud(os.path.join(folder_name, write_name), pcd_filtered)


def prepare_dataset(img_root, intrinsics, write_folder_pcd):
    supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")

    rgbaimages = []
    depthimages = []

    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.lower().endswith(supported_cv2_formats):
                if "_rgb" in file:
                    rgbaimages.append(os.path.join(root, file))
                elif "_depth" in file:
                    depthimages.append(os.path.join(root, file))

    rgbaimages.sort()
    depthimages.sort()

    for rgbafile, depthfile in zip(rgbaimages, depthimages):
        rgba = cv2.imread(rgbafile, cv2.IMREAD_UNCHANGED)
        rgb = rgba[:,:,:-1]
        mask = rgba[:,:,-1]

        depth = cv2.imread(depthfile, -1)

        save_pcd(rgb, depth, mask.astype(bool), intrinsics, rgbafile, write_folder_pcd)

        height, width, _ = rgb.shape
        mask_img = np.zeros((height, width)).astype(np.uint8)
        mask_img = cv2.add(mask_img, mask)
        mask_img = np.expand_dims(mask_img, -1)
        mask_img = np.repeat(mask_img, 3, axis=2)
        img_vis = cv2.addWeighted(rgb, 0.5, mask_img, 0.5, 0)
        cv2.imshow('Image with Annotations', img_vis)
        cv2.waitKey(1)

    cv2.destroyAllWindows()



if __name__ == '__main__':
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin', '1_rgbd')

    img_root = os.path.join(datafolder, "1_image")
    intrinsics = load_intrinsics(os.path.join(datafolder, "0_camera_intrinsics/realsense_d405_camera_intrinsic.json"))
    write_folder_pcd = os.path.join(datafolder, "2_pcd")
    check_direxcist(write_folder_pcd)
    
    prepare_dataset(img_root, intrinsics, write_folder_pcd)