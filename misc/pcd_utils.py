import numpy as np
import json
import open3d as o3d


class PointCloudUtils:
    def __init__(self, intrinsics, depth_scale, depth_trunc):
        with open(intrinsics) as json_file:
            intric = json.load(json_file)
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(intric['width'], intric['height'], intric['intrinsic_matrix'][0], intric['intrinsic_matrix'][4], intric['intrinsic_matrix'][6], intric['intrinsic_matrix'][7])
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc


    def adjust_intrinsics(self, intrinsic, x_min, y_min, x_max, y_max):
        new_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        new_intrinsic.set_intrinsics(
                width=x_max-x_min,
                height=y_max-y_min,
                fx=intrinsic.intrinsic_matrix[0, 0],
                fy=intrinsic.intrinsic_matrix[1, 1],
                cx=intrinsic.intrinsic_matrix[0, 2] - x_min,
                cy=intrinsic.intrinsic_matrix[1, 2] - y_min
            )
        return new_intrinsic
    

    def remove_distant_clusters(self, pcd, eps=0.05, min_points=100, distance_threshold=1.0):
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

        # Convert point cloud to numpy array of points
        points = np.asarray(pcd.points)
        
        # Perform DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

        # Identify the largest cluster and filter by distance from origin
        filtered_points = []
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # Skip noise points (label -1)
            
            # Extract points and corresponding colors belonging to the current cluster
            cluster_points = points[labels == cluster_id]
            
            # Check if the cluster is within the specified distance threshold
            centroid = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(centroid)  # Calculate distance from origin (0, 0, 0)
            
            if distance <= distance_threshold:
                filtered_points.append(cluster_points)

        # Combine the filtered points and colors
        filtered_points = np.vstack(filtered_points)

        # Create a new point cloud from the filtered points and colors
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        return filtered_pcd    
    

    def create_pcd(self, depth, mask, x_min, y_min, x_max, y_max):
        mask_img = mask[y_min:y_max, x_min:x_max]
        mask_img = mask_img // 255

        depth_slice = depth[y_min:y_max, x_min:x_max]
        depth_mask = np.multiply(depth_slice, mask_img)
        depth_input = o3d.geometry.Image(depth_mask.astype(np.uint16))
        
        new_intrinsics = self.adjust_intrinsics(self.intrinsics, x_min, y_min, x_max, y_max)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_input, new_intrinsics, depth_scale=self.depth_scale, depth_trunc=self.depth_trunc)

        return pcd