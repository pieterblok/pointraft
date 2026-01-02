import numpy as np
import open3d as o3d

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch_fpsample

from trac_msgs.msg import Output
from o3d_msgs.msg import Open3D
from sensor_msgs.msg import PointField
import sensor_msgs_py.point_cloud2 as pc2

from trac.prft_utils.models import *
model_dict = {'pointraft': PointRAFT}


class PointRaft:
    def __init__(self, weights, num_points, conveyor_depths, max_height):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = 'pointraft'
        self.model = model_dict[model_name]().to(self.device)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

        ## transformations
        self.pre_transform = T.Center()

        ## custom parameters
        self.num_points = num_points
        self.conveyor_depths = conveyor_depths
        self.max_height = max_height
        

    def run_once(self, radius=30):
        theta = np.random.uniform(0, 2 * np.pi, self.num_points)
        phi = np.random.uniform(0, 0.5 * np.pi, self.num_points)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points = np.stack((x, y, z), axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        _ = self.inference(pcd)


    def inference(self, pcd, height_key=None):
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float)

        ## height embedding
        if height_key is not None:
            try:
                z_max = self.conveyor_depths[str(height_key)]
                z_min = torch.min(points[:, 2])
                height = z_max - z_min
                height = torch.clamp(height, min=0.0, max=self.max_height)
            except:
                height = self.max_height / 2
        else:
            height = 0.0

        data = Data(pos=points)
        data = self.pre_transform(data)
        points = data.pos

        if points.size(0) > self.num_points:  
            points, _ = torch_fpsample.sample(points, self.num_points)

        data = Data(pos=points, height=torch.tensor([height], dtype=torch.float32))
        data.batch = torch.zeros(points.size(0), dtype=torch.int64)
        data = data.to(self.device)

        with torch.no_grad():
            output = self.model(data)
            weight = output.cpu().numpy()

        return weight, height
    

    def pcd_msg(self, pcd, o3d_msgs, mode="xyz"):
        if mode == "xyzrgb":
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            data = np.hstack((points, colors))
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="r", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="g", offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name="b", offset=20, datatype=PointField.FLOAT32, count=1),
            ]

            pcd_msg = pc2.create_cloud(o3d_msgs.header, fields, data)

        else:
            data = np.asarray(pcd.points)
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            pcd_msg = pc2.create_cloud(o3d_msgs.header, fields, data)

        return pcd_msg
    

    def output_message(self, output_msgs, gps_msg, weight, height, track_id):
        output_msg = Output()

        output_msg.height = float(height*1e3)
        output_msg.track_id = int(track_id)
        weight = np.clip(weight, 5, 650)
        output_msg.weight = float(weight)

        if gps_msg != []:
            output_msg.gps.lat = gps_msg.lat
            output_msg.gps.lon = gps_msg.lon
            output_msg.gps.alt = gps_msg.alt
            output_msg.gps.lat_trans = gps_msg.lat_trans
            output_msg.gps.lon_trans = gps_msg.lon_trans
            output_msg.gps.qual = gps_msg.qual

        output_msgs.outputs.append(output_msg)

        return output_msgs
    

    def o3d_message(self, o3d_msgs, track_id, pcd):
        o3d_msg = Open3D()

        o3d_msg.track_id = int(track_id)
        o3d_msg.pcd = self.pcd_msg(pcd, o3d_msgs, mode="xyz")
        o3d_msgs.open3d.append(o3d_msg)

        return o3d_msgs