## altered from: https://github.com/AndreiMoraru123/PointNet

import torch
import torch.nn as nn


class TNet(nn.Module):
    # paper: https://arxiv.org/pdf/1612.00593.pdf
    def __init__(self, num_points=1024, num_features=3):
        super(TNet, self).__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(self.num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_features * self.num_features)
        self.relu = nn.ReLU()
        self.num_features = num_features

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(num_points)

    def forward(self, x, batch_size):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.num_features, requires_grad=True).float().view(
            1,self.num_features * self.num_features).repeat(batch_size, 1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.num_features, self.num_features)
        return x


class SpatialTransformer(nn.Module):
    """ Spatial Transformer Network """

    def __init__(self, num_points=1024):
        super(SpatialTransformer, self).__init__()
        self.num_points = num_points
        self.input_transform = TNet(num_points=self.num_points, num_features=3)
        self.feature_transform = TNet(num_points=self.num_points, num_features=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.mp1 = nn.MaxPool1d(num_points)
        self.relu = nn.ReLU()

    def forward(self, x, batch_size):
        # 3x3 transform
        tr3x3 = self.input_transform(x, batch_size)
        x = torch.bmm(torch.transpose(x, 1, 2), tr3x3).transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))

        # 64x64 transform
        tr64x64 = self.feature_transform(x, batch_size)
        x = torch.bmm(torch.transpose(x, 1, 2), tr64x64).transpose(1, 2)
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        return x
    

class PointNet(nn.Module):
    def __init__(self, num_points=1024):
        super(PointNet, self).__init__()
        self.stn = SpatialTransformer()
        self.num_points = num_points

        # regression head
        self.fc1 = nn.Linear(self.num_points, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

        self.relu = nn.ReLU()

    def forward(self, data):
        batch_size = data.batch.max().item() + 1
        x = data.pos.view(batch_size, self.num_points, 3)  ## reshape to (batch_size, num_points, 3)
        x = x.permute(0, 2, 1)   ## transpose to (batch_size, 3, num_points)

        x = self.stn(x, batch_size)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x.squeeze(-1)