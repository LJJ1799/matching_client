import torch
import open3d as o3d
from torch.utils.data import Dataset
import os
import numpy as np
import xml.etree.ElementTree as ET
import json
import os
from tools import get_weld_info

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)

# XML文件路径
def find_xml_files(directory):
    """ 查找给定目录及其子目录下的所有XML文件 """
    xml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files


def get_json(xml_file_path):
    # 读取XML文件并解析
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 初始化存储结果的字典
    data_separated = []

    # 遍历所有的SNaht元素
    for snaht in root.findall('.//SNaht'):
        # 提取点云文件名
        point_cloud_file_name = snaht.attrib['Name']

        # 在SNaht元素下查找所有Frame标签
        for frame in snaht.findall('.//Frame'):
            # 提取Pos标签
            pos = frame.find('Pos')
            weld_position = [float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])]

            # 提取旋转矩阵
            x_vek = frame.find('XVek')
            y_vek = frame.find('YVek')
            z_vek = frame.find('ZVek')

            # rotation_matrix = [
            #     [float(x_vek.attrib['X']), float(x_vek.attrib['Y']), float(x_vek.attrib['Z'])],
            #     [float(y_vek.attrib['X']), float(y_vek.attrib['Y']), float(y_vek.attrib['Z'])],
            #     [float(z_vek.attrib['X']), float(z_vek.attrib['Y']), float(z_vek.attrib['Z'])]
            # ]
            rotation_matrix = [
                [float(x_vek.attrib['X']), float(y_vek.attrib['X']), float(z_vek.attrib['X'])],
                [float(x_vek.attrib['Y']), float(y_vek.attrib['Y']), float(z_vek.attrib['Y'])],
                [float(x_vek.attrib['Z']), float(y_vek.attrib['Z']), float(z_vek.attrib['Z'])]
            ]
            # 创建或更新字典
            data_separated.append({
                "point_cloud_file_name": os.path.join('\\'.join(xml_file_path.split('\\')[:-1]),
                                                      (point_cloud_file_name + '.pcd')),
                "weld_position": weld_position,
                "rotation_matrix": rotation_matrix
            })
    print(data_separated[:3])
    return data_separated


# directory_path = 'data'
# xml_files = find_xml_files(directory_path)
#
# result_json = []
# 打印找到的XML文件路径
# for path in xml_files:
#     result_json += get_json(path)

# 将结果保存为JSON文件
# with open('dataset/data.json', 'w') as file:
#     json.dump(result_json, file)
#
# file_path = 'data/torch/MRW510_10GH.obj'

# 读取焊枪模型
def read_obj(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    vertices = []
    faces = []

    for line in lines:
        if line.startswith('v '):
            vertex = [float(value) for value in line.split()[1:]]
            vertices.append(vertex)
        elif line.startswith('f '):
            face = [int(value.split('/')[0]) for value in line.split()[1:]]
            faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices

# vertices = read_obj(file_path)
# vertices

import torch
from torch.utils.data import Dataset
import json

class PointCloudDataset(Dataset):
    def __init__(self, json_data, welding_gun_pcd):
        """
        Args:
            json_data (list): 包含点云文件名、焊点位置和旋转矩阵的数据列表。
        """
        self.data = json_data
        self.welding_gun_pcd = welding_gun_pcd

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从json数据中获取点云文件名、焊点位置和旋转矩阵
        item = self.data[idx]
        point_cloud_file_name = item['point_cloud_file_name']
        weld_position = torch.tensor(item['weld_position'], dtype=torch.float32)
        rotation_matrix = torch.tensor(item['rotation_matrix'], dtype=torch.float32)

        pcd = o3d.io.read_point_cloud(point_cloud_file_name)
        point_cloud = torch.tensor(pcd.points, dtype=torch.float32)

        return point_cloud.cuda(), weld_position.cuda(), rotation_matrix.cuda()

# 读取JSON文件
# json_file_path = 'dataset/data.json'

# 加载焊枪点云
welding_gun_pcd = read_obj('./data/torch/MRW510_10GH.obj')
welding_gun_pcd = torch.tensor(welding_gun_pcd, dtype=torch.float32)
# print('--------------------')
# print(welding_gun_pcd)
# print('--------------------')
# 加载JSON数据
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
# dataset = PointCloudDataset(data, welding_gun_pcd)

# 检查第一个元素
# dataset[0]
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.view(batchsize, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = torch.eye(self.k).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointCloudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 输出旋转矩阵

    def forward(self, point_cloud, weld_position, welding_gun_pcd):
        # print(point_cloud.is_cuda,weld_position.is_cuda,welding_gun_pcd.is_cuda)
        x = torch.cat([point_cloud, weld_position.unsqueeze(1), welding_gun_pcd.cuda()], dim=1)
        x = x.transpose(2, 1)

        trans = self.tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        rotation_matrix = x.view(-1, 3, 3)
        return rotation_matrix

def poseestimation(wz_path,xml_path):
    weld_infos = get_weld_info(xml_path)
    weld_infos = np.vstack(weld_infos)
    # print(weld_infos[0,:])
    device = torch.device('cuda:0')
    model = PointCloudNet()
    model.load_state_dict(torch.load(os.path.join(ROOT,'model.pth')))
    model=model.to(device)
    true_matrices = []
    predicted_matrices = []
    with torch.no_grad():
        for weld_info in weld_infos:
            # print(weld_info)
            # pcd = o3d.io.read_point_cloud(str('../data/Reisch/'+weld_info[0]+'.pcd'))
            pcd = o3d.io.read_point_cloud(os.path.join(wz_path,weld_info[0]+'.pcd'))
            point_cloud = torch.tensor(pcd.points, dtype=torch.float32).cuda()
            weld_position=torch.tensor(weld_info[4:7].astype(float),dtype=torch.float32).cuda()
            rotation_matrix=np.zeros((3,3))
            rotation_matrix[0,0:3]=weld_info[17:20].astype(float)
            rotation_matrix[1, 0:3] = weld_info[20:23].astype(float)
            rotation_matrix[2, 0:3] = weld_info[23:26].astype(float)
            # print(rotation_matrix)
            rotation_matrix=torch.tensor(rotation_matrix,dtype=torch.float32).cuda()
            # print(point_cloud,weld_position,rotation_matrix)
            # print(weld_info)
        # point_cloud, weld_position, rotation_matrix = dataset[0]

            predicted_rotation_matrix = model(point_cloud.unsqueeze(0), weld_position.unsqueeze(0), welding_gun_pcd.unsqueeze(0))
            true_matrices.append(rotation_matrix)
            predicted_matrices.append(predicted_rotation_matrix.squeeze(0))
            print(predicted_rotation_matrix)
        true_matrices = torch.stack(true_matrices)
        predicted_matrices = torch.stack(predicted_matrices)
        mse = torch.mean((true_matrices - predicted_matrices) ** 2)
        print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    main()
    # model = PointCloudNet()
    # point_cloud, weld_position, rotation_matrix = dataset[0]
    #
    # (point_cloud.unsqueeze(0), weld_position.unsqueeze(0), welding_gun_pcd.unsqueeze(0))
    #
    # predicted_rotation_matrix = model(point_cloud.unsqueeze(0), weld_position.unsqueeze(0), welding_gun_pcd.unsqueeze(0))
    # print(predicted_rotation_matrix)
# predicted_rotation_matrix

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
#
# def rotation_matrix_to_angle_axis(rotation_matrix):
#     rotation_matrix = rotation_matrix.reshape(-1, 3, 3).float()
#
#     cos_theta = (rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2] - 1) / 2
#     cos_theta = torch.clamp(cos_theta, -1, 1)
#     theta = torch.acos(cos_theta)
#
#     axis = torch.stack([
#         rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2],
#         rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0],
#         rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]
#     ], dim=1) / (2 * torch.sin(theta).unsqueeze(1))
#
#     return theta, axis
#
# class RotationVectorLoss(nn.Module):
#     def __init__(self):
#         super(RotationVectorLoss, self).__init__()
#
#     def forward(self, pred_rot_matrix, true_rot_matrix):
#         pred_theta, pred_axis = rotation_matrix_to_angle_axis(pred_rot_matrix)
#         true_theta, true_axis = rotation_matrix_to_angle_axis(true_rot_matrix)
#
#         loss = torch.mean((pred_theta - true_theta)**2 + torch.sum((pred_axis - true_axis)**2, dim=1))
#         return loss
#
#
# import torch.optim as optim
#
# # 初始化模型
# model = PointCloudNet().cuda()
# loss_function = nn.MSELoss()
#
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 设置训练参数
# epochs = 2
# welding_gun_pcd = welding_gun_pcd.cuda()
#
# for epoch in range(epochs):
#     for i, data in enumerate(dataset):
#         point_cloud, weld_position, true_rotation_matrix = data
#         point_cloud = point_cloud.cuda()
#         weld_position = weld_position.cuda()
#         true_rotation_matrix = true_rotation_matrix.cuda()
#
#         try:
#             optimizer.zero_grad()
#             predicted_rotation_matrix = model(point_cloud.unsqueeze(0), weld_position.unsqueeze(0),
#                                               welding_gun_pcd.unsqueeze(0))
#
#             loss = loss_function(predicted_rotation_matrix, true_rotation_matrix.unsqueeze(0))
#
#             loss.backward()
#             optimizer.step()
#         except:
#             print('error:')
#
#         # 打印日志
#         if i % 1000 == 0:  # 每10个批次打印一次
#             print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataset)}], Loss: {loss.item()}")
#             torch.save(model.state_dict(), 'model.pth')
#
# test_dataset = PointCloudDataset(data, welding_gun_pcd)
#
# model.eval()
#
# # 用于存储真实和预测的旋转矩阵
# true_matrices = []
# predicted_matrices = []
#
# with torch.no_grad():
#     for point_cloud, weld_position, true_rotation_matrix in dataset:
#         # 获取预测结果
#         predicted_matrix = model(point_cloud.unsqueeze(0).cuda(), weld_position.unsqueeze(0).cuda(), welding_gun_pcd.unsqueeze(0).cuda())
#
#         # 保存预测和真实值
#         true_matrices.append(true_rotation_matrix)
#         predicted_matrices.append(predicted_matrix.squeeze(0))
#
# true_matrices = torch.stack(true_matrices)
# predicted_matrices = torch.stack(predicted_matrices)
# mse = torch.mean((true_matrices - predicted_matrices) ** 2)
# print(f"Mean Squared Error: {mse}")
#
# point_cloud, weld_position, rotation_matrix = dataset[0]
# model = model.cuda()
# model.eval()
# welding_gun_pcd = welding_gun_pcd.cuda()
# predicted_rotation_matrix = model(point_cloud.unsqueeze(0), weld_position.unsqueeze(0), welding_gun_pcd.unsqueeze(0))
# predicted_rotation_matrix, weld_position
#
# import open3d as o3d
# import numpy as np
# import torch
#
# def apply_rotation(point_cloud, rotation_matrix):
#     """ 应用旋转矩阵到点云 """
#     return np.dot(point_cloud, rotation_matrix.T)
#
# def visualize_point_clouds(pcd1, pcd2):
#     """ 可视化两个点云 """
#     pcd1.paint_uniform_color([1, 0, 0])  # 工件点云为红色
#     pcd2.paint_uniform_color([0, 1, 0])  # 焊枪点云为绿色
#     o3d.visualization.draw_geometries([pcd1, pcd2])
#
# # workpiece_pcd_np: 工件点云的NumPy数组
# # welding_gun_pcd_np: 焊枪点云的NumPy数组
# # rotation_matrix_np: 旋转矩阵的NumPy数组
# welding_gun_pcd_np = read_obj('焊枪/MRW510_10GH.obj')
# workpiece_pcd_np = o3d.io.read_point_cloud('dataset/1/AutoDetect_82_0.pcd').points
# # rotation_matrix_np = np.array([[-1, 2.232692517241787e-14, 8.659560562355031e-17],
# #                                [-1.584875253241849e-14, -0.7071067811865394, -0.7071067811865557],
#                             #    [-1.572628785250411e-14, -0.7071067811865557, 0.7071067811865394]])
# # rotation_matrix_np = np.array([[-0.1441952544170425, -4.018721652665657e-16, -0.9895492552690869],
# #                                 [-0.6890239540753798, -0.7177500772647247, 0.1004032733371307],
# #                                 [-0.710249054426638, 0.6963008161610049, 0.1034961549990386]])
# rotation_matrix_np = predicted_rotation_matrix.squeeze(0).detach().numpy()
# weld_position = np.array([-80.58491709269771, 4.000000000000018, -108.2453207309909])
# print(rotation_matrix_np)
# print()
# # 将NumPy数组转换为Open3D点云
# workpiece_pcd = o3d.geometry.PointCloud()
# workpiece_pcd.points = o3d.utility.Vector3dVector(workpiece_pcd_np)
#
# welding_gun_pcd = o3d.geometry.PointCloud()
# welding_gun_pcd.points = o3d.utility.Vector3dVector(welding_gun_pcd_np)
# # 将焊枪根据焊点位置平移
# welding_gun_pcd_np -= weld_position
#
# # 应用旋转矩阵
# rotated_welding_gun_pcd_np = apply_rotation(welding_gun_pcd_np, rotation_matrix_np)
# rotated_welding_gun_pcd = o3d.geometry.PointCloud()
# rotated_welding_gun_pcd.points = o3d.utility.Vector3dVector(rotated_welding_gun_pcd_np)
#
# # 可视化合并后的点云
#
# visualize_point_clouds(workpiece_pcd, welding_gun_pcd)
# visualize_point_clouds(workpiece_pcd, rotated_welding_gun_pcd)