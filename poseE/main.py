import torch
import open3d as o3d
from torch.utils.data import Dataset
import os
import numpy as np
import xml.etree.ElementTree as ET
import json
from copy import copy
import os
from tools import get_weld_info
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_PATH)


def get_distance_and_translate(weld_info):
    x_center = (np.max(weld_info[:, 1]) + np.min(weld_info[:, 1])) / 2
    y_center = (np.max(weld_info[:, 2]) + np.min(weld_info[:, 2])) / 2
    z_center = (np.max(weld_info[:, 3]) + np.min(weld_info[:, 3])) / 2
    translate = np.array([x_center, y_center, z_center])
    x_diff = np.max(weld_info[:, 1]) - np.min(weld_info[:, 1])
    if x_diff < 2:
        x_diff = 0
    y_diff = np.max(weld_info[:, 2]) - np.min(weld_info[:, 2])
    if y_diff < 2:
        y_diff = 0
    z_diff = np.max(weld_info[:, 3]) - np.min(weld_info[:, 3])
    if z_diff < 2:
        z_diff = 0
    distance = int(pow(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2), 0.5)) + 50

    return distance, translate



def find_xml_files(directory):
    xml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files


def get_json(xml_file_path):

    tree = ET.parse(xml_file_path)
    root = tree.getroot()


    data_separated = []


    for snaht in root.findall('.//SNaht'):

        point_cloud_file_name = snaht.attrib['Name']


        for frame in snaht.findall('.//Frame'):

            pos = frame.find('Pos')
            weld_position = [float(pos.attrib['X']), float(pos.attrib['Y']), float(pos.attrib['Z'])]


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

            data_separated.append({
                "point_cloud_file_name": os.path.join('\\'.join(xml_file_path.split('\\')[:-1]),
                                                      (point_cloud_file_name + '.pcd')),
                "weld_position": weld_position,
                "rotation_matrix": rotation_matrix
            })
    print(data_separated[:3])
    return data_separated


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
        self.data = json_data
        self.welding_gun_pcd = welding_gun_pcd

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        point_cloud_file_name = item['point_cloud_file_name']
        weld_position = torch.tensor(item['weld_position'], dtype=torch.float32)
        rotation_matrix = torch.tensor(item['rotation_matrix'], dtype=torch.float32)

        pcd = o3d.io.read_point_cloud(point_cloud_file_name)
        point_cloud = torch.tensor(pcd.points, dtype=torch.float32)

        return point_cloud.cuda(), weld_position.cuda(), rotation_matrix.cuda()



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
        self.fc3 = nn.Linear(256, 9)

    def forward(self, point_cloud, weld_position, welding_gun_pcd):
        print(weld_position)
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

def poseestimation(data_path,wz_path,xml_path,SNahts,tree,retrieved_map,vis=False):
    weld_infos = get_weld_info(xml_path)
    weld_infos = np.vstack(weld_infos)
    # print(weld_infos[0,:])
    # device = torch.device('cuda:1')
    model = PointCloudNet().cuda()
    # model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(CURRENT_PATH,'model.pth')))
    # model=model.to(device)
    true_matrices = []
    predicted_matrices = []
    predict_rot_dict={}
    slice_rot_list=[]
    slice_rot_dict={}
    with torch.no_grad():
        for key,vals in retrieved_map.items():
            if key in slice_rot_dict:
                continue
            pcd = o3d.io.read_point_cloud(os.path.join(wz_path, key + '.pcd'))
            pcd.paint_uniform_color([1,0,0])
            point_cloud = torch.tensor(pcd.points, dtype=torch.float32).cuda()
            torch_name=weld_infos[weld_infos[:,0]==key][0,2]
            welding_gun_pcd = read_obj(os.path.join(data_path, 'torch', torch_name + '.obj'))
            welding_gun_pcd = torch.tensor(welding_gun_pcd, dtype=torch.float32)
            weld_info=weld_infos[weld_infos[:,0]==key][:,3:].astype(float)
            _, translate = get_distance_and_translate(weld_info)
            weld_seam = o3d.geometry.PointCloud()
            weld_seam.points = o3d.utility.Vector3dVector(weld_infos[weld_infos[:,0]==key][:,3:][:,1:4].astype(float))
            weld_seam.translate(-translate)
            for i in range(len(weld_info)):
                weld_spot_points=weld_info[i,1:4]
                weld_spot = o3d.geometry.PointCloud()
                weld_spot.points = o3d.utility.Vector3dVector(weld_spot_points.reshape((1,3)))
                weld_spot.translate(-translate)
                translate_weld_spot_points = np.array(weld_spot.points)
                pose_position = torch.tensor(translate_weld_spot_points.reshape(3, ), dtype=torch.float32).cuda()
                rotation_matrix=weld_info[i,14:23].reshape((3,3))
                rotation_matrix = torch.tensor(rotation_matrix.astype(float), dtype=torch.float32).cuda()

                # print(rotation_matrix)
                predicted_rotation_matrix = model(point_cloud.unsqueeze(0), pose_position.unsqueeze(0),
                                                  welding_gun_pcd.unsqueeze(0))
                slice_rot_list.append(predicted_rotation_matrix.squeeze(0))
                predict_rot_dict[key] = predicted_rotation_matrix.squeeze(0)
                true_matrices.append(rotation_matrix)
                predicted_matrices.append(predicted_rotation_matrix.squeeze(0))
            slice_rot_dict[key]=slice_rot_list
            if len(vals)==0:
                continue
            for val in vals:
                if val in slice_rot_dict:
                    continue
                else:
                    slice_rot_dict[val]=slice_rot_list
        true_matrices = torch.stack(true_matrices)
        predicted_matrices = torch.stack(predicted_matrices)
        mse = torch.mean((true_matrices - predicted_matrices) ** 2)
        print(f"Mean Squared Error: {mse}")
        print('slice_rot_dict',slice_rot_dict)
        for SNaht in SNahts:
            slice_name=SNaht.attrib.get('Name')
            rotation_matrix_list=slice_rot_dict[slice_name]
            # print('rotation_matrix_list',rotation_matrix_list,rotation_matrix_list)
            for Frames in SNaht.findall('Frames'):
                Pose_num=len(rotation_matrix_list)
                j=0
                if j>Pose_num:
                    break
                for Frame in Frames.findall('Frame'):
                    for XVek in Frame.findall('XVek'):
                        XVek.set('X', str(np.array(rotation_matrix_list[j][0, 0].cpu())))
                        XVek.set('Y', str(np.array(rotation_matrix_list[j][1, 0].cpu())))
                        XVek.set('Z', str(np.array(rotation_matrix_list[j][2, 0].cpu())))
                    for YVek in Frame.findall('YVek'):
                        YVek.set('X', str(np.array(rotation_matrix_list[j][0, 1].cpu())))
                        YVek.set('Y', str(np.array(rotation_matrix_list[j][1, 1].cpu())))
                        YVek.set('Z', str(np.array(rotation_matrix_list[j][2, 1].cpu())))
                    for ZVek in Frame.findall('YVek'):
                        ZVek.set('X', str(np.array(rotation_matrix_list[j][0, 2].cpu())))
                        ZVek.set('Y', str(np.array(rotation_matrix_list[j][1, 2].cpu())))
                        ZVek.set('Z', str(np.array(rotation_matrix_list[j][2, 2].cpu())))
                j+=1
    return tree


            # for info in weld_info:

            #
            #
            # weld_seam.paint_uniform_color([0,1,0])
            # translate_weld_seam_points=np.array(weld_seam.points)
            #
            # torch_name=weld_infos[weld_infos[:,0]==key][0,2]
            # print('torch_name',torch_name)
            # print(key,'\n',weld_info)
        # for SNaht in SNahts:
        #
            # coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
        #
        #     torch_name=SNaht.attrib.get('WkzName')
        #     welding_gun_pcd = read_obj(os.path.join(data_path,'torch',torch_name+'.obj'))
        #     welding_gun_pcd = torch.tensor(welding_gun_pcd, dtype=torch.float32)
        #     for weld_info in weld_infos:
        #         translate_weld_seam_points=
        #     slice_name = SNaht.attrib.get('Name')
        #     slice_id = SNaht.attrib.get('ID')
        #     weld_info = weld_infos[weld_infos[:, 0] == slice_name][:, 3:].astype(float)
        #     _,translate=get_distance_and_translate(weld_info)
        #     pcd = o3d.io.read_point_cloud(os.path.join(wz_path,slice_name+'.pcd'))
        #     pcd.paint_uniform_color([1,0,0])
        #     point_cloud = torch.tensor(pcd.points, dtype=torch.float32).cuda()
        #
        #     for Frames in SNaht.findall('Frames'):
        #         for Frame in Frames.findall('Frame'):
        #             rotation_matrix = np.zeros((3, 3))
        #             for Pos in Frame.findall('Pos'):
        #                 weld_seam_points=np.array([Pos.get('X'),Pos.get('Y'),Pos.get('Z')]).astype(float)
        #                 weld_seam = o3d.geometry.PointCloud()
        #                 weld_seam.points = o3d.utility.Vector3dVector(weld_seam_points.reshape(1,3))
        #                 weld_seam.translate(-translate)
        #                 weld_seam.paint_uniform_color([0,1,0])
        #                 translate_weld_seam_points=np.array(weld_seam.points)
        #                 # print('weld_seam_points',weld_seam_points)
        #                 pose_position=torch.tensor(translate_weld_seam_points.reshape(3,),dtype=torch.float32).cuda()
        #             for XVek in Frame.findall('XVek'):
        #                 # 3x3 rotation
        #                 Xrot = np.array([XVek.get('X'), XVek.get('Y'), XVek.get('Z')])
        #                 rotation_matrix[0:3, 0] = Xrot
        #             for YVek in Frame.findall('YVek'):
        #                 # 3x3 rotation
        #                 Yrot = np.array([YVek.get('X'), YVek.get('Y'), YVek.get('Z')])
        #                 rotation_matrix[0:3, 1] = Yrot
        #             for ZVek in Frame.findall('ZVek'):
        #                 # 3x3 rotation
        #                 Zrot = np.array([ZVek.get('X'), ZVek.get('Y'), ZVek.get('Z')])
        #                 rotation_matrix[0:3, 2] = Zrot
        #             rotation_matrix = torch.tensor(rotation_matrix.astype(float), dtype=torch.float32).cuda()
        #             predicted_rotation_matrix = model(point_cloud.unsqueeze(0), pose_position.unsqueeze(0),
        #                                               welding_gun_pcd.unsqueeze(0))
        #             for XVek in Frame.findall('XVek'):
        #                 XVek.set('X', str(np.array(rotation_matrix[0, 0].cpu())))
        #                 XVek.set('Y', str(np.array(rotation_matrix[1, 0].cpu())))
        #                 XVek.set('Z', str(np.array(rotation_matrix[2, 0].cpu())))
        #             for YVek in Frame.findall('YVek'):
        #                 YVek.set('X', str(np.array(rotation_matrix[0, 1].cpu())))
        #                 YVek.set('Y', str(np.array(rotation_matrix[1, 1].cpu())))
        #                 YVek.set('Z', str(np.array(rotation_matrix[2, 1].cpu())))
        #             for ZVek in Frame.findall('YVek'):
        #                 ZVek.set('X', str(np.array(rotation_matrix[0, 2].cpu())))
        #                 ZVek.set('Y', str(np.array(rotation_matrix[1, 2].cpu())))
        #                 ZVek.set('Z', str(np.array(rotation_matrix[2, 2].cpu())))
        #             # print(rotation_matrix)
        #             predict_rot_dict[slice_name] = predicted_rotation_matrix
        #             true_matrices.append(rotation_matrix)
        #             predicted_matrices.append(predicted_rotation_matrix.squeeze(0))
        #
        #             if vis:
        #                 elements = []
        #                 # print(rotation_matrix)
        #                 # print(np.array(predicted_rotation_matrix.cpu())[0].T)
        #                 torch_model = o3d.io.read_triangle_mesh(os.path.join(data_path, 'torch', torch_name + '.obj'))
        #                 copy_torch = copy(torch_model)
        #                 GT_torch=copy(torch_model)
        #                 elements.append(pcd)
        #                 elements.append(coor1)
        #                 tf = np.zeros((4, 4))
        #                 tf[3, 3] = 1.0
        #                 tf[0:3,0:3]=np.array(rotation_matrix.cpu())
        #                 tf[0:3,3]=translate_weld_seam_points
        #
        #                 gt_tf = np.zeros((4, 4))
        #                 gt_tf[3, 3] = 1.0
        #                 gt_tf[0:3,0:3]=np.array(rotation_matrix.cpu())
        #                 gt_tf[0:3,3]=translate_weld_seam_points
        #                 GT_torch.compute_vertex_normals()
        #                 GT_torch.paint_uniform_color([0, 0, 1])
        #                 GT_torch.transform(gt_tf)
        #
        #
        #                 # print('tf',tf)
        #                 copy_torch.compute_vertex_normals()
        #                 copy_torch.paint_uniform_color([0,1,0])
        #                 copy_torch.transform(tf)
        #                 # elements.append(GT_torch)
        #                 elements.append(copy_torch)
        #                 elements.append(weld_seam)
        #                 o3d.visualization.draw_geometries(elements)
        #             # print(predicted_rotation_matrix)
        #     # print(elements)
        #     # if vis:
        #
        # true_matrices = torch.stack(true_matrices)
        # predicted_matrices = torch.stack(predicted_matrices)
        # mse = torch.mean((true_matrices - predicted_matrices) ** 2)
        # print(f"Mean Squared Error: {mse}")
        # for weld_info in weld_infos:
        #     # print(weld_info)
        #     # pcd = o3d.io.read_point_cloud(str('../data/Reisch/'+weld_info[0]+'.pcd'))
        #     name=str(weld_info[0])
        #     pcd = o3d.io.read_point_cloud(os.path.join(wz_path,name+'.pcd'))
        #     point_cloud = torch.tensor(pcd.points, dtype=torch.float32).cuda()
        #     pose_position=torch.tensor(weld_info[4:7].astype(float),dtype=torch.float32).cuda()
        #     rotation_matrix=np.zeros((3,3))
        #     rotation_matrix[0, 0:3]=weld_info[17:20].astype(float)
        #     rotation_matrix[1, 0:3] = weld_info[20:23].astype(float)
        #     rotation_matrix[2, 0:3] = weld_info[23:26].astype(float)
        #     # print(rotation_matrix)
        #     rotation_matrix=torch.tensor(rotation_matrix,dtype=torch.float32).cuda()
        #     # print(point_cloud,weld_position,rotation_matrix)
        #     # print(weld_info)
        # # point_cloud, weld_position, rotation_matrix = dataset[0]
        #
        #     predicted_rotation_matrix = model(point_cloud.unsqueeze(0), pose_position.unsqueeze(0), welding_gun_pcd.unsqueeze(0))
        #     predict_rot_dict[name]=predicted_rotation_matrix
        #     true_matrices.append(rotation_matrix)
        #     predicted_matrices.append(predicted_rotation_matrix.squeeze(0))
        #     print(predicted_rotation_matrix)
        # true_matrices = torch.stack(true_matrices)
        # predicted_matrices = torch.stack(predicted_matrices)
        # mse = torch.mean((true_matrices - predicted_matrices) ** 2)
        # print(f"Mean Squared Error: {mse}")
    # return predict_rot_dict


# if __name__ == "__main__":
    # main()sss
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

# model = PointCloudNet().cuda()
# loss_function = nn.MSELoss()
#
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#

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

#         if i % 1000 == 0:  # 每10个批次打印一次
#             print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataset)}], Loss: {loss.item()}")
#             torch.save(model.state_dict(), 'model.pth')
#
# test_dataset = PointCloudDataset(data, welding_gun_pcd)
#
# model.eval()
#

# true_matrices = []
# predicted_matrices = []
#
# with torch.no_grad():
#     for point_cloud, weld_position, true_rotation_matrix in dataset:

#         predicted_matrix = model(point_cloud.unsqueeze(0).cuda(), weld_position.unsqueeze(0).cuda(), welding_gun_pcd.unsqueeze(0).cuda())
#

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

#     return np.dot(point_cloud, rotation_matrix.T)
#
# def visualize_point_clouds(pcd1, pcd2):

#     pcd1.paint_uniform_color([1, 0, 0])
#     pcd2.paint_uniform_color([0, 1, 0])
#     o3d.visualization.draw_geometries([pcd1, pcd2])
#
# # workpiece_pcd_np:
# # welding_gun_pcd_np:
# # rotation_matrix_np:
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


# workpiece_pcd = o3d.geometry.PointCloud()
# workpiece_pcd.points = o3d.utility.Vector3dVector(workpiece_pcd_np)
#
# welding_gun_pcd = o3d.geometry.PointCloud()
# welding_gun_pcd.points = o3d.utility.Vector3dVector(welding_gun_pcd_np)

# welding_gun_pcd_np -= weld_position
#

# rotated_welding_gun_pcd_np = apply_rotation(welding_gun_pcd_np, rotation_matrix_np)
# rotated_welding_gun_pcd = o3d.geometry.PointCloud()
# rotated_welding_gun_pcd.points = o3d.utility.Vector3dVector(rotated_welding_gun_pcd_np)
#

#
# visualize_point_clouds(workpiece_pcd, welding_gun_pcd)
# visualize_point_clouds(workpiece_pcd, rotated_welding_gun_pcd)