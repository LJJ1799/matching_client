import json
from scipy.spatial.transform import Rotation
import open3d as o3d
import numpy as np
import torch
import time
import os
from model import DCP
from util import transform_point_cloud, npmat2euler
import argparse

def run_one_pointcloud(src, target, net):
    if len(src.shape) == 2 and len(target.shape) == 2:  ##  (N,3)

        # print("src/target shape:", src.shape, target.shape)

        src = np.expand_dims(src[:, :3], axis=0)
        src = np.transpose(src, [0, 2, 1])  ##  (1, 3, 1024)
        target = np.expand_dims(target[:, :3], axis=0)
        target = np.transpose(target, [0, 2, 1])  ##  (1, 3, 1024)

    net.eval()

    src = torch.from_numpy(src).cuda()
    target = torch.from_numpy(target).cuda()

    rotation_ab_pred, translation_ab_pred, \
        rotation_ba_pred, translation_ba_pred = net(src, target)
    target_pred = transform_point_cloud(src, rotation_ab_pred,
                                        translation_ab_pred)

    src_pred = transform_point_cloud(target, rotation_ba_pred,
                                     translation_ba_pred)

    mse_s_t = torch.mean((target_pred - target) ** 2, dim=[0, 1, 2]).item()
    mae_s_t = torch.mean(torch.abs(target_pred - target), dim=[0, 1, 2]).item()

    mse_t_s = torch.mean((src_pred - src) ** 2, dim=[0, 1, 2]).item()
    mae_t_s = torch.mean(torch.abs(src_pred - src), dim=[0, 1, 2]).item()
    # put on cpu and turn into numpy
    src_pred = src_pred.detach().cpu().numpy()
    src_pred = np.transpose(src_pred[0], [1, 0])

    target_pred = target_pred.detach().cpu().numpy()
    target_pred = np.transpose(target_pred[0], [1, 0])

    rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
    translation_ab_pred = translation_ab_pred.detach().cpu().numpy()

    rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
    translation_ba_pred = translation_ba_pred.detach().cpu().numpy()

    return src_pred, target_pred, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred,mse_s_t,mae_s_t,mse_t_s,mae_t_s

def matching(xml_file):
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--save', type=bool, default=False, metavar='N',
                        help='save the matching image')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/dcp_v1/models/model.best.t7',
                        metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    net = DCP(args).cuda()
    net.load_state_dict(torch.load(args.model_path), strict=False)
    file_path='Reisch_pc_seam'
    #load point cloud 1
    slice=xml_file.split('.')[0]
    pcd1=o3d.io.read_point_cloud(file_path + '/' + str(slice) + '.pcd')
    point1=np.array(pcd1.points).astype('float32')
    centroid1=np.mean(point1,axis=0)
    m1=np.max(np.sqrt(np.sum(point1 ** 2, axis=1)))
    point1=(point1-centroid1)/m1

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    files = os.listdir(file_path)
    dict1 = {}
    src=point1
    for file in files:
        # load point cloud 2
        pcd2 = o3d.io.read_point_cloud(file_path + '/' + file)
        point2 = np.array(pcd2.points).astype('float32')
        centroid2 = np.mean(point2, axis=0)
        piont2 = point2 - centroid2
        m2 = np.max(np.sqrt(np.sum(point2 ** 2, axis=1)))
        point2 = point2 / m2
        # path2 = 'result_img/' + str(files.split('.')[0])
        target=point2

        #start mathing
        src_cloud = o3d.geometry.PointCloud()
        src_cloud.points = o3d.utility.Vector3dVector(src)
        tgt_cloud = o3d.geometry.PointCloud()
        tgt_cloud.points = o3d.utility.Vector3dVector(target)
        icp_s_t = o3d.pipelines.registration.registration_icp(source=src_cloud, target=tgt_cloud,
                                                              max_correspondence_distance=0.2,
                                                              estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        icp_t_s = o3d.pipelines.registration.registration_icp(source=tgt_cloud, target=src_cloud,
                                                              max_correspondence_distance=0.2,
                                                              estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

        mean_distance_s_t = np.mean(src_cloud.compute_point_cloud_distance(tgt_cloud))
        fitness_s_t = icp_s_t.fitness
        rmse_s_t = icp_s_t.inlier_rmse
        correspondence_s_t = len(np.asarray(icp_s_t.correspondence_set))

        mean_distance_t_s = np.mean(tgt_cloud.compute_point_cloud_distance(src_cloud))
        fitness_t_s = icp_t_s.fitness
        rmse_t_s = icp_t_s.inlier_rmse
        correspondence_t_s = len(np.asarray(icp_t_s.correspondence_set))

        if mean_distance_s_t > 0.03 or rmse_s_t > 0.03 or correspondence_s_t < 2000 or mean_distance_t_s > 0.03 or rmse_t_s > 0.03 or correspondence_t_s < 2000:
            continue
        dict1[file.split('.')[0]]=rmse_s_t
    dict2 = sorted(dict1.items(), key=lambda dict1: dict1[1])
    dict3 = dict(dict2)
    return dict3

if __name__ == "__main__":
    matching(6)