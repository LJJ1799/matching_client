import numpy as np
import torch
import time
import os
from model import DCP
from util import transform_point_cloud, npmat2euler
import argparse
from scipy.spatial.transform import Rotation
from data import ModelNet40
import glob
import h5py
import open3d as o3d
import pandas as pd
import json
def run_one_pointcloud(src, target, net):
    if len(src.shape) == 2 and len(target.shape) == 2:  ##  (N,3)

        print("src/target shape:", src.shape, target.shape)

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







if __name__ == "__main__":
    start=time.time()
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
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/dcp_v1/models/model.best.t7',
                        metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # net prepared
    net = DCP(args).cuda()
    net.load_state_dict(torch.load(args.model_path), strict=False)

    file_list = os.listdir('metrics')
    for i in range(len(file_list)):
        file_path = 'metrics/' + file_list[i]
        source_name = file_list[i].split('.')[0]
        source_path = 'CustomData/train_data/' + source_name + '.pcd'
        path1='result_img/'+source_name
        os.mkdir(path1)
        select_data1 = []
        select_data2 = []
        # print(file_path)
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
            data = data.items()
            select_data1 = [component for component in list(data)[:3]]
            select_data2 = [component for component in list(data)[-3:]]
            select_data1.extend(select_data2)
            # print(select_data1)
            for j in range(len(select_data1)):
                target_name = select_data1[j][0]
                target_path = 'CustomData/train_data/' + target_name + '.pcd'
                pcd1 = o3d.io.read_point_cloud(source_path)
                pcd2 = o3d.io.read_point_cloud(target_path)
                point1 = np.array(pcd1.points).astype('float32') / 100
                point2 = np.array(pcd2.points).astype('float32') / 100
                src, target = point1, point2
                src_pred, target_pred, r_ab, t_ab, r_ba, t_ba, mse_s_t, mae_s_t, mse_t_s, mae_t_s = run_one_pointcloud(
                    src, target, net)
                print(mse_s_t,select_data1[j][1])
                src_cloud = o3d.geometry.PointCloud()
                src_cloud.points = o3d.utility.Vector3dVector(point1)
                tgt_cloud = o3d.geometry.PointCloud()
                tgt_cloud.points = o3d.utility.Vector3dVector(point2)
                trans_cloud = o3d.geometry.PointCloud()
                trans_cloud.points = o3d.utility.Vector3dVector(src_pred)
                #view
                src_cloud.paint_uniform_color([1, 0, 0])
                tgt_cloud.paint_uniform_color([0, 1, 0])
                trans_cloud.paint_uniform_color([0, 0, 1])
                all_point = src_cloud + tgt_cloud + trans_cloud
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(all_point)
                vis.update_geometry(all_point)
                vis.poll_events()
                vis.update_renderer()
                save_name=str(mse_s_t)+source_name+'_'+target_name+'_'
                save_file=os.path.join(path1,save_name+'.png')
                vis.capture_screen_image(save_file)
                vis.destroy_window()
                time.sleep(0.2)