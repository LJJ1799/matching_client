import copy

import numpy as np
# import torch
import time
import os
# from model import DCP
# from util import transform_point_cloud, npmat2euler
import argparse
# from scipy.spatial.transform import Rotation
# from data import ModelNet40
# import glob
# import h5py
import open3d as o3d
# import pandas as pd
import json
# from scipy.spatial.transform import Rotation

# def npmat2euler(mats, seq='zyx'):
#     eulers = []
#     for i in range(mats.shape[0]):
#         r = Rotation.from_matrix(mats[i])
#         eulers.append(r.as_euler(seq, degrees=True))
#     return np.asarray(eulers, dtype='float32')

# def transform_input(pointcloud):
#     """
#     random rotation and transformation the input
#     pointcloud: N*3
#     """
#
#     anglex = np.random.uniform() * np.pi / 4
#     angley = np.random.uniform() * np.pi / 4
#     anglez = np.random.uniform() * np.pi / 4
#
#     # anglex = 0.04
#     # angley = 0.04
#     # anglez = 0.04
#
#     print('angle: ', anglex, angley, anglez)
#
#     cosx = np.cos(anglex)
#     cosy = np.cos(angley)
#     cosz = np.cos(anglez)
#     sinx = np.sin(anglex)
#     siny = np.sin(angley)
#     sinz = np.sin(anglez)
#     Rx = np.array([[1, 0, 0],
#                    [0, cosx, -sinx],
#                    [0, sinx, cosx]])
#     Ry = np.array([[cosy, 0, siny],
#                    [0, 1, 0],
#                    [-siny, 0, cosy]])
#     Rz = np.array([[cosz, -sinz, 0],
#                    [sinz, cosz, 0],
#                    [0, 0, 1]])
#     R_ab = Rx.dot(Ry).dot(Rz)
#     R_ba = R_ab.T
#     translation_ab = np.array([np.random.uniform(-0.5, 0.5),
#                                np.random.uniform(-0.5, 0.5),
#                                np.random.uniform(-0.5, 0.5)])
#
#     # translation_ab = np.array([0.01,0.05,0.05])
#     # print('trans: ', translation_ab)
#
#     translation_ba = -R_ba.dot(translation_ab)
#
#     pointcloud1 = pointcloud[:, :3].T
#
#     rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
#     pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
#
#     euler_ab = np.asarray([anglez, angley, anglex])
#     euler_ba = -euler_ab[::-1]
#     rotation_ba = Rotation.from_euler('zyx', euler_ba)
#
#     pointcloud1 = np.random.permutation(pointcloud1.T)
#     pointcloud2 = np.random.permutation(pointcloud2.T)
#
#     return pointcloud1.astype('float32'), pointcloud2.astype('float32'), \
#         rotation_ab, translation_ab, rotation_ba, translation_ba


# def run_one_pointcloud(src, target, net):
#     if len(src.shape) == 2 and len(target.shape) == 2:  ##  (N,3)
#
#         # print("src/target shape:", src.shape, target.shape)
#
#         src = np.expand_dims(src[:, :3], axis=0)
#         src = np.transpose(src, [0, 2, 1])  ##  (1, 3, 1024)
#         target = np.expand_dims(target[:, :3], axis=0)
#         target = np.transpose(target, [0, 2, 1])  ##  (1, 3, 1024)
#
#     net.eval()
#
#     src = torch.from_numpy(src).cuda()
#     target = torch.from_numpy(target).cuda()
#
#     rotation_ab_pred, translation_ab_pred, \
#         rotation_ba_pred, translation_ba_pred = net(src, target)
#     target_pred = transform_point_cloud(src, rotation_ab_pred,
#                                         translation_ab_pred)
#
#     src_pred = transform_point_cloud(target, rotation_ba_pred,
#                                      translation_ba_pred)
#
#     mse_s_t = torch.mean((target_pred - target) ** 2, dim=[0, 1, 2]).item()
#     mae_s_t = torch.mean(torch.abs(target_pred - target), dim=[0, 1, 2]).item()
#
#     mse_t_s = torch.mean((src_pred - src) ** 2, dim=[0, 1, 2]).item()
#     mae_t_s = torch.mean(torch.abs(src_pred - src), dim=[0, 1, 2]).item()
#     # put on cpu and turn into numpy
#     src_pred = src_pred.detach().cpu().numpy()
#     src_pred = np.transpose(src_pred[0], [1, 0])
#
#     target_pred = target_pred.detach().cpu().numpy()
#     target_pred = np.transpose(target_pred[0], [1, 0])
#
#     rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
#     translation_ab_pred = translation_ab_pred.detach().cpu().numpy()
#
#     rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
#     translation_ba_pred = translation_ba_pred.detach().cpu().numpy()
#
#     return src_pred, target_pred, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred,mse_s_t,mae_s_t,mse_t_s,mae_t_s


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
    parser.add_argument('--testall', action='store_true', default=False,
                        help='matching all slices')
    parser.add_argument('--data',nargs='*', type=str, help='two slice for matching')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--save', type=bool, default=False, metavar='N',
                        help='save the matching image')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/dcp_v1/models/model.best.t7',
                        metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)

    # net prepared
    # net = DCP(args).cuda()
    # net.load_state_dict(torch.load(args.model_path), strict=False)
    file_path='welding_zone'
    # file_path='../welding_zone'
    if args.testall:
##################This part is for matching all the point cloud one to one and save the dictionary to folder metrics#########################
        # files_origin=os.listdir(file_path)
        # finished=os.listdir('metrics_finished')
        # for i in range(len(finished)):
        #     finished[i]=finished[i].split('.')[0]
        # files=[]
        # for i in range(len(files_origin)):
        #     files_origin[i]=files_origin[i].split('.')[0]
        # for i in files_origin:
        #     if i not in finished:
        #         files.append(i)
        # for i in range(len(files)):
        #     files[i]=files[i]+'.pcd'
        # print(len(files))
        # origin_metrics = {}
        files=os.listdir(file_path)
        for i in range(0,len(files)):
            pcd1 = o3d.io.read_point_cloud(file_path + '/' + files[i])
            point1 = np.array(pcd1.points).astype('float32')
            centroid1 = np.mean(point1, axis=0)
            piont1 = point1 - centroid1
            m1 = np.max(np.sqrt(np.sum(point1 ** 2, axis=1)))
            point1 = point1 / m1
            # path1='result_img/'+str(files[i].split('.')[0])
            # os.mkdir(path1)
            dict1={}
            dict2={}
            dict3={}
            sub_dict={}
            for j in range(0,len(files)):
                pcd2 = o3d.io.read_point_cloud(file_path+'/'+files[j])
                point2 = np.array(pcd2.points).astype('float32')
                centroid2 = np.mean(point2, axis=0)
                piont2 = point2 - centroid2
                m2 = np.max(np.sqrt(np.sum(point2 ** 2, axis=1)))
                point2 = point2 / m2

                src, target = point1, point2

                ## run
                # src_pred, target_pred, r_ab, t_ab, r_ba, t_ba, mse_s_t,mae_s_t,mse_t_s,mae_t_s= run_one_pointcloud(src, target, net)
                # anglex = np.random.uniform() * np.pi / 4
                # angley = np.random.uniform() * np.pi / 4
                # anglez = np.random.uniform() * np.pi / 4
                # euler_ab = np.asarray([anglez, angley, anglex])
                # euler_ab = npmat2euler(np.array([np.eye(3)]))
                # r_ab_euler=npmat2euler(r_ab)
                # r_mse_ab=np.mean((r_ab_euler - np.degrees(euler_ab)) ** 2)
                # if mse_s_t>0.16 and mse_s_t <0.22:
                #     continue
                # if r_mse_ab >50 and r_mse_ab <200:
                #     continue
                # sub_dict['mse']=mse_s_t
                # sub_dict['r_mse']=r_mse_ab
                # dict1[files[j].split('.')[0]]=sub_dict
                # print("#############  src -> target :\n", r_ab, t_ab,'mse:',mse_s_t,'mae:',mae_s_t)
                # print("#############  src <- target :\n", r_ba, t_ba,'mse:',mse_t_s,'mae:',mae_t_s)
                # np->open3d

                # src_cloud = o3d.geometry.PointCloud()
                # src_cloud.points = o3d.utility.Vector3dVector(point1*m1+centroid1)
                # tgt_cloud = o3d.geometry.PointCloud()
                # tgt_cloud.points = o3d.utility.Vector3dVector(point2*m2+centroid2)
                # trans_cloud = o3d.geometry.PointCloud()
                # trans_cloud.points = o3d.utility.Vector3dVector(target_pred*m1+centroid1)
                src_cloud = o3d.geometry.PointCloud()
                src_cloud.points = o3d.utility.Vector3dVector(src)
                src_cloud_copy = copy.copy(src_cloud)
                tgt_cloud = o3d.geometry.PointCloud()
                tgt_cloud.points = o3d.utility.Vector3dVector(target)
                icp_s_t = o3d.pipelines.registration.registration_icp(source=src_cloud, target=tgt_cloud,
                                                                  max_correspondence_distance=0.2,
                                                                  estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
                icp_t_s = o3d.pipelines.registration.registration_icp(source=tgt_cloud, target=src_cloud,
                                                                  max_correspondence_distance=0.2,
                                                                  estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
                result_s_t = src_cloud_copy.transform(icp_s_t.transformation)

                mean_distance_s_t = np.mean(src_cloud.compute_point_cloud_distance(tgt_cloud))
                fitness_s_t=icp_s_t.fitness
                rmse_s_t=icp_s_t.inlier_rmse
                correspondence_s_t=len(np.asarray(icp_s_t.correspondence_set))

                mean_distance_t_s=np.mean(tgt_cloud.compute_point_cloud_distance(src_cloud))
                fitness_t_s=icp_t_s.fitness
                rmse_t_s=icp_t_s.inlier_rmse
                correspondence_t_s=len(np.asarray(icp_t_s.correspondence_set))

                if mean_distance_s_t>0.03 or rmse_s_t>0.03 or correspondence_s_t<2000 or mean_distance_t_s>0.03 or rmse_t_s>0.03 or correspondence_t_s<2000:
                    continue
                print(file_path + '/' + files[i])
                print(file_path + '/' + files[j])
                print('fitness', icp_s_t.fitness)
                print('rmse', icp_s_t.inlier_rmse)
                print('correspondences', correspondence_s_t)
                print('mean_distance', mean_distance_s_t)
                # view
                src_cloud.paint_uniform_color([1, 0, 0])
                tgt_cloud.paint_uniform_color([0, 1, 0])
                result_s_t.paint_uniform_color([0, 0, 1])
                dict1[files[j].split('.')[0]] = {'rmse':icp_s_t.inlier_rmse,'distance':mean_distance_s_t}
                # sub_dict['rmse'] = icp.inlier_rmse
                # sub_dict['distance'] = distance

                # o3d.visualization.draw_geometries([src_cloud, tgt_cloud, result], width=800)
                # all_point=src_cloud+tgt_cloud+trans_cloud

                # vis=o3d.visualization.Visualizer()
                # vis.create_window()
                # vis.add_geometry(all_point)
                # vis.update_geometry(all_point)
                # vis.poll_events()
                # vis.update_renderer()
                # save_name=str(mse_s_t)+'_'+str(r_mse_ab)+'_'+files[i].split('.')[0]+'_'+files[j].split('.')[0]+'_'
                # save_path=os.path.join(path1,save_name+'.png')
                # vis.capture_screen_image(save_path)
                # vis.destroy_window()
                # time.sleep(0.2)
                # print('dict1',dict1)
            dict2=sorted(dict1.items(),key=lambda dict1:dict1[1]['rmse'])
            dict3=dict(dict1)
            # print('dict3',dict3)
            dict_name=files[i].split('.')[0]+'.json'
            with open('metric/'+dict_name,'w') as f:
                f.write(json.dumps(dict3,indent=1))
            # tf=open('metric/'+dict_name,'w')
            # json.dump(str(dict3),tf,indent=0)
        end = time.time()
        print('time:',end-start)
    '''End of matching one to one'''
    if not args.testall:
###################This part is for matching two special point cloud, then visualize the result#########################
        # f=h5py.File('data/modelnet40_ply_hdf5_2048/ply_data_train2.h5','r')
        # data = f['data'][:].astype('float32')
        # f.close()
        # point1 = data[0,:,:]
        # print(len(point1))
        file1=file_path+'/'+args.data[0]+'.pcd'
        file2=file_path+'/'+args.data[1]+'.pcd'
        pcd1=o3d.io.read_point_cloud(file1)
        point1=np.array(pcd1.points).astype('float32')
        centroid1=np.mean(point1,axis=0)
        piont1=point1-centroid1
        m1 = np.max(np.sqrt(np.sum(point1 ** 2, axis=1)))
        point1=point1/m1
        pcd2=o3d.io.read_point_cloud(file2)
        point2=np.array(pcd2.points).astype('float32')
        centroid2 = np.mean(point2,axis=0)
        piont2 = point2-centroid2
        m2 = np.max(np.sqrt(np.sum(point2 ** 2, axis=1)))
        point2 = point2 / m2
        # _,point2,_,_,_,_ = transform_input(point1)
        src, target = point1, point2


        ## run
        # src_pred, target_pred, r_ab, t_ab, r_ba, t_ba, mse_s_t,mae_s_t,mse_t_s,mae_t_s= run_one_pointcloud(src, target, net)
        # print('mse',mse_s_t)
        # print('r_ab',r_ab)
        # # anglex = np.random.uniform() * np.pi / 4
        # # angley = np.random.uniform() * np.pi / 4
        # # anglez = np.random.uniform() * np.pi / 4
        # # euler_ab = np.asarray([anglez, angley, anglex])
        # print(np.eye(3))
        # euler_ab=npmat2euler(np.array([np.eye(3)]))
        # print('euler_ab',euler_ab)
        # r_ab_euler = npmat2euler(r_ab)
        # print('r_ab_euler',r_ab_euler)
        # r_mse_ab = np.mean((r_ab_euler - np.degrees(euler_ab)) ** 2)
        # print('r_mse_ab',r_mse_ab)
        # print('total_error=error+rotation_norm+translation_norm',)
        # print('total_error=',mse_s_t,"+",norm_r,"+",norm_t,"=",mse_s_t+norm_r+norm_t)
        # np->open3d
        src_cloud = o3d.geometry.PointCloud()
        src_cloud.points = o3d.utility.Vector3dVector(src)
        src_cloud_copy=copy.copy(src_cloud)
        tgt_cloud = o3d.geometry.PointCloud()
        tgt_cloud.points = o3d.utility.Vector3dVector(target)
        icp_s_t = o3d.pipelines.registration.registration_icp(source=src_cloud, target=tgt_cloud,
                                                              max_correspondence_distance=0.2,
                                                              estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        icp_t_s = o3d.pipelines.registration.registration_icp(source=tgt_cloud, target=src_cloud,
                                                              max_correspondence_distance=0.2,
                                                              estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
        result_s_t = src_cloud_copy.transform(icp_s_t.transformation)

        mean_distance_s_t = np.mean(src_cloud.compute_point_cloud_distance(tgt_cloud))
        fitness_s_t = icp_s_t.fitness
        rmse_s_t = icp_s_t.inlier_rmse
        correspondence_s_t = len(np.asarray(icp_s_t.correspondence_set))

        mean_distance_t_s = np.mean(tgt_cloud.compute_point_cloud_distance(src_cloud))
        fitness_t_s = icp_t_s.fitness
        rmse_t_s = icp_t_s.inlier_rmse
        correspondence_t_s = len(np.asarray(icp_t_s.correspondence_set))
        print('fitness', fitness_s_t)
        print('rmse', rmse_s_t)
        print('correspondences', correspondence_s_t)
        print('mean_distance', mean_distance_s_t)

        print('fitness2', fitness_t_s)
        print('rmse2', rmse_t_s)
        print('correspondences2', correspondence_t_s)
        print('mean_distance2', mean_distance_t_s)
        # print('mahalanobis_distance_src',mahalanobis_distance_src)
        # print('mahalanobis_distance_tgt',mahalanobis_distance_tgt)
        # print('diff',mahalanobis_distance_src-mahalanobis_distance_tgt)
        # view
        src_cloud.paint_uniform_color([1, 0, 0])
        tgt_cloud.paint_uniform_color([0, 1, 0])
        result_s_t.paint_uniform_color([0, 0, 1])
        # all_cloud=src_cloud+tgt_cloud+trans_cloud
        # vis=o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(all_cloud)
        # vis.update_geometry(all_cloud)
        # vis.poll_events()
        # vis.update_renderer()
        # save_name=str(mse_s_t)+'_'+str(r_mse_ab)+'_'+args.data[0]+'_'+args.data[1]
        # save_path=os.path.join(save_name+'.png')
        o3d.visualization.draw_geometries([src_cloud,tgt_cloud,result_s_t], width=800)
        # vis.capture_screen_image(save_path)
        # vis.destroy_window()