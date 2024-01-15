#################################################################
# Perform point cloud slicing, generate feature dictionary, make# 
# lookup table, feature dictionary de-duplication, define       #
# feature dictionary similarity                                 #
#################################################################
import open3d as o3d
import numpy as np
import os
import sys
from sklearn.cluster import DBSCAN
from utils.compatibility import listdir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, '../utils'))

from utils.foundation import load_pcd_data, points2pcd, fps
from utils.xml_parser import parse_frame_dump, list2array
from utils.math_util import rotate_mat, rotation_matrix_from_vectors
import pickle
import copy
import time
from shutil import copyfile
from tools import get_weld_info
from numba import njit


# # load label dictionary
# f = open('../data/train/parts_classification/label_dict.pkl', 'rb')
# lable_list = pickle.load(f)
# # reverse dictionary for subsequent processing
# label_dict_r = dict([val, key] for key, val in lable_list.items())   


class WeldScene:
    '''
    Component point cloud processing, mainly for slicing

    Attributes:
        path_pc: path to labeled pc

    '''

    def __init__(self, pc_path):
        self.pc = o3d.geometry.PointCloud()
        xyzl = load_pcd_data(pc_path)
        # print (xyzl.shape)
        self.xyz = xyzl[:, 0:3]
        self.l = xyzl[:, 3]
        self.pc.points = o3d.utility.Vector3dVector(self.xyz)

    def rotation(self,axis,norm):
        rot_axis = np.cross(axis, norm) / (np.linalg.norm(axis) * np.linalg.norm(norm))
        theta = np.arccos((axis @ norm)) / (np.linalg.norm(axis) * np.linalg.norm(norm))
        rotation = rotate_mat(axis=rot_axis, radian=theta)
        return rotation

    def get_distance_and_translate(self,weld_info):
        x_center = (np.max(weld_info[:,1]) + np.min(weld_info[:,1])) / 2
        y_center = (np.max(weld_info[:,2]) + np.min(weld_info[:,2])) / 2
        z_center = (np.max(weld_info[:,3]) + np.min(weld_info[:,3])) / 2
        translate=np.array([x_center,y_center,z_center])
        x_diff=np.max(weld_info[:,1])-np.min(weld_info[:,1])
        if x_diff<2:
            x_diff=0
        y_diff = np.max(weld_info[:, 2])-np.min(weld_info[:, 2])
        if y_diff<2:
            y_diff=0
        z_diff = np.max(weld_info[:, 3])-np.min(weld_info[:, 3])
        if z_diff<2:
            z_diff=0
        distance = int(pow(pow(x_diff,2)+pow(y_diff,2)+pow(z_diff,2),0.5))+50

        return distance,translate

    def crop(self, weld_info, num_points=2048, name='10107', vis=False):
        '''Cut around welding spot

        Args:
            weld_info (np.ndarray): welding info, including torch type, weld position, surface normals, torch pose
            crop_size (int): side length of cutting bbox in mm
            num_points (int): the default point cloud contains a minimum of 2048 points, if not enough then copy and fill
            vis (Boolean): True for visualization of the slice while slicing
        Returns:
            xyzl_crop (np.ndarray): cropped pc with shape num_points*4, cols are x,y,z,label
            cropped_pc (o3d.geometry.PointCloud): cropped pc for visualization
            weld_info (np.ndarray): update the rotated component pose for torch (if there is)
        '''
        pc = copy.copy(self.pc)

        # tow surface normals at the welding spot
        distance, translate = self.get_distance_and_translate(weld_info)
        extent = 250
        crop_extent = np.array([distance, extent + 5, extent + 5])
        weld_seam_points = weld_info[:, 1:4]
        weld_seam = o3d.geometry.PointCloud()
        weld_seam.points = o3d.utility.Vector3dVector(weld_seam_points)

        # move the coordinate center to the welding spot
        pc.translate(-translate)
        weld_seam.translate(-translate)
        # rotation at this welding spot  1.
        rot = weld_info[0, 10:13] * np.pi / 180
        rotation = rotate_mat(axis=[1, 0, 0], radian=rot[0])

        tf1 = np.zeros((4, 4))
        tf1[3, 3] = 1.0
        tf1[0:3, 0:3] = rotation
        # pc.transform(tf1)
        # new normals
        norm1 = np.around(weld_info[0, 4:7], decimals=6)
        norm2 = np.around(weld_info[0, 7:10], decimals=6)

        norm1_r = np.matmul(rotation, norm1.T)
        norm2_r = np.matmul(rotation, norm2.T)
        # torch pose

        for i in range(weld_info.shape[0]):
            pose = np.zeros((3, 3))
            pose[0:3, 0] = weld_info[i,14:17].astype(float)
            pose[0:3, 1] = weld_info[i,17:20].astype(float)
            pose[0:3, 2] = weld_info[i,20:23].astype(float)
            pose_new = np.matmul(rotation, pose)
            weld_info[i, 4:7] = norm1_r
            weld_info[i, 7:10] = norm2_r
            weld_info[i, 14:17] = pose_new[0:3, 0]
            weld_info[i, 17:20] = pose_new[0:3, 1]
            weld_info[i, 20:23] = pose_new[0:3, 2]

        coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
        norm_ori = np.array([0, 0, 1])
        # bounding box of cutting area
        rotation_bbox = rotation_matrix_from_vectors(norm_ori, norm_ori)
        seams_direction=np.cross(norm1,norm2)
        if (abs(seams_direction[0])==0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)) or (abs(seams_direction[0])!=0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)):
            rotation_bbox=self.rotation(np.array([1,0,0]),seams_direction)
        center_bbox=norm2/np.linalg.norm(norm2)*extent/2+norm1/np.linalg.norm(norm1)*extent/2
        bbox = o3d.geometry.OrientedBoundingBox(center=center_bbox, R=rotation_bbox,extent=crop_extent)

        pc.paint_uniform_color([0.9,0,0])
        weld_seam.paint_uniform_color([0, 0.9, 0])
        cropped_pc = pc.crop(bbox)
        cropped_pc.paint_uniform_color([0,0,0.9])
        idx_crop = bbox.get_point_indices_within_bounding_box(pc.points)
        xyz_crop = self.xyz[idx_crop]
        xyz_crop -= translate
        xyz_crop_new = np.matmul(rotation_matrix_from_vectors(norm_ori, norm_ori), xyz_crop.T).T
        l_crop = self.l[idx_crop]
        xyzl_crop = np.c_[xyz_crop_new, l_crop]
        xyzl_crop = np.unique(xyzl_crop, axis=0)
        # if(xyzl_crop.shape[0]<500):
        #     print('name',name)
        #     print(xyzl_crop.shape)
        if xyzl_crop.shape[0]<300:
            return xyzl_crop, cropped_pc, weld_info
        if vis:
            o3d.visualization.draw_geometries([cropped_pc, coor1, bbox,weld_seam,pc])
        while xyzl_crop.shape[0] < num_points:
            xyzl_crop = np.vstack((xyzl_crop, xyzl_crop))
        xyzl_crop = fps(xyzl_crop, num_points)
        # if vis:

        return xyzl_crop, cropped_pc, weld_info


def slice_one_parallel(args):
    files, path_welding_zone, path_lookup_table, crop_size, num_points, path_pcd, path_models = args
    for file in files:
        pc_path = os.path.join(path_pcd, file + '.pcd')
        xml_path = os.path.join(path_models, file, file + '.xml')
        name = file
        try:
            slice_one(pc_path, path_welding_zone, path_lookup_table, xml_path, name, crop_size, num_points)
        except Exception as e:
            print(e)
            with open('failed_slices.txt', 'a') as f:
                f.write(str(e))
                f.write('\n')
    print('slicing done ... ...', files)


def slice_one(pc_path, path_wz, path_lookup_table, xml_path, name, crop_size=400, num_points=2048):
    '''Slicing one component

    Args:
        pc_path (str): path to a pcd format point cloud
        xml_path (path): path to the xml file corresponding to the pc
        name (str): name of the pc
    '''
    # create welding scene
    ws = WeldScene(pc_path)
    # load welding info contains position, pose, normals, torch, etc.
    frames = list2array(parse_frame_dump(xml_path))
    # a summary of the filename of all the welding slices in one component with theirs welding infomation
    d = {}
    minpoints = 1000000
    weld_infos = get_weld_info(xml_path)
    # weld_infos = np.vstack(weld_infos)
    # print('weld_infos',weld_infos)
    print('kkk',name)
    for i in range(weld_infos.shape[0]):
        print('iii',i)
        weld_info = weld_infos[i][:,3:].astype(float)
        cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=num_points,name=name)
        # draw(cxyzl[:,0:3], cxyzl[:,3])
        # save the pc slice
        if cxyzl.shape[0]<400:
            continue
        points2pcd(os.path.join(path_wz, name + '_' + str(i) + '.pcd'), cxyzl)
        d[name + '_' + str(i)] = new_weld_info
    with open(os.path.join(path_lookup_table, name + '.pkl'), 'wb') as tf:
        pickle.dump(d, tf, protocol=2)
    # print ('num of welding spots: ',len(d))


def merge_lookup_table(path_lookup_table):
    '''Merge all the lookup table of single component into one

    '''
    dict_all = {}
    files = listdir(path_lookup_table)
    for file in files:
        with open(os.path.join(path_lookup_table, file), 'rb') as f:
            fd = pickle.load(f)
            dict_all.update(fd)
    with open(os.path.join(path_lookup_table, 'lookup_table.pkl'), 'wb') as tf:
        pickle.dump(dict_all, tf, protocol=2)


def get_feature_dict(path_data, path_wz, path_lookup_table, label_dict_r):
    '''Save a feature dictionary for each slice in a .pkl file, the dict contains
    the nummber of each class and the coordinates of bounding box of each cluster of class
    '''

    if not os.path.exists(os.path.join(path_data, 'ss_lookup_table/dict')):
        os.makedirs(os.path.join(path_data, 'ss_lookup_table/dict'))
    with open(os.path.join(path_lookup_table, 'lookup_table.pkl'), 'rb') as f:
        dict_all = pickle.load(f)
    files = listdir(path_wz)

    for i, file in enumerate(files):
        # print (str(i)+'/'+str(len(files)), file)
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        if os.path.splitext(file)[1] == '.pcd':
            name = os.path.splitext(file)[0]
            print('get feature dict --> ', name)
            # load slice
            xyzl = load_pcd_data(os.path.join(path_wz, file))
            # draw(xyzl[:,0:3],xyzl[:,3])
            xyz = xyzl[:, 0:3]
            l = xyzl[:, 3]
            label = np.squeeze(l)
            feature_dict = {}
            normals = dict_all[name][4:7] + dict_all[name][7:10]
            feature_dict['normals'] = normals
            torch = dict_all[name][0]
            feature_dict['torch'] = torch
            elements = []
            min_edge = np.min(np.max(xyz, axis=0) - np.min(xyz, axis=0))

            for i in range(len(label_dict_r)):
                idx = np.argwhere(label == i)
                idx = np.squeeze(idx)
                print(label,idx)
                l_i = label[idx]
                xyz_i = xyz[idx]
                if xyz_i.shape[0] > 10:
                    eps = min_edge / 2
                    c = DBSCAN(eps=eps, min_samples=10).fit(xyz_i)
                    number = c.labels_.max() + 1
                    # print ('The number of '+label_dict_r[i]+':'+str(number))
                    # draw(xyz_i, c.labels_)
                    feature_info = np.zeros(shape=(number, 8, 3), dtype=float)
                    for _ in range(number):
                        idx_f = np.argwhere(c.labels_ == _)
                        idx_f = np.squeeze(idx_f)
                        xyz_f = xyz_i[idx_f]  # each cluster of each class
                        l_f = l_i[idx_f]
                        # print xyz_f.shape
                        geometry = o3d.geometry.PointCloud()
                        geometry.points = o3d.utility.Vector3dVector(xyz_f)
                        # geometry.paint_uniform_color([0.53, 0.81, 1.0])

                        bbox = geometry.get_axis_aligned_bounding_box()

                        feature_info[_, :, :] = np.asarray(bbox.get_box_points())
                    feature_dict[label_dict_r[i]] = feature_info
                else:
                    feature_dict[label_dict_r[i]] = None
            # o3d.visualization.draw_geometries(elements)
            with open(os.path.join(path_data, 'ss_lookup_table/dict', os.path.splitext(file)[0] + '.pkl'), 'wb') as tf:
                pickle.dump(feature_dict, tf, protocol=2)


def similarity_njit(feature_dict1, feature_dict2, label_dict_r):
    norm1 = feature_dict1['normals']
    torch1 = feature_dict1['torch']
    array_classes1 = np.asarray([feature_dict1[key] for key in label_dict_r.values()])
    norm2 = feature_dict2['normals']
    torch2 = feature_dict2['torch']
    array_classes2 = np.asarray([feature_dict2[key] for key in label_dict_r.values()])
    return _similarity_njit(norm1, torch1, array_classes1, norm2, torch2, array_classes2)


@njit  # (float64(float64[:], int64, float64[:, :, :], float64[:], int64, float64[:, :, :]))
def _similarity_njit(norm1, torch1, array_classes1, norm2, torch2, array_classes2):
    loss_norm = np.sum((norm1 - norm2) ** 2)
    loss_torch = int(torch1 == torch2)
    geo_loss = 0
    for i in range(len(array_classes1)):
        if array_classes1[i] is None:
            ac1 = np.array([])
        else:
            ac1 = array_classes1[i]
        if array_classes2[i] is None:
            ac2 = np.array([])
        else:
            ac2 = array_classes2[i]
        if ac1.shape == ac2.shape:
            geo_loss += np.sum((ac1 - ac2) ** 2) / 100000
        else:
            geo_loss += np.abs(ac1.shape[0] - ac2.shape[0])

    return 10 * loss_norm + 10 * loss_torch + geo_loss


def similarity(feature_dict1, feature_dict2, label_dict_r):
    '''Calculate the similarity error between two feature dictionaries

    Args:
        feature_dict1 (dict)
        feature_dict2 (dict)
    Returns:
        Total similarity error (float)

    '''
    loss_amount = 0
    loss_geo = 0
    loss_norm = 0
    norm1 = feature_dict1['normals']
    norm2 = feature_dict2['normals']
    loss_norm = np.sum((norm1 - norm2) ** 2)
    torch1 = feature_dict1['torch']
    torch2 = feature_dict2['torch']
    loss_torch = int(torch1 == torch2)
    # loss_torch = (torch1-torch2)**2
    for i in range(len(label_dict_r)):
        # print feature_dict1[labeldict[i]]
        # get the number of each class
        if type(feature_dict1[label_dict_r[i]]) == type(None):
            class_num_1_cur = 0
        else:
            class_num_1_cur = feature_dict1[label_dict_r[i]].shape[0]

        if type(feature_dict2[label_dict_r[i]]) == type(None):
            class_num_2_cur = 0
        else:
            class_num_2_cur = feature_dict2[label_dict_r[i]].shape[0]

        if class_num_1_cur == class_num_2_cur and class_num_1_cur != 0:
            f1 = feature_dict1[label_dict_r[i]]
            f1.sort(axis=0)
            f2 = feature_dict2[label_dict_r[i]]
            f2.sort(axis=0)
            for _ in range(class_num_1_cur):
                box1_all_points = f1[_]  # shape(8, 3)
                box2_all_points = f2[_]  # shape(8, 3)
                loss_geo += np.sum((box1_all_points - box2_all_points) ** 2)
        loss_amount += abs(class_num_1_cur - class_num_2_cur)
    return 10 * loss_norm + 10 * loss_torch + loss_amount + loss_geo / 100000


def decrease_lib_dummy(path_data, path_train, path_wz, label_dict_r=None):
    path = os.path.join(path_data, 'ss_lookup_table/dict')
    if not os.path.exists(os.path.join(path_train, 'welding_zone_comp')):
        os.makedirs(os.path.join(path_train, 'welding_zone_comp'))
    files = listdir(path)
    new_lib = files
    fileObject = open(os.path.join(path_data, 'ss_lookup_table/comp.txt'), 'w')
    for ip in new_lib:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    for file in new_lib:
        name = os.path.splitext(file)[0]
        src = os.path.join(path_wz, name + '.pcd')
        # src2 = './data/welding_zone/'+name+'.xml'
        if os.path.exists(src):
            copyfile(src, os.path.join(path_train, 'welding_zone_comp', f'{name}.pcd'))
        # os.system('cp %s %s' % (src, os.path.join(path_train, 'welding_zone_comp')))
        # os.system('cp %s ./data/welding_zone_comp' % (src2))


def decrease_lib(path_data, path_train, path_wz, label_dict_r):
    '''Removal of redundant slices

    '''
    path = os.path.join(path_data, 'ss_lookup_table/dict')
    if not os.path.exists(os.path.join(path_train, 'welding_zone_comp')):
        os.makedirs(os.path.join(path_train, 'welding_zone_comp'))
    files = listdir(path)
    used = np.zeros(len(files))
    new_lib = []
    for i in range(len(files)):
        # for i, file in enumerate(files):
        # print (i, files[i])
        if used[i] == 0:
            used[i] = 1
            new_lib.append(files[i])
            with open(os.path.join(path, files[i]), 'rb') as f:
                fd1 = pickle.load(f)
            # for j, _ in enumerate(files):
            for j in range(i + 1, len(files)):
                if used[j] == 1:
                    continue
                with open(os.path.join(path, files[j]), 'rb') as g:
                    fd2 = pickle.load(g)
                if similarity(fd1, fd2, label_dict_r) < 1e-5:
                    used[j] = 1
                    print('remove redundant slices: ', str(i) + ' and ' + str(j) + ' are repeated')

    print('remove redundant slices --> before: ', len(files))
    print('remove redundant slices --> after: ', len(new_lib))

    fileObject = open(os.path.join(path_data, 'ss_lookup_table/comp.txt'), 'w')
    for ip in new_lib:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()
    for file in new_lib:
        name = os.path.splitext(file)[0]
        src = os.path.join(path_wz, name + '.pcd')
        # src2 = './data/welding_zone/'+name+'.xml'
        copyfile(src, os.path.join(path_train, 'welding_zone_comp', f'{name}.pcd'))
        # os.system('cp %s %s' % (src, os.path.join(path_train, 'welding_zone_comp')))
        # os.system('cp %s ./data/welding_zone_comp' % (src2))


def move_files(path_data):
    '''Move the feature dicts to the compact lib

    '''

    path_dict = os.path.join(path_data, 'ss_lookup_table/dict_comp')
    if not os.path.exists(path_dict):
        os.makedirs(path_dict)
    path = os.path.join(path_data, 'train/welding_zone_comp')
    files = listdir(path)
    for file in files:
        if os.path.splitext(file)[1] == '.pcd':
            name = os.path.splitext(file)[0]
            # print (name)
            copyfile(os.path.join(path_data, 'ss_lookup_table/dict/' + name + '.pkl'),
                     os.path.join(path_data, 'ss_lookup_table/dict_comp/' + name + '.pkl'))
            # os.system('cp %s %s' % (os.path.join(path_data, 'ss_lookup_table/dict/'+name+'.pkl'),
            #                         os.path.join(path_data, 'ss_lookup_table/dict_comp')))


def norm_index(path_data):
    '''Use normal strings to create sub-tables to speed up lookup

    '''
    path = os.path.join(path_data, 'ss_lookup_table/dict_comp')
    d = {}
    files = listdir(path)
    for file in files:
        with open(os.path.join(path, file), 'rb') as f:
            fd = pickle.load(f)
        # print (type(fd['normals']))
        norm = np.round(fd['normals'], 2).astype(float)
        # print(norm)
        s = ''
        for i in norm:
            if i == 0:
                s += str(0.0)
            else:
                s += str(i)
        # print (s)
        if s not in d:
            l = []
            l.append(os.path.join(path, file))
            d[s] = l
        else:
            l = d[s]
            l.append(os.path.join(path, file))
            d[s] = l
    with open(os.path.join(path_data, 'ss_lookup_table', 'norm_fd.pkl'), 'wb') as tf:
        pickle.dump(d, tf, protocol=2)


if __name__ == '__main__':
    # path to dir of welding slices
    path_welding_zone = '../data/train/welding_zone'
    # path to lookup table
    path_lookup_table = '../data/train/lookup_table'
    if not os.path.exists(path_welding_zone):
        os.makedirs(path_welding_zone)
    if not os.path.exists(path_lookup_table):
        os.makedirs(path_lookup_table)
    files = listdir('../data/train/models')
    print('Generate one point cloud slice per welding spot')
    i = 1
    for file in files:
        print(str(i) + '/' + str(len(files)), file)
        i += 1
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        pc_path = '../data/train/labeled_pc/' + file + '.pcd'
        xml_path = '../data/train/models/' + file + '/' + file + '.xml'
        name = file
        slice_one(pc_path, xml_path, name)

    merge_lookup_table()
    print('Extract feature dictionary from point cloud slices\n')
    input('(Press Enter)')
    get_feature_dict()
    print('Removing duplicate point cloud slices\n')
    input('(Press Enter)')
    decrease_lib()
    move_files()
    print('Use the normal information to generate an index for easy searching\n')
    input('(Press Enter)')
    norm_index()
    print('FINISHED')