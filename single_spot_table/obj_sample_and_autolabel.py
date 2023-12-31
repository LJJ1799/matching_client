#################################################################
# Merge disassembled parts with labels and sample point clouds  #
# with adjustable sampling density.                             #
#################################################################

import os
import sys
import open3d as o3d
import numpy as np
import time
import pickle

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH)  # dir /utils
ROOT = os.path.dirname(BASE)  # dir /lookup_table
sys.path.insert(0, os.path.join(ROOT, '../utils'))
from math_util import get_projections_of_triangle, get_angle
from foundation import points2pcd
from compatibility import listdir

PATH_COMP = '../data/train/models'
PATH_XYZ = '../data/train/unlabeled_pc'
PATH_PCD = '../data/train/labeled_pc'


def _sample_and_label_worker(path, path_pcd, path_xyz, label_dict, class_dict, density=40) -> np.ndarray:
    '''Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    Args:
        path (str): path to single component
        label_dict (dict): the class name with an index
        density (int): Sampling density, the smaller the value the greater the point cloud density
    '''
    # get the current component name
    namestr = os.path.split(path)[-1]
    files = listdir(path)

    points = np.zeros(shape=(1, 4))
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            # load mesh
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path, file))
                label = label_dict[class_dict[key]]
                # get number of points according to surface area
                number_points = int(mesh.get_surface_area() / density)
                # poisson disk sampling
                pc = mesh.sample_points_poisson_disk(number_points, init_factor=5)
                xyz = np.asarray(pc.points)
                l = label * np.ones(xyz.shape[0])
                xyzl = np.c_[xyz, l]
                # print (file, 'sampled point cloud: ', xyzl.shape)
                allpoints = np.concatenate((allpoints, xyzl), axis=0)

    return points[1:]


def sample_and_label_parallelized_old(path, path_pcd, path_xyz, label_dict, class_dict, density=40,
                                      leave_free_processors=2):
    '''
    Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    Args:
        path (str): path to single component
        label_dict (dict): the class name with an index
        density (int): Sampling density, the smaller the value the greater the point cloud density
    '''
    from multiprocessing import Pool, cpu_count

    # get the current component name
    namestr = os.path.split(path)[-1]
    files = listdir(path)
    # label_list = {}
    label_count = 0

    allpoints = np.zeros(shape=(1, 4))

    nr_processes = cpu_count() - leave_free_processors
    k, m = divmod(len(files), nr_processes)
    split_files = list(files[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
    repeated_args = [[path, class_dict, label_dict, density]] * nr_processes
    args = [[*_args, _files] for _args, _files in zip(repeated_args, split_files)]

    with Pool(nr_processes) as p:
        q = p.map(_sample_and_label_worker, [_args for _args in args])

    allpoints = np.concatenate(q, axis=0)
    points2pcd(os.path.join(path_pcd, namestr + '.pcd'), allpoints[1:])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(allpoints[1:, 0:3])
    o3d.io.write_point_cloud(os.path.join(path_xyz, namestr + '.xyz'), pc)


def sample_and_label_parallel(args):
    folders, path_pcd, path_xyz, class_dict, label_dict, density, path_split, sample_alternative = args
    for folder in folders:
        try:
            if sample_alternative:
                sample_and_label_alternative(os.path.join(path_split, folder), path_pcd, path_xyz, label_dict,
                                             class_dict, density)
            else:
                sample_and_label(os.path.join(path_split, folder), path_pcd, path_xyz, label_dict, class_dict, density)
        except Exception as e:
            with open('failed_samples.txt', 'a') as f:
                f.write(str(e))
                f.write('\n')
    print('sampling done ... ...', folders)


def sample_and_label(path, path_pcd, path_xyz, label_dict, class_dict, density=40):
    '''Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    Args:
        path (str): path to single component
        label_dict (dict): the class name with an index
        density (int): Sampling density, the smaller the value the greater the point cloud density
    '''
    # get the current component name
    namestr = os.path.split(path)[-1]
    files = listdir(path)
    # label_list = {}
    label_count = 0

    allpoints = np.zeros(shape=(1, 4))
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            # load mesh
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path, file))
                label = label_dict[class_dict[key]]
                # get number of points according to surface area
                number_points = int(mesh.get_surface_area() / density)
                if number_points <= 0:
                    number_points = 1000
                    f = open('objects_with_0_points.txt', 'a')
                    f.write(file)
                    f.write('\n')
                    f.close()
                # poisson disk sampling
                pc = mesh.sample_points_poisson_disk(number_points, init_factor=5)
                xyz = np.asarray(pc.points)
                l = label * np.ones(xyz.shape[0])
                xyzl = np.c_[xyz, l]
                # print (file, 'sampled point cloud: ', xyzl.shape)
                allpoints = np.concatenate((allpoints, xyzl), axis=0)
    points2pcd(os.path.join(path_pcd, namestr + '.pcd'), allpoints[1:])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(allpoints[1:, 0:3])
    o3d.io.write_point_cloud(os.path.join(path_xyz, namestr + '.xyz'), pc)


def sample_and_label_alternative(path, path_pcd, path_xyz, label_dict, class_dict, density=40):
    '''Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    Args:
        path (str): path to single component
        label_dict (dict): the class name with an index
        density (int): Sampling density, the smaller the value the greater the point cloud density
    '''
    # get the current component name
    namestr = os.path.split(path)[-1]
    files = listdir(path)
    # label_list = {}
    label_count = 0

    allpoints = np.zeros(shape=(1, 4))
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            # load mesh
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path, file))
                label = label_dict[class_dict[key]]
                # get number of points according to surface area
                number_points = int(mesh.get_surface_area() / density)
                if number_points <= 0:
                    number_points = 1000
                    f = open('objects_with_0_points.txt', 'a')
                    f.write(file)
                    f.write('\n')
                    f.close()
                # poisson disk sampling
                if number_points > 10101:
                    pc = mesh.sample_points_uniformly(number_points)
                    # o3d.visualization.draw_geometries([pc])
                else:
                    pc = mesh.sample_points_poisson_disk(number_points)
                    # o3d.visualization.draw_geometries([pc])
                xyz = np.asarray(pc.points)
                l = label * np.ones(xyz.shape[0])
                xyzl = np.c_[xyz, l]
                # print (file, 'sampled point cloud: ', xyzl.shape)
                allpoints = np.concatenate((allpoints, xyzl), axis=0)
    points2pcd(os.path.join(path_pcd, namestr + '.pcd'), allpoints[1:])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(allpoints[1:, 0:3])
    o3d.io.write_point_cloud(os.path.join(path_xyz, namestr + '.xyz'), pc)


def alternative_sampling_method(mesh, number_points):
    hull_mesh, hull_inds = mesh.compute_convex_hull()
    start = time.time()
    pc_hull = hull_mesh.sample_points_uniformly(number_points)
    print('pc hull : ', time.time() - start)
    start = time.time()
    pc_all = mesh.sample_points_uniformly(number_points)
    print('pc full : ', time.time() - start)

    o3d.visualization.draw_geometries([pc_all])
    o3d.visualization.draw_geometries([pc_hull])
    o3d.visualization.draw_geometries([pc_hull + pc_all])

    return pc_hull + pc_all


if __name__ == '__main__':
    # load the parts and corresponding labels from part feature extractor
    f = open('data/train/parts_classification/class_dict.pkl', 'rb')
    classdict = pickle.load(f)
    # a dict that stores current labels
    label_dict = {}
    i = 0
    for v in classdict.values():
        if v not in label_dict:
            label_dict[v] = i
            i += 1
    with open(os.path.join('data/train/parts_classification/label_dict.pkl'), 'wb') as tf:
        pickle.dump(label_dict, tf, protocol=2)

    # path to disassembled parts
    path = 'data/train/split'
    # folder to save unlabeled pc in xyz format
    path_xyz = 'data/train/unlabeled_pc'
    # folder to save labeled pc in pcd format
    path_pcd = 'data/train/labeled_pc'
    if not os.path.exists(path_xyz):
        os.makedirs(path_xyz)
    if not os.path.exists(path_pcd):
        os.makedirs(path_pcd)
    folders = listdir(path)
    count = 0
    total = len(folders)
    for folder in folders:
        # for each component merge the labeled part mesh and sample mesh into pc
        if os.path.isdir(os.path.join(path, folder)):
            count += 1
            print('sampling... ...', folder)
            print(str(count) + '/' + str(total - 2))
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            sample_and_label_alternative(os.path.join(path, folder), PATH_PCD, PATH_XYZ, label_dict, classdict)

    # =======================================================
    # for test data
    # path = '../data/test/models'
    # path_xyz = '../data/test/pc'
    # if not os.path.exists(path_xyz):
    #     os.makedirs(path_xyz)
    # folders = listdir(path)
    # count = 0
    # total = len(folders)
    # for folder in folders:
    #     count += 1
    #     print ('sampling... ...', folder)
    #     print (str(count)+'/'+str(total))
    #     print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    #     sample_test_pc(os.path.join(path, folder))
