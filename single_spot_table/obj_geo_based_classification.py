import os
from re import L
import sys
CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH)
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, './utils'))
import numpy as np
import trimesh
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import calinski_harabasz_score
import pickle
import matplotlib.pyplot as plt
from single_spot_table.compatibility import listdir
import time
from multiprocessing import Pool, cpu_count
import argparse
import pandas as pd



from math_util import get_projections_of_triangle, get_angle
from foundation import show_obj


class PFE():
    '''Part Feature Extractor

    Splitting assemblies, extracting hybrid feature descriptors for clustering, and generating labels

    Attributes:
        path_models: Path to single model folder containing dwg, obj, mtl and xml files
        path_split: Path to the folder which stores the mesh model of the disassembled parts
        path_label_temp: Path to the intermediate folder for storing labels
        path_classes: Path to the final classification results

    '''

    def __init__(self,
                 path_models: str,
                 path_split: str,
                 path_label_temp: str,
                 path_classes: str,
                 parallel: bool = True,
                 n_clusters=None):
        self.path_models = path_models
        self.path_split = path_split
        self.path_label_temp = path_label_temp
        self.path_classes = path_classes
        self.parallel = parallel
        self.n_clusters = n_clusters

    def split_parallel(self, components):
        for comp in components:
            path_to_comp = os.path.join(self.path_models, comp)
            files = listdir(path_to_comp)
            for file in files:
                if os.path.splitext(file)[1] == '.obj':
                    self.split(os.path.join(path_to_comp, file))

    def split(self, path_file: str):
        '''Take the assembly apart
        Create a folder for each mesh file, which stores the mesh model of the disassembled parts
        Args:
            path_file (str): path to single mesh in obj format
        '''
        i = 0
        idx_start = 0
        with open(path_file, 'r') as f:
            count = 0
            # get the name of current component mesh
            namestr = os.path.splitext(os.path.split(path_file)[-1])[0]
            # make directory for disassembled parts
            os.makedirs(os.path.join(ROOT, self.path_split, namestr))
            # prefix of single disassembled part
            outstr = os.path.join(ROOT, self.path_split, namestr) + '/' + namestr + '_'
            for line in f.readlines():
                if line[0:1] == 'o':
                    idx_start = count
                    path_part = outstr + str(i) + '.obj'
                    i += 1
                    # create new obj file
                    g = open(path_part, 'w')
                    g.write('mtllib ' + namestr + '.mtl' + '\n')
                    g.write(line)
                elif line[0:1] == 'v':
                    if i > 0:
                        # add vertices
                        g = open(outstr + str(i - 1) + '.obj', 'a')
                        g.write(line)
                    count += 1
                elif line[0:1] == 'f':
                    new_line = 'f '
                    new_line += str(int(line.split()[1]) - idx_start) + ' '
                    new_line += str(int(line.split()[2]) - idx_start) + ' '
                    new_line += str(int(line.split()[3]) - idx_start) + '\n'
                    if i > 0:
                        # define the connection of vertices using the order of the new file
                        g = open(outstr + str(i - 1) + '.obj', 'a')
                        g.write(new_line)
                else:
                    if i > 0:
                        g = open(outstr + str(i - 1) + '.obj', 'a')
                        g.write(line)
            # copy the mtl file to the output folder
            os.system('cp %s %s' % (os.path.join(ROOT, self.path_models, namestr, namestr + '.mtl'),
                                    os.path.join(ROOT, self.path_split, namestr)))

    def extract_facet_normals(self, mesh: trimesh.Trimesh):
        '''Histogram of normal distribution
        This histogram uses only one normal from coplanar meshes,
        the voting interval is 30 degrees
        Args:
            mesh (trimesh.Trimesh): single mesh model
        Returns:
            x_bins (np.array((6,))): Votes for the interval of 30 degrees between the normals and x-axis angles
            y_bins (np.array((6,))): Votes for the interval of 30 degrees between the normals and y-axis angles
            z_bins (np.array((6,))): Votes for the interval of 30 degrees between the normals and z-axis angles
        '''
        x_bins = np.zeros((6,))
        y_bins = np.zeros((6,))
        z_bins = np.zeros((6,))
        for facet in mesh.facets:
            face = facet[0]
            face_normal = mesh.face_normals[face]
            _, angle_x = get_angle(face_normal, [1, 0, 0])
            _, angle_y = get_angle(face_normal, [0, 1, 0])
            _, angle_z = get_angle(face_normal, [0, 0, 1])
            if angle_x == 180:
                angle_x = 179
            if angle_y == 180:
                angle_y = 179
            if angle_z == 180:
                angle_z = 179
            x_bins[int(angle_x // 30)] += 1
            y_bins[int(angle_y // 30)] += 1
            z_bins[int(angle_z // 30)] += 1
        return x_bins, y_bins, z_bins

    def label_special_cases(features, files):
        '''
        if there is obvious part in the plates, then label it first manually
        usually can be ignored
        Args:
            features (np.ndarray): features set
            files (list): all the disassembled parts
        Returns:
            features_del (np.ndarray): features of special cases are subtracted
            files (list): files of special cases are subtracted
            labels_dict (dict): set the value as -1 for the key of the name of special cases mesh
        '''
        index = np.where((features[:, 4] < 0.6) & (features[:, 0] > 8e6))[0]
        labels_dict = {}
        list = []
        for i in range(len(index)):
            # print(files[index[i]])
            list.append(files[index[i]])
            labels_dict[files[index[i]].strip()] = -1
        features_del = np.delete(features, index, axis=0)
        os.system('mkdir -p ../data/train/label_temp_folder/special_cases')
        for case in list:
            files.remove(case)
            case = case.rstrip()
            os.system('cp %s %s' % (case, '../data/train/label_temp_folder/special_cases'))
        # print (features_del.shape)
        # print (labels_dict)
        return features_del, files, labels_dict

    def extract_feature_from_mesh(self, path_to_mesh: str):
        '''Extract feature from a mesh

        Args:
            path_to_mesh (str): The path of the mesh to extract
        Returns:
            feature(np.ndarray): A feature descriptor
        '''
        feature = np.zeros(shape=(26,), dtype=np.float64)
        mesh = trimesh.load_mesh(path_to_mesh, validate=True)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        # move the coordinate system to the inertia principal axis
        mesh.apply_transform(mesh.principal_inertia_transform)

        # 0: area
        feature[0] = mesh.area
        # 1: volume
        feature[1] = mesh.volume
        # 2: euler_number
        feature[2] = mesh.euler_number
        # 3: side length of a cube ratio, 1.0 for cube
        feature[3] = mesh.identifier[2]
        # 4: compare the volume of the mesh with the volume of its convex hull
        feature[4] = np.divide(mesh.volume, mesh.convex_hull.volume)
        # 5. the sum of the projections of each mesh on the three coordinate planes
        xoy = 0.0
        yoz = 0.0
        xoz = 0.0
        for i in range(len(mesh.faces)):
            p1 = mesh.vertices[mesh.faces[i]][0]
            p2 = mesh.vertices[mesh.faces[i]][1]
            p3 = mesh.vertices[mesh.faces[i]][2]
            proj_xoy, proj_yoz, proj_xoz = get_projections_of_triangle(p1, p2, p3)
            xoy += proj_xoy
            yoz += proj_yoz
            xoz += proj_xoz
        feature[5:8] = [xoy, yoz, xoz]

        # 6. histogram of normal distribution
        x_bins, y_bins, z_bins = self.extract_facet_normals(mesh)
        feature[8:14] = x_bins
        feature[14:20] = y_bins
        feature[20:26] = z_bins

        return feature

    def write_all_parts(self):
        '''Get the filename of all the single plate
        Eliminate non-shaped meshes

        Args:
            None
        Returns:
            None
        '''
        all_components = listdir(self.path_split)
        with open(os.path.join(self.path_split, 'all_parts.txt'), 'w') as f:
            for component in all_components:
                files = listdir(os.path.join(self.path_split, component))
                for file in files:
                    if os.path.splitext(file)[1] == '.obj':
                        content = open(os.path.join(self.path_split, component, file), 'r')
                        lines = content.readlines()
                        # ignore some noise mesh faces
                        if len(lines) > 10:
                            f.writelines(os.path.join(self.path_split, component, file) + '\n')

    def extract_features_from_mesh_parallel(self, files, free_cores=2):
        nr_processes = max(min(len(files), cpu_count() - free_cores), 1)
        k, m = divmod(len(files), nr_processes)  # divide among processors
        split_files = list(files[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
        print(f'Extracting features ... {nr_processes} workers ... {len(files)} files')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        with Pool(nr_processes) as p:
            q = p.map(self._extract_features_from_mesh_worker, split_files)
        features = np.concatenate(q)
        print('Extraction finished')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        return features

    def _extract_features_from_mesh_worker(self, files):
        features = []
        for file in files:
            feature = self.extract_feature_from_mesh(file.strip())
            features.append(feature)
        features = np.asarray(features)
        return features

    def label(self, feats=None, parallel=True):
        '''Automatic labeling
        Cluster analysis, each cluster is a class
        Args:
            None
        Returns:
            None
        '''
        if feats is None:
            features = []
            labels_dict = {}
            # load all disassembled parts
            files = open(os.path.join(ROOT, self.path_split, 'all_parts.txt'), 'r').readlines()

            if not self.parallel:
                # extract features
                for file in files:
                    feature = self.extract_feature_from_mesh(file.strip())
                    features.append(feature)
                features = np.asarray(features)
            else:
                features = self.extract_features_from_mesh_parallel(files, 2)

            # save features
            np.save(os.path.join(ROOT, self.path_split, 'features.npy'), features)
        else:
            features = feats
        # features = np.load(ROOT+'/data/train/split/features.npy')
        # label special cases, usually not used
        # features, files, labels_dict = label_special_cases(features, files)

        # reduce the dimension of the normal features
        dim_norm = 3
        norm_info = features[:, 5:26]
        transformer = PCA(n_components=dim_norm)
        norm_transformed = transformer.fit_transform(norm_info)

        new_features = np.zeros(shape=(features.shape[0], 5 + dim_norm))
        new_features[:, 0:5] = features[:, 0:5]
        new_features[:, 5:(5 + dim_norm)] = norm_transformed
        # normalization
        minMax = MinMaxScaler((0, 1))
        features_norma = minMax.fit_transform(new_features)

        # select the suitable parameters
        gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        gamma_finl = 0
        n_finl = 0
        score_finl = 0
        n_clusters_range = range(5, 14) if self.n_clusters is None else [self.n_clusters]
        for n in n_clusters_range:
            print('\n\nFor nr_clusters = ', n, '\n')
            if not parallel:
                for _, gamma in enumerate(gammas):
                    y_pred = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(features_norma)
                    score = calinski_harabasz_score(features_norma, y_pred)
                    if score > score_finl:
                        score_finl = score
                        gamma_finl = gamma
                        n_finl = n
                        y_pred_finl = y_pred
                        # ds_finl = ds
                        # dn_finl = dn
                    # print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", n, "score:", score)
            else:
                gammas = np.linspace(0, 1, max(11, cpu_count()))
                score, gamma, y_pred = self.parallel_spectral_clustering(features_norma, n, gammas)
                if score > score_finl:
                    score_finl = score
                    gamma_finl = gamma
                    n_finl = n
                    y_pred_finl = y_pred
        print('best score: ', score_finl, 'best gamma: ', gamma_finl, 'best n_clusters: ', n_finl)
        # print('ds: ', ds, 'dn: ', dn)
        # =========================================================================
        if self.n_clusters is None:
            print('Next you will see the classification results, the category names on the window are from 0 to n.')
            print('If you want to adjust them later, please remember the category corresponding to each visualization.')
            input("(Press Enter)")
        n_clusters = n_finl
        gamma = gamma_finl
        # clusterer = SpectralClustering(n_clusters=n_clusters, gamma=gamma)
        # y_pred = clusterer.fit_predict(features_norma)
        y_pred = y_pred_finl
        # print("Calinski-Harabasz Score", calinski_harabasz_score(features_norma, y_pred))
        # print(y_pred.shape)
        for i in range(n_clusters):
            # create folder for each cluster
            os.system('mkdir -p %s' % self.path_label_temp + '/' + str(i))
            idx = np.where(y_pred == i)
            for j in range(len(idx[0])):
                labels_dict[files[idx[0][j]].strip()] = i
                os.system('cp %s %s' % (files[idx[0][j]].strip(), self.path_label_temp + '/' + str(i)))

        # save the automatically generated labels
        # print (labels_dict)
        with open(self.path_label_temp + '/labels_dict.pkl', 'wb') as tf:
            pickle.dump(labels_dict, tf, protocol=2)

        # display
        for i in range(n_clusters):
            show_obj(self.path_label_temp + '/' + str(i))
        # show_obj('../data/train/label_temp_folder/special_cases')

        # =========================================================================

    def _spectral_clustering(self, args):
        features_norma, n, gammas = args
        gamma_finl = 0
        score_finl = 0
        for gamma in gammas:
            y_pred = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(features_norma)
            score = calinski_harabasz_score(features_norma, y_pred)
            # print('gamma: ', gamma, ' - score: ',  score)
            if score > score_finl:
                score_finl = score
                gamma_finl = gamma
                y_pred_finl = y_pred
        return (score_finl, gamma_finl, y_pred_finl)

    def parallel_spectral_clustering(self, feature_norma, n, gammas, free_cores=2):
        nr_processes = max(min(len(gammas), cpu_count() - free_cores), 1)
        k, m = divmod(len(gammas), nr_processes)  # divide among processors
        split_gammas = list(gammas[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
        repeated_args = [[feature_norma, n]] * nr_processes
        args = [[*_args, _gammas] for _args, _gammas in zip(repeated_args, split_gammas)]

        # print (f'clustering ... {nr_processes} workers ...', gammas)
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        with Pool(nr_processes) as p:
            q = p.map(self._spectral_clustering, [_args for _args in args])

        # print('clustering finished')
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        score, gamma, y_pred = sorted(q, key=lambda x: x[0])[-1]

        return score, gamma, y_pred

    def relabel(self, n, parallelize=True):
        '''Redo the automatic labeling
        Cluster analysis, each cluster is a class
        Args:
            n(int): the number of classes you want
        Returns:
            None
        '''
        features = []
        labels_dict = {}
        # load all disassembled parts
        files = open(os.path.join(ROOT, self.path_split, 'all_parts.txt'), 'r').readlines()

        # extract features
        for file in files:
            feature = self.extract_feature_from_mesh(file.strip())
            features.append(feature)
        features = np.asarray(features)
        features = np.load(os.path.join(ROOT, self.path_split, 'features.npy'))
        # features, files, labels_dict = self.label_special_cases(features, files)

        # reduce the dimension of the normal features
        dim_norm = 3
        norm_info = features[:, 5:26]
        transformer = PCA(n_components=dim_norm)
        norm_transformed = transformer.fit_transform(norm_info)

        new_features = np.zeros(shape=(features.shape[0], 5 + dim_norm))
        new_features[:, 0:5] = features[:, 0:5]
        new_features[:, 5:(5 + dim_norm)] = norm_transformed
        # normalization
        minMax = MinMaxScaler((0, 1))
        features_norma = minMax.fit_transform(new_features)
        gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        if not parallelize:
            # select the suitable parameters
            gamma_finl = 0
            score_finl = 0
            for _, gamma in enumerate(gammas):
                y_pred = SpectralClustering(n_clusters=n, gamma=gamma).fit_predict(features_norma)
                score = calinski_harabasz_score(features_norma, y_pred)
                if score > score_finl:
                    score_finl = score
                    gamma_finl = gamma
            # print('ds: ', ds, 'dn: ', dn)
            # =========================================================================

            input("(Press Enter)")
            n_clusters = n
            gamma = gamma_finl
            score = score_finl
            clusterer = SpectralClustering(n_clusters=n_clusters, gamma=gamma)
            y_pred = clusterer.fit_predict(features_norma)
        else:
            score, gamma, y_pred = self.parallel_spectral_clustering(features_norma, n, gammas)
            n_clusters = n

        print('best score: ', score, 'best gamma: ', gamma)
        # print("Calinski-Harabasz Score", calinski_harabasz_score(features_norma, y_pred))
        # print(y_pred.shape)
        for i in range(n_clusters):
            os.system('mkdir -p %s' % self.path_label_temp + '/' + str(i))
            idx = np.where(y_pred == i)
            for j in range(len(idx[0])):
                labels_dict[files[idx[0][j]].strip()] = i
                os.system('cp %s %s' % (files[idx[0][j]].strip(), self.path_label_temp + '/' + str(i)))
        # save the automatically generated labels
        # print (labels_dict)
        with open(self.path_label_temp + '/labels_dict.pkl', 'wb') as tf:
            pickle.dump(labels_dict, tf, protocol=2)

        # display
        for i in range(n_clusters):
            show_obj(self.path_label_temp + '/' + str(i))
        # show_obj('../data/train/label_temp_folder/special_cases')


def main(pfe):
    if pfe.n_clusters is None:
        # 1. take the assembly apart
        print('Ensure that the components to be split are placed in the required directory format')
        input('(Press Enter)')
        print('Step1. Split the assembly')
    components = listdir(pfe.path_models)
    if not pfe.parallel:
        for comp in components:
            path_to_comp = os.path.join(pfe.path_models, comp)
            files = listdir(path_to_comp)
            for file in files:
                if os.path.splitext(file)[1] == '.obj':
                    pfe.split(os.path.join(path_to_comp, file))
    else:
        components = [component for component in components if
                      os.path.isdir(os.path.join(pfe.path_models, component))]  # remove non-folders
        nr_processes = max(min(len(components), cpu_count() - 2), 1)
        k, m = divmod(len(components), nr_processes)  # divide among processors
        split_components = list(components[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
        args = split_components
        print(f'splitting... {nr_processes} workers ... {len(components)} components')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        with Pool(nr_processes) as p:
            p.map(pfe.split_parallel, [_args for _args in args])

        print('splitting finished')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    pfe.write_all_parts()
    if pfe.n_clusters is None:
        input('(Press Enter)')
        # 2. label these parts
        print('Step2. Label the split parts')
        print(
            'The following parameters are automatically selected by the algorithm, please adjust later if you are not satisfied')
    pfe.label()
    # 3. relabel
    if pfe.n_clusters is None:
        print(
            'For the classification results, it is recommended that the initial classification be more detailed, and the subsequent operations can combine different categories')
        print(
            'In other words, if some similar parts are classified in different categories, then there is no need to reclassify them and they can be merged in subsequent operations.')
        input('(Press Enter)')
        satisfaction = True
        satisfaction = True if input('Are you satisfied with this classification?(y/n)\n') == 'y' else False
        while not satisfaction:
            os.system('rm -rf %s' % pfe.path_label_temp)
            n = int(input('How many categories do you want to divide into?\n'))
            pfe.relabel(n)
            satisfaction = True if input('Are you satisfied with this classification?(y/n)\n') == 'y' else False
            # 4. merge
        print('Current categories are')
        f = open(os.path.join(pfe.path_label_temp, 'labels_dict.pkl'), 'rb')
        labeldict = pickle.load(f)
        current_classes = []
        for v in labeldict.values():
            if v not in current_classes:
                current_classes.append(v)
        print(current_classes)
        is_merge = input('Do you want to merge some classes?(y/n)\n')
    else:
        f = open(os.path.join(pfe.path_label_temp, 'labels_dict.pkl'), 'rb')
        labeldict = pickle.load(f)
        is_merge = 'n'
    if is_merge == 'y':
        used = []
        new_classes = []
        end = False
        i = 0
        while not end:
            print('Which classes do you want to merge into the new class?')
            input_str = input('(i.e. [0<Space>1<Space>2<Enter>])\n')
            to_be_merged = list(map(int, input_str.split()))
            used += to_be_merged
            new_classes.append(i)
            for c in to_be_merged:
                current_classes.remove(c)
                for key in labeldict:
                    if labeldict[key] == c:
                        labeldict[key] = 100 + i
            if input('Are there any more classes you want to merge?(y/n)\n') == 'y':
                end = False
            else:
                end = True
            i += 1
    values = []
    for key in labeldict:
        if labeldict[key] not in values:
            values.append(labeldict[key])
    class_dict = {}
    for i in range(len(values)):
        class_name = 'class_' + str(i)
        for key in labeldict:
            if labeldict[key] == values[i]:
                class_dict[key] = class_name

    os.makedirs(pfe.path_classes)
    with open(os.path.join(pfe.path_classes, 'class_dict.pkl'), 'wb') as tf:
        pickle.dump(class_dict, tf, protocol=2)
    for key in class_dict:
        class_dir = os.path.join(pfe.path_classes, class_dict[key])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        os.system('cp %s %s' % (key, class_dir))

    for i in range(len(values)):
        show_obj(pfe.path_classes + '/class_' + str(i))
    print('FINISHED!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, required=False, default=None,
                        help='pre-defined the amount of clusters, skips all user inputs.')
    n_clusters = parser.parse_args().n_clusters
    pfe = PFE(path_models=os.path.join(ROOT, '../data', 'train', 'models'),
              path_split=os.path.join(ROOT, '../data', 'train', 'split'),
              path_label_temp=os.path.join(ROOT, '../data', 'train', 'label_temp_folder'),
              path_classes=os.path.join(ROOT, '../data', 'train', 'parts_classification'),
              parallel=True,
              n_clusters=n_clusters,
              )
    main(pfe)
    