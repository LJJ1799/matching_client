import os
import sys
import pickle
import time

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH)

# Force python XML parser not faster C accelerators
# because we can't hook the C implementation
# used to hack into the XML parser library and extract line numbers :)
sys.modules['_elementtree'] = None

sys.path.insert(0, os.path.join(BASE, 'utils'))
sys.path.insert(0, os.path.join(BASE, 'single_spot_table'))
from single_spot_table.pre_defined_label import PDL
from single_spot_table.obj_sample_and_autolabel import sample_and_label_alternative, sample_and_label_parallel
import slice
from utils.compatibility import listdir
from multiprocessing import Pool, cpu_count
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_sampling', action='store_true', help='If flag is present sampling won\'t be executed')
    parser.add_argument('--skip_slicing', action='store_true', help='If flag is present slicing won\'t be executed')
    parser.add_argument('-s', '--skip_both', action='store_true',
                        help='If flag is present sampling AND slicing will be skipped')
    parser.add_argument('--fast_sampling', action='store_true',
                        help='If flag is active, meshes with high vertex density are uniformly sampled into pointclouds (fast boi)')
    parser.add_argument('--decrease_lib', action='store_true',
                        help='If active, remove redundant slices from lookup table. (SLOW FOR BIG DATA)')
    parser.add_argument('--free_cores', type=int, default=2,
                        help='Amount of NOT USED cores "used_cores = total_cores - free_cores"')
    parser.add_argument('--label', type=str, default='PDL',
                        help='Type of splitting, default "PDL". To skip splitting use "skip_split"')
    parser.add_argument('--pcl_density', type=int, default=40,
                        help='Pointcloud Density (Must be the same for training and testing)')
    parser.add_argument('--crop_size', type=int, default=400,
                        help='Cropped Slice Size (Must be the same for training and testing)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points per PCL (Must be the same for training and testing)')

    return parser.parse_args()


class LookupTable():
    '''Lookup Table
    Mesh transformation, slicing, making lookup tables

    Attributes:
        path_data: The path to the data folder, see readme for the specific directory structure
        label: Labeling methods, there are 'PDL' and 'HFD'. PDL is pre-defined lable,
               if the color (material) of the parts of assemblies in the CAD design is
               differentiated according to the different kinds of parts, use PDL. HFD
               uses hybrid feature descriptors to cluster the parts of the assemblies,
               use it by running obj_geo_based_classification.py to generate the class-
               ification first, then enter the classification directory in arg hfd_path_classes
        hfd_path_classes: Path to the classification result by HFD method. The default folder is
                          "./data/train/parts_classification"
        pcl_density: A parameter that controls the density of the point cloud, the smaller
                     the value the higher the density
        crop_size: Edge lengths of point cloud slices in millimeters
        num_points: Number of points contained in the point cloud slice
    '''

    def __init__(self,
                 path_data: str,
                 label: str,
                 hfd_path_classes: str = './data/train/parts_classification',
                 pcl_density: int = 40,
                 crop_size: int = 400,
                 num_points: int = 2048,
                 profile=False,
                 skip_splitting=False,
                 skip_sampling=False,
                 skip_slicing=False,
                 fast_sampling=False,
                 decrease_lib=True,
                 ):
        self.path_data = path_data
        self.path_models = os.path.join(self.path_data, 'models')
        self.label = label
        self.path_classes = None
        self.hfd_path_classes = hfd_path_classes
        self.pcl_density = pcl_density
        self.crop_size = crop_size
        self.num_points = num_points
        self.profile = profile
        self.skip_splitting = skip_splitting
        self.skip_sampling = skip_sampling
        self.skip_slicing = skip_slicing
        self.fast_sampling = fast_sampling
        self.decrease_lib = decrease_lib
        # Make sure the directory structure is correct
        components = listdir(self.path_models)
        for component in components:
            if component.startswith('.'): continue
            files = listdir(os.path.join(self.path_models, component))
            for file in files:
                if os.path.splitext(file)[-1] == '.obj':
                    old_name = os.path.splitext(file)[0]
                    if not old_name == component:
                        os.rename(os.path.join(self.path_models, component, file),
                                  os.path.join(self.path_models, component, component + '.obj'))
                    file_data = ''
                    with open(os.path.join(self.path_models, component, component + '.obj'), 'r') as f:
                        for line in f:
                            if '.mtl' in line:
                                file_data += 'mtllib ' + component + '.mtl\n'
                            else:
                                file_data += line
                    with open(os.path.join(self.path_models, component, component + '.obj'), 'w') as f:
                        f.write(file_data)
                if os.path.splitext(file)[-1] == '.xml':
                    old_name = os.path.splitext(file)[0]
                    if not old_name == component:
                        os.rename(os.path.join(self.path_models, component, file),
                                  os.path.join(self.path_models, component, component + '.xml'))
                if os.path.splitext(file)[-1] == '.mtl':
                    old_name = os.path.splitext(file)[0]
                    if not old_name == component:
                        os.rename(os.path.join(self.path_models, component, file),
                                  os.path.join(self.path_models, component, component + '.mtl'))

    def make(self, free_cores=1):
        if self.label == 'PDL':
            self.path_classes = os.path.join(BASE, 'data', 'train', 'parts_classification')
            pdl = PDL(path_models=os.path.join(BASE, 'data', 'train', 'models'),
                      path_split=os.path.join(BASE, 'data', 'train', 'split'),
                      path_classes=self.path_classes)
            components = listdir(pdl.path_models)

            if self.profile:
                for comp in components:
                    if comp.startswith('.'): continue
                    path_to_comp = os.path.join(pdl.path_models, comp)
                    files = listdir(path_to_comp)
                    for file in files:
                        if os.path.splitext(file)[1] == '.obj':
                            pdl.split(os.path.join(path_to_comp, file))

            elif not self.profile and not self.skip_splitting:
                components = [component for component in components if
                              os.path.isdir(os.path.join(pdl.path_models, component))]  # remove non-folders
                nr_processes = max(min(len(components), cpu_count() - free_cores), 1)
                k, m = divmod(len(components), nr_processes)  # divide among processors
                split_components = list(
                    components[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
                args = split_components
                print(f'splitting... {nr_processes} workers ...', components)
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                with Pool(nr_processes) as p:
                    p.map(pdl.split_parallel, [_args for _args in args])

                print('splitting finished')
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            else:
                raise ValueError('Weird param combination')

            pdl.write_all_parts()
            pdl.label()

        elif self.label == 'HFD':
            self.path_classes = self.hfd_path_classes
        elif self.label == 'skip_split':
            self.path_classes = os.path.join(BASE, 'data', 'train', 'parts_classification')
            pass
        else:
            raise NotImplementedError
        f = open(os.path.join(self.path_classes, 'class_dict.pkl'), 'rb')
        class_dict = pickle.load(f)
        # a dict that stores current labels
        label_dict = {}
        i = 0
        for v in class_dict.values():
            if v not in label_dict:
                label_dict[v] = i
                i += 1
        with open(os.path.join(self.path_classes, 'label_dict.pkl'), 'wb') as tf:
            pickle.dump(label_dict, tf, protocol=2)
        # load the parts and corresponding labels from part feature extractor
        f = open(os.path.join(self.path_classes, 'label_dict.pkl'), 'rb')
        label_dict = pickle.load(f)
        label_dict_r = dict([val, key] for key, val in label_dict.items())

        # path to disassembled parts
        path_split = os.path.join(self.path_data, 'split')
        # folder to save unlabeled pc in xyz format
        path_xyz = os.path.join(self.path_data, 'unlabeled_pc')
        # folder to save labeled pc in pcd format
        path_pcd = os.path.join(self.path_data, 'labeled_pc')
        if not os.path.exists(path_xyz):
            os.makedirs(path_xyz)
        if not os.path.exists(path_pcd):
            os.makedirs(path_pcd)
        folders = listdir(path_split)

        if self.profile and not self.skip_sampling:
            for folder in folders:
                # for each component merge the labeled part mesh and sample mesh into pc
                if os.path.isdir(os.path.join(path_split, folder)) and self.label != 'debug':
                    print('sampling... ...', folder)
                    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    sample_and_label_alternative(os.path.join(path_split, folder), path_pcd, path_xyz, label_dict,
                                                 class_dict, self.pcl_density)
        elif not self.skip_sampling:
            folders = [folder for folder in folders if
                       os.path.isdir(os.path.join(path_split, folder))]  # remove non-folders
            nr_processes = max(min(len(folders), cpu_count() - free_cores), 1)
            k, m = divmod(len(folders), nr_processes)  # divide among processors
            split_folders = list(folders[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
            repeated_args = [[path_pcd, path_xyz, class_dict, label_dict, self.pcl_density, path_split,
                              self.fast_sampling]] * nr_processes
            args = [[_folders, *_args] for _args, _folders in zip(repeated_args, split_folders)]
            print(f'sampling... {nr_processes} workers ...', folders)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            with Pool(nr_processes) as p:
                p.map(sample_and_label_parallel, [_args for _args in args])

            print('sampling finished')
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        else:
            pass

        # path to dir of welding slices
        path_welding_zone = os.path.join(self.path_data, 'welding_zone')
        # path to lookup table
        path_lookup_table = os.path.join(self.path_data, 'lookup_table')
        if not os.path.exists(path_welding_zone):
            os.makedirs(path_welding_zone)
        if not os.path.exists(path_lookup_table):
            os.makedirs(path_lookup_table)
        files = listdir(self.path_models)
        print('Generate one point cloud slice per welding spot')

        if self.profile and not self.skip_slicing:
            i = 1
            for file in files:
                print(str(i) + '/' + str(len(files)), file)
                i += 1
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                pc_path = os.path.join(path_pcd, file + '.pcd')
                xml_path = os.path.join(self.path_models, file, file + '.xml')
                name = file
                slice.slice_one(pc_path, path_welding_zone, path_lookup_table, xml_path, name, self.crop_size,
                                self.num_points)
        elif not self.skip_slicing:
            files = [file for file in files if os.path.isdir(os.path.join(path_split, file))]  # remove non-folders
            nr_processes = max(min(len(files), cpu_count() - free_cores), 1)
            k, m = divmod(len(files), nr_processes)  # divide among processors
            split_files = list(files[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(nr_processes))
            repeated_args = [[path_welding_zone, path_lookup_table, self.crop_size, self.num_points, path_pcd,
                              self.path_models]] * nr_processes
            args = [[_files, *_args] for _args, _files in zip(repeated_args, split_files)]

            print(f'slicing... {nr_processes} workers ...', files)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            with Pool(nr_processes) as p:
                p.map(slice.slice_one_parallel, [_args for _args in args])

            print('slicing finished')
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        else:
            pass

        slice.merge_lookup_table(path_lookup_table)
        print('Extract feature dictionary from point cloud slices\n')
        slice.get_feature_dict(self.path_data, path_welding_zone, path_lookup_table, label_dict_r)
        if self.decrease_lib:
            print('Removing duplicate point cloud slices\n')
            slice.decrease_lib(self.path_data, self.path_train, path_welding_zone, label_dict_r)
        else:
            print('not reducing stuff :)')
            slice.decrease_lib_dummy(self.path_data, path_welding_zone, label_dict_r)
        slice.move_files(self.path_data)
        print('Use the normal information to generate an index for easy searching\n')
        slice.norm_index(self.path_data)
        print('FINISHED')


if __name__ == '__main__':
    args = parse_args()
    lut = LookupTable(path_data='./data', label=args.label, hfd_path_classes='./data/train/parts_classification',
                      pcl_density=args.pcl_density, crop_size=args.crop_size, num_points=args.num_points, \
                      skip_sampling=args.skip_sampling or args.skip_both,
                      skip_slicing=args.skip_slicing or args.skip_both, fast_sampling=args.fast_sampling,
                      decrease_lib=args.decrease_lib)
    lut.make(args.free_cores)