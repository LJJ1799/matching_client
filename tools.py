from utils.xml_parser import list2array,parse_frame_dump
from utils.foundation import load_pcd_data, points2pcd, fps
from utils.math_util import rotate_mat, rotation_matrix_from_vectors
import os.path
import sys



import open3d as o3d
import numpy as np
import copy
def get_distance(SNaht):
    list1 = []
    for punkt in SNaht.iter('Punkt'):
        list2 = []
        list2.append(float(punkt.attrib['X']))
        list2.append(float(punkt.attrib['Y']))
        list2.append(float(punkt.attrib['Z']))
        list1.append(list2)
    weld_info=np.asarray(list1)
    seam_vector=weld_info[-1,:]-weld_info[0,:]
    x_diff = np.max(weld_info[:, 0]) - np.min(weld_info[:, 0])
    if x_diff < 2:
        x_diff = 0
    y_diff = np.max(weld_info[:, 1]) - np.min(weld_info[:, 1])
    if y_diff < 2:
        y_diff = 0
    z_diff = np.max(weld_info[:, 2]) - np.min(weld_info[:, 2])
    if z_diff < 2:
        z_diff = 0
    distance = int(pow(pow(x_diff, 2) + pow(y_diff, 2) + pow(z_diff, 2), 0.5)) + 25
    print('list1',list1)
    print('seam_vector',seam_vector)
    return distance,seam_vector.astype(int)
def get_weld_info(xml_path):
    frames = list2array(parse_frame_dump(xml_path))
    weld_infos=[]
    for i in range(len(frames)):
        tmp = frames[frames[:, -2] == str(i)]
        if len(tmp) != 0:
            weld_infos.append(tmp)
    weld_infos=np.vstack(weld_infos)
    return weld_infos

def sample_and_label_alternative(path, path_pcd,label_dict, class_dict, density=40):
    '''Convert mesh to pointcloud
    two pc will be generated, one is .pcd format with labels, one is .xyz format withou labels
    Args:
        path (str): path to single component
        label_dict (dict): the class name with an index
        density (int): Sampling density, the smaller the value the greater the point cloud density
    '''
    # get the current component name
    namestr = os.path.split(path)[-1]
    files = os.listdir(path)
    # label_list = {}
    label_count = 0

    allpoints = np.zeros(shape=(1,4))
    for file in files:
        if os.path.splitext(file)[1] == '.obj':
            # load mesh
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))
            if np.asarray(mesh.triangles).shape[0] > 1:
                key = os.path.abspath(os.path.join(path, file))
                label = label_dict[class_dict[key]]
                # get number of points according to surface area
                number_points = int(mesh.get_surface_area()/density)
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
    points2pcd(os.path.join(path_pcd, namestr+'.pcd'), allpoints[1:])


def points2pcd(path, points):
    """
    path: ***/***/1.pcd
    points: ndarray, xyz+lable
    """
    point_num = points.shape[0]
    # handle.write('VERSION .7\nFIELDS x y z label object\nSIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1')
    # string = '\nWIDTH '+str(point_num)
    # handle.write(string)
    # handle.write('\nHEIGHT 1')
    # string = '\nPOINTS '+str(point_num)
    # handle.write(string)
    # handle.write('\nVIEWPOINT 0 0 0 1 0 0 0')
    # handle.write('\nDATA ascii')
    content = ''
    content += 'VERSION .7\nFIELDS x y z label object\nSIZE 4 4 4 4 4\nTYPE F F F I I\nCOUNT 1 1 1 1 1'
    content += '\nWIDTH ' + str(points.shape[0])
    content += '\nHEIGHT 1'
    content += '\nPOINTS ' + str(point_num)
    content += '\nVIEWPOINT 0 0 0 1 0 0 0'
    content += '\nDATA ascii'
    obj = -1 * np.ones((point_num, 1))
    points_f = np.c_[points, obj]
    for i in range(point_num):
        content += '\n' + str(points_f[i, 0]) + ' ' + str(points_f[i, 1]) + ' ' + \
                   str(points_f[i, 2]) + ' ' + str(int(points_f[i, 3])) + ' ' + str(int(points_f[i, 4]))

    handle = open(path, 'w')
    handle.write(content)
    handle.close()
class WeldScene:
    '''
    Component point cloud processing, mainly for slicing

    Attributes:
        path_pc: path to labeled pc

    '''

    def __init__(self, pc_path):
        self.pc = o3d.geometry.PointCloud()
        pcd=o3d.io.read_point_cloud(pc_path)
        # print (xyzl.shape)
        self.xyz = np.asarray(pcd.points)
        self.pc.points = o3d.utility.Vector3dVector(np.asarray(self.xyz))

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
        # print(weld_info[2][1:4])
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

    def crop(self, weld_info,num_points=2048, vis=False):
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
        weld_seam_points=weld_info[:,1:4]
        weld_seam=o3d.geometry.PointCloud()
        weld_seam.points=o3d.utility.Vector3dVector(weld_seam_points)
        distance,translate=self.get_distance_and_translate(weld_info)
        extent = 200
        # crop_extent = np.array([max(x_diff,extent), max(y_diff,extent),max(z_diff,extent)])
        crop_extent=np.array([distance,extent+5,extent+5])
        # move the coordinate center to the welding spot
        pc.translate(-translate)
        weld_seam.translate(-translate)
        # rotation at this welding spot  1.
        rot = weld_info[0,10:13] * np.pi / 180
        rotation = rotate_mat(axis=[1, 0, 0], radian=rot[0])

        tf1 = np.zeros((4, 4))
        tf1[3, 3] = 1.0
        tf1[0:3, 0:3] = rotation
        # pc.transform(tf)
        # weld_seam.transform(tf)
        # new normals
        norm1 = np.around(weld_info[0, 4:7], decimals=6)
        norm2 = np.around(weld_info[0, 7:10], decimals=6)


        norm1_r = np.matmul(rotation, norm1.T)
        norm2_r = np.matmul(rotation, norm2.T)
        # torch pose
        pose = np.zeros((3, 3))
        for i in range(weld_info.shape[0]):
            pose[0:3, 0] = weld_info[i,14:17]
            pose[0:3, 1] = weld_info[i,17:20]
            pose[0:3, 2] = weld_info[i,20:23]
        # cauculate the new pose after rotation
            pose_new = np.matmul(rotation, pose)

            weld_info[i,4:7] = norm1_r
            weld_info[i,7:10] = norm2_r
            weld_info[i,14:17] = pose_new[0:3, 0]
            weld_info[i,17:20] = pose_new[0:3, 1]
            weld_info[i,20:23] = pose_new[0:3, 2]

        coor1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
        mesh_arrow1 = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=20 * 1,
            cone_radius=1.5 * 1,
            cylinder_height=20 * 1,
            cylinder_radius=1.5 * 1
        )
        mesh_arrow1.paint_uniform_color([0, 0, 1])

        mesh_arrow2 = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=20 * 1,
            cone_radius=1.5 * 1,
            cylinder_height=20 * 1,
            cylinder_radius=1.5 * 1
        )
        mesh_arrow2.paint_uniform_color([0, 1, 0])
        norm_ori = np.array([0, 0, 1])
        # bounding box of cutting area
        rotation_bbox = rotation_matrix_from_vectors(norm_ori, norm_ori)
        seams_direction=np.cross(norm1,norm2)
        if (abs(seams_direction[0])==0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)) or (abs(seams_direction[0])!=0 and (abs(seams_direction[1])!=0 or abs(seams_direction[2])!=0)):
            rotation_bbox=self.rotation(np.array([1,0,0]),seams_direction)
        center_bbox=norm2/np.linalg.norm(norm2)*extent/2+norm1/np.linalg.norm(norm1)*extent/2
        bbox = o3d.geometry.OrientedBoundingBox(center=center_bbox, R=rotation_bbox,extent=crop_extent)

        pc.paint_uniform_color([1,0,0])
        weld_seam.paint_uniform_color([0,1,0])
        cropped_pc_large=pc.crop(bbox)
        idx_crop_large=bbox.get_point_indices_within_bounding_box(pc.points)
        xyz_crop = self.xyz[idx_crop_large]
        xyz_crop -= translate
        xyz_crop_new = np.matmul(rotation_matrix_from_vectors(norm_ori, norm_ori), xyz_crop.T).T


        while xyz_crop_new.shape[0] < num_points:
            xyz_crop_new = np.vstack((xyz_crop_new, xyz_crop_new))
        xyz_crop_new = fps(xyz_crop_new, num_points)
        if vis:
            o3d.visualization.draw_geometries([cropped_pc_large,bbox])
        return xyz_crop_new, cropped_pc_large, weld_info
if __name__ == "__main__":
    xml_path = 'xml/Reisch.xml'
    weld_info=get_weld_info(xml_path)

