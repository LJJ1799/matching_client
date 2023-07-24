import xml.etree.ElementTree as ET
import os.path
from single_spot_table.obj_geo_based_classification import PFE,main
from lut import LookupTable
from tools import WeldScene,points2pcd,get_distance,get_weld_info
import open3d as o3d
import numpy as np
import time
import sys
from create_pc import split,convert
CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH)
ROOT = os.path.dirname(BASE)
# def get_weld_info(xml_path):
#     frames = list2array(parse_frame_dump(xml_path))
#     weld_infos=[]
#     for i in range(len(frames)):
#         tmp = frames[frames[:, -2] == str(i)]
#         if len(tmp) != 0:
#             weld_infos.append(tmp)
#     weld_infos=np.vstack(weld_infos)
#     return weld_infos

def matching(xml_file):
    start_time=time.time()
    xml_path=os.path.join('xml',xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    Baugruppe=root.attrib['Baugruppe']
    data_path = os.path.join('data',Baugruppe)
    pc_path = os.path.join(data_path,'labeled_pc')
    wz_path = os.path.join(data_path,'welding_zone')
    if not os.path.exists(os.path.join(data_path,Baugruppe+'.pcd')):
        # pfe = PFE(path_models=os.path.join(BASE,data_path,'models'),
        #           path_split=os.path.join(BASE,data_path,'split'),
        #           path_label_temp=os.path.join(BASE,data_path, 'label_temp_folder'),
        #           path_classes=os.path.join(BASE,data_path, 'parts_classification'),
        #           parallel=False,
        #           n_clusters=2,
        #           )
        # main(pfe)
        # lut = LookupTable(path_data=os.path.join(BASE,data_path), label='HFD', hfd_path_classes=os.path.join(BASE,data_path,'parts_classification'),
        #                   pcl_density=40, crop_size=400, num_points=2048, \
        #                   skip_sampling=False,
        #                   skip_slicing=True, fast_sampling=True,
        #                   decrease_lib=False)
        # lut.make(2)
        split(data_path)
        convert(data_path,40)

    inter_time=time.time()
    print('creating pointcloud time',inter_time-start_time)
    os.makedirs(wz_path,exist_ok=True)
    ws = WeldScene(os.path.join(data_path,Baugruppe+'.pcd'))
    weld_infos=get_weld_info(xml_path)
    for SNaht in root.iter('SNaht'):
        slice_name = SNaht.attrib['Name']
        if os.path.exists(os.path.join(wz_path,slice_name,'.pcd'))==False:
            weld_info=weld_infos[weld_infos[:,0]==slice_name][:,3:].astype(float)
            if len(weld_info)==0:
                continue
            cxyz, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(cxyz)
            o3d.io.write_point_cloud(os.path.join(wz_path, slice_name + '.pcd'), pointcloud=pc, write_ascii=True)
            # points2pcd(os.path.join(wz_path, slice_name + '.pcd'), cxyzl)
    SNahts = root.findall("SNaht")
    for SNaht_src in SNahts:
        dict={}
        similar_str=''
        src_name=SNaht_src.attrib.get('Name')
        src_path=wz_path + '/' + src_name + '.pcd'
        if os.path.exists(src_path)!=True:
            continue
        seam_length_src,seam_vec_src=get_distance(SNaht_src)
        pcd1=o3d.io.read_point_cloud(src_path)
        point1=np.array(pcd1.points).astype('float32')
        centroid1=np.mean(point1,axis=0)
        point1=point1-centroid1
        m1 = np.max(np.sqrt(np.sum(point1 ** 2, axis=1)))
        point1=point1/m1
        src=point1
        for SNaht_tgt in SNahts:
            tgt_name=SNaht_tgt.attrib.get('Name')
            print(src_name,tgt_name)
            if src_name==tgt_name:
                continue
            tgt_path=wz_path + '/' + tgt_name + '.pcd'
            if os.path.exists(tgt_path)!=True:
                continue
            seam_length_tgt,seam_vec_tgt=get_distance(SNaht_tgt)
            pcd2 = o3d.io.read_point_cloud(tgt_path)
            point2 = np.array(pcd2.points).astype('float32')
            centroid2 = np.mean(point2, axis=0)
            point2 = point2 - centroid2
            m2 = np.max(np.sqrt(np.sum(point2 ** 2, axis=1)))
            point2 = point2 / m2
            target = point2

            seam_vec_diff=seam_vec_src-seam_vec_tgt
            if abs(seam_vec_diff[0])>3 or abs(seam_vec_diff[1])>3 or abs(seam_vec_diff[2])>3:
                continue
            if abs(seam_length_src-seam_length_tgt)>5:
                continue
            src_cloud = o3d.geometry.PointCloud()
            src_cloud.points = o3d.utility.Vector3dVector(src)
            tgt_cloud = o3d.geometry.PointCloud()
            tgt_cloud.points = o3d.utility.Vector3dVector(target)
            icp_s_t = o3d.pipelines.registration.registration_icp(source=src_cloud, target=tgt_cloud,
                                                                  max_correspondence_distance=0.2,
                                                                  estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
            mean_distance_s_t = np.mean(src_cloud.compute_point_cloud_distance(tgt_cloud))
            fitness_s_t = icp_s_t.fitness
            rmse_s_t = icp_s_t.inlier_rmse
            correspondence_s_t = len(np.asarray(icp_s_t.correspondence_set))
            if mean_distance_s_t > 0.03 or rmse_s_t > 0.03 or correspondence_s_t < 1900:
                continue

            icp_t_s = o3d.pipelines.registration.registration_icp(source=tgt_cloud, target=src_cloud,
                                                                  max_correspondence_distance=0.2,
                                                                  estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

            mean_distance_t_s = np.mean(tgt_cloud.compute_point_cloud_distance(src_cloud))
            fitness_t_s = icp_t_s.fitness
            rmse_t_s = icp_t_s.inlier_rmse
            correspondence_t_s = len(np.asarray(icp_t_s.correspondence_set))
            src_cloud.paint_uniform_color([1, 0, 0])
            tgt_cloud.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([src_cloud, tgt_cloud], width=800)
            if mean_distance_t_s > 0.03 or rmse_t_s > 0.03 or correspondence_t_s < 1900:
                continue
            if similar_str=='':
                similar_str+=SNaht_tgt.attrib.get('ID')
            else:
                similar_str += (','+SNaht_tgt.attrib.get('ID'))

        for key,value in SNaht_src.attrib.items():
            if key=='ID':
                dict[key]=value
                dict['Naht_IDs']=similar_str
            elif key=='Naht_IDs':
                continue
            else:
                dict[key]=value
        SNaht_src.attrib.clear()
        for key,value in dict.items():
            SNaht_src.set(key,value)
        tree.write(xml_path)
    end_time=time.time()
    print('total time=',end_time-start_time)
    return

if __name__ == "__main__":
    matching('Reisch.xml')
    # frame = list2array(parse_frame_dump(os.path.join('xml','Reisch.xml')))
    # print(frame)