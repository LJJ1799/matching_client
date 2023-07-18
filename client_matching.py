from utils.xml_parser import list2array,parse_frame_dump
import os.path
import xml.etree.ElementTree as ET
from obj_geo_based_classification import PFE,main
from lut import LookupTable
from tools import WeldScene,points2pcd,get_weld_info
import open3d as o3d
import numpy as np
import time

def get_distance(SNaht):
    list1 = []
    for punkt in SNaht.iter('Punkt'):
        list2 = []
        list2.append(float(punkt.attrib['X']))
        list2.append(float(punkt.attrib['X']))
        list2.append(float(punkt.attrib['X']))
        list1.append(list2)
    weld_info=np.asarray(list1)
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
    return distance

def matching(xml_file):
    start_time=time.time()
    xml_path=os.path.join('xml',xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    Baugruppe=root.attrib['Baugruppe']
    data_path = os.path.join('data',Baugruppe)
    pc_path = os.path.join(data_path,'labeled_pc')
    wz_path = os.path.join(data_path,'welding_zone')
    if not os.path.exists(os.path.join(pc_path,Baugruppe+'.pcd')):
        pfe = PFE(path_models=os.path.join(data_path,'models'),
                  path_split=os.path.join(data_path,'split'),
                  path_label_temp=os.path.join(data_path, 'label_temp_folder'),
                  path_classes=os.path.join(data_path, 'parts_classification'),
                  parallel=False,
                  n_clusters=2,
                  )
        main(pfe)
        lut = LookupTable(path_data=data_path, label='HFD', hfd_path_classes=os.path.join(data_path,'parts_classification'),
                          pcl_density=40, crop_size=400, num_points=2048, \
                          skip_sampling=False,
                          skip_slicing=True, fast_sampling=True,
                          decrease_lib=False)
        lut.make(2)
    inter_time=time.time()
    print('creating pointcloud time',inter_time-start_time)
    ws = WeldScene(os.path.join(pc_path,Baugruppe+'.pcd'))
    weld_infos=get_weld_info(xml_path)
    for SNaht in root.iter('SNaht'):
        slice_name = SNaht.attrib['Name']
        if os.path.exists(os.path.join(wz_path,slice_name,'.pcd'))==False:
            weld_info=weld_infos[weld_infos[:,0]==slice_name][:,3:].astype(float)
            if len(weld_info)==0:
                continue
            cxyzl, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
            points2pcd(os.path.join(wz_path, slice_name + '.pcd'), cxyzl)
    SNahts = root.findall("SNaht")
    for SNaht_src in SNahts:
        dict={}
        similar_str=''
        src_name=SNaht_src.attrib.get('Name')
        src_path=wz_path + '/' + src_name + '.pcd'
        if os.path.exists(src_path)!=True:
            continue
        seam_length_src=get_distance(SNaht_src)
        pcd1=o3d.io.read_point_cloud(src_path)
        point1=np.array(pcd1.points).astype('float32')
        centroid1=np.mean(point1,axis=0)
        point1=point1-centroid1
        m1 = np.max(np.sqrt(np.sum(point1 ** 2, axis=1)))
        point1=point1/m1
        src=point1
        for SNaht_tgt in SNahts:
            tgt_name=SNaht_tgt.attrib.get('Name')
            if src_name==tgt_name:
                continue
            tgt_path=wz_path + '/' + tgt_name + '.pcd'
            if os.path.exists(tgt_path)!=True:
                continue
            seam_length_tgt=get_distance(SNaht_tgt)
            pcd2 = o3d.io.read_point_cloud(tgt_path)
            point2 = np.array(pcd2.points).astype('float32')
            centroid2 = np.mean(point2, axis=0)
            point2 = point2 - centroid2
            m2 = np.max(np.sqrt(np.sum(point2 ** 2, axis=1)))
            point2 = point2 / m2
            target = point2

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
            src_cloud.paint_uniform_color([1, 0, 0])
            tgt_cloud.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([src_cloud, tgt_cloud], width=800)
            if mean_distance_s_t > 0.04 or rmse_s_t > 0.04 or correspondence_s_t < 1900 or mean_distance_t_s > 0.04 or rmse_t_s > 0.04 or correspondence_t_s < 1900 or abs(seam_length_src-seam_length_tgt)>10:
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