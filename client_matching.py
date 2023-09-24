import xml.etree.ElementTree as ET
# import open3d.core as o3c
from pointnn.save_pn_feature import save_feature
from pointnn.cossim import pointnn
from pointnet2.pointnet2 import pointnet2
from pointnext.pointnext import pointnext
from ICP_RMSE import ICP
import os.path
from util import npy2pcd
from tools import get_ground_truth,get_weld_info,WeldScene
from evaluation import mean_metric
import open3d as o3d
import numpy as np
import time
import torch
import shutil
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
# ROOT = os.path.dirname(BASE)

def matching(xml_file,model,auto_del=False):
    start_time=time.time()
    xml_path=xml_file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    Baugruppe=root.attrib['Baugruppe']
    data_path = os.path.join(ROOT,'data')
    wz_path = os.path.join(data_path,Baugruppe+'_welding_zone')
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
        split(data_path,Baugruppe)
        convert(data_path,40,Baugruppe)

    inter_time=time.time()
    print('creating pointcloud time',inter_time-start_time)
    os.makedirs(wz_path,exist_ok=True)
    ws = WeldScene(os.path.join(data_path,Baugruppe+'.pcd'))
    weld_infos=get_weld_info(xml_path)
    gt_map=get_ground_truth(weld_infos)
    print(gt_map)
    weld_infos=np.vstack(weld_infos)
    SNahts = root.findall("SNaht")

    for SNaht in SNahts:
        slice_name = SNaht.attrib['Name']
        # if os.path.exists(os.path.join(wz_path,slice_name+'.pcd'))==False:
        weld_info=weld_infos[weld_infos[:,0]==slice_name][:,3:].astype(float)
        if len(weld_info)==0:
            continue
        cxy, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cxy)
        o3d.io.write_point_cloud(os.path.join(wz_path, slice_name + '.pcd'), pointcloud=pc, write_ascii=True)
    retrieved_map={}
    if model == 'icp':
        print('run icp')
        retrieved_map=ICP(SNahts,wz_path,tree,xml_path)

    elif model == 'pointnn':
        save_feature(wz_path)
        retrieved_map=pointnn(SNahts,tree,xml_path)

    elif model == 'pointnet2':
        retrieved_map=pointnet2(wz_path,SNahts,tree,xml_path)

    elif model == 'pointnext':
        retrieved_map=pointnext(wz_path,SNahts,tree,xml_path)

    print('gt_map',gt_map)
    print('retrieved_map',retrieved_map)
    metrice=mean_metric(gt_map,retrieved_map)
    print(metrice)
    if auto_del:
        shutil.rmtree(wz_path)
    end_time=time.time()
    print('total time=',end_time-start_time)
    return

if __name__ == "__main__":
    model = 'pointnet2'
    matching(os.path.join(ROOT,'data','Reisch.xml'),model)
