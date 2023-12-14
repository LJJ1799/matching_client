import xml.etree.ElementTree as ET
# import open3d.core as o3c
# from single_spot_table.obj_geo_based_classification import PFE,main as clustering
# from lut import LookupTable
from pointnn.save_pn_feature import save_feature
from pointnn.cossim import pointnn
from pointnet2.main import pointnet2
from pointnext.main import pointnext
from ICP_RMSE import ICP
from poseE.main import poseestimation
# from PoseEstimation.train import TrainPointNet2
# from PoseEstimation.test import PoseLookup
import os.path
from tools import get_ground_truth,get_weld_info,WeldScene,image_save
from evaluation import mean_metric
import open3d as o3d
import numpy as np
import time
import shutil
# from model_splitter import split_models
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
# ROOT = os.path.dirname(BASE)




def matching(data_folder,xml_file,model,dienst_number,pose_estimation=True,save_image=False,auto_del=False):
    start_time=time.time()
    xml_path=os.path.join(ROOT,data_folder,xml_file)
    data_path = data_folder
    # if model == 'POSE':
    #     # split_models('data3', 'data3/train/models','data3/test/models')
    #     # pfe = PFE(path_models=os.path.join(ROOT, 'data3', 'train', 'models'),
    #     #           path_split=os.path.join(ROOT, 'data3', 'train', 'split'),
    #     #           path_label_temp=os.path.join(ROOT, 'data3', 'train', 'label_temp_folder'),
    #     #           path_classes=os.path.join(ROOT, 'data3', 'train', 'parts_classification'),
    #     #           parallel=True,
    #     #           n_clusters=8,
    #     #           )
    #     # clustering(pfe)
    #     # lut = LookupTable(path_data=data_path, label='HFD',
    #     #                   hfd_path_classes=os.path.join(data_path, 'train/parts_classification'),
    #     #                   pcl_density=40, crop_size=400, num_points=2048, \
    #     #                   skip_sampling=False,
    #     #                   skip_slicing=False, fast_sampling=True,
    #     #                   decrease_lib=False)
    #     # lut.make(2)
    #     # tr=TrainPointNet2(path_data=data_path)
    #     # tr.make_dataset(crop_size=255,num_points=2048)
    #     te=PoseLookup(path_data=data_path)
    #     # te.processing(path_test=str(data_path)+'/test/models/')
    #     # tr.train()
    #     te.inference(model_path='./data3/seg_model/model1.ckpt',test_input='./data3/test/welding_zone_test')
    # else:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    SNahts = root.findall("SNaht")
    Baugruppe = root.attrib['Baugruppe']
    wz_path = os.path.join(data_path, Baugruppe)
    xml_output_path = os.path.join(ROOT, 'output')
    weld_infos=get_weld_info(xml_path)
    gt_id_map,gt_name_map=get_ground_truth(weld_infos)
    weld_infos=np.vstack(weld_infos)
    os.makedirs(xml_output_path, exist_ok=True)
    os.makedirs(wz_path,exist_ok=True)
    ws = WeldScene(os.path.join(data_path,Baugruppe+'.pcd'))
    slice_name_list=[]
    if not os.path.exists(os.path.join(data_path,Baugruppe+'.pcd')):
        split(data_path,Baugruppe)
        convert(data_path,40,Baugruppe)

        inter_time=time.time()
        print('creating pointcloud time',inter_time-start_time)

    for SNaht in SNahts:
        slice_name = SNaht.attrib['Name']
        # if os.path.exists(os.path.join(wz_path,slice_name+'.pcd'))==False:
        weld_info=weld_infos[weld_infos[:,0]==slice_name][:,3:].astype(float)
        if len(weld_info)==0:
            continue
        slice_name_list.append(slice_name + '.pcd')
        cxy, cpc, new_weld_info = ws.crop(weld_info=weld_info, num_points=2048)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cxy)
        o3d.io.write_point_cloud(os.path.join(wz_path, slice_name + '.pcd'), pointcloud=pc, write_ascii=True)

    retrieved_map={}
    methoe_time=time.time()
    if model == 'icp':
        print('run icp')
        retrieved_map,retrieved_map_name,tree=ICP(SNahts,wz_path,tree,xml_path)

    elif model == 'pointnn':
        print('run pointnn')
        save_feature(wz_path,slice_name_list)
        retrieved_map=pointnn(SNahts,tree,xml_path)

    elif model == 'pointnet2':
        if dienst_number==61:
            print('training pointnet2')
            os.system('python pointnet2/train_siamese_fortools.py --file_path data/Reisch')
            print("pointnet2 training finished")
            return
        elif dienst_number==63:
            print('run pointnet2')
            retrieved_map,retrieved_map_name,tree=pointnet2(wz_path,SNahts,tree,xml_path,slice_name_list)

    elif model == 'pointnext':
        if dienst_number==61:
            print('training pointnext')
            os.system('python pointnext/classification/main.py --file_path data/Reisch')
            print("pointnext training finished")
            return
        elif dienst_number==63:
            print('run pointnext')
            retrieved_map,retrieved_map_name,tree=pointnext(wz_path,SNahts,tree,xml_path,slice_name_list)
    print('retrieved_map_name',retrieved_map_name)
    #
    tree.write(os.path.join(xml_output_path, Baugruppe + '_similar.xml'))
    if pose_estimation:
        print('POSE ESTIMATION')
        tree=poseestimation(data_path,wz_path,xml_path,SNahts,tree,retrieved_map_name,vis=True)
        tree.write(os.path.join(xml_output_path, Baugruppe + '_predict.xml'))
    #
    # tree.write(os.path.join(xml_output_path,Baugruppe+'.xml'))
    # if save_image:
    #     image_save(retrieved_map_name,wz_path)
    #
    # print('gt_map',gt_id_map)
    # print('retrieved_map',retrieved_map)
    #
    # metric=mean_metric(gt_id_map,retrieved_map)
    # print('metric',metric)
    # if auto_del:
    #     shutil.rmtree(wz_path)
    # end_time=time.time()
    # total_time=end_time-methoe_time
    # print(model,' running time= ',total_time)
    return

if __name__ == "__main__":

    data_folder=os.path.join(ROOT,'data')
    xml='5012.xml'
    model='pointnext'
    pose_estimation=True
    dienst_number=61## 1 training_similarity;2 predict torch pose; 3 training LUT
    matching(data_folder, xml, model,dienst_number,pose_estimation=True,save_image=False,auto_del=False)

