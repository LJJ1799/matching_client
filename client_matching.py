import xml.etree.ElementTree as ET
# import open3d.core as o3c
from pointnn.save_pn_feature import save_feature
from pointnn.cossim import pointnn
from pointnet2.main import pointnet2
from pointnext.main import pointnext
from ICP_RMSE import ICP
import os.path
from tools import get_ground_truth,get_weld_info,WeldScene
from evaluation import mean_metric
import open3d as o3d
import numpy as np
import time
import shutil
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
# ROOT = os.path.dirname(BASE)

def matching(data_folder,xml_file,model,dienst_number,save_image=False,auto_del=False):
    start_time=time.time()
    xml_path=os.path.join(ROOT,data_folder,xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    SNahts = root.findall("SNaht")
    Baugruppe=root.attrib['Baugruppe']
    data_path = data_folder
    wz_path = os.path.join(data_path,Baugruppe)
    if not os.path.exists(os.path.join(data_path,Baugruppe+'.pcd')):
        split(data_path,Baugruppe)
        convert(data_path,40,Baugruppe)

    inter_time=time.time()
    print('creating pointcloud time',inter_time-start_time)
    os.makedirs(wz_path,exist_ok=True)
    ws = WeldScene(os.path.join(data_path,Baugruppe+'.pcd'))
    weld_infos=get_weld_info(xml_path)
    gt_id_map,gt_name_map=get_ground_truth(weld_infos)
    weld_infos=np.vstack(weld_infos)
    slice_name_list=[]
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
        retrieved_map,retrieved_map_name=ICP(SNahts,wz_path,tree,xml_path)

    elif model == 'pointnn':
        print('run pointnn')
        save_feature(wz_path,slice_name_list)
        retrieved_map=pointnn(SNahts,tree,xml_path)

    elif model == 'pointnet2':
        if dienst_number==60:
            print('training pointnet2')
            os.system('python pointnet2/train_siamese_fortools.py --file_path data/Reisch')
            print("pointnet2 training finished")
            return
        elif dienst_number==61:
            print('run pointnet2')
            retrieved_map,retrieved_map_name=pointnet2(wz_path,SNahts,tree,xml_path,slice_name_list)

    elif model == 'pointnext':
        if dienst_number==60:
            print('training pointnext')
            os.system('python pointnext/classification/main.py --file_path data/Reisch')
            print("pointnext training finished")
            return
        elif dienst_number==61:
            print('run pointnext')
            retrieved_map,retrieved_map_name=pointnext(wz_path,SNahts,tree,xml_path,slice_name_list)
    # print('retrieved_map_name',retrieved_map_name)

    if save_image:
        result_image_dir=os.path.join(ROOT,'result_image')
        os.makedirs(result_image_dir,exist_ok=True)
        for key,values in retrieved_map_name.items():
            if len(values):
                query_img_dir=os.path.join(result_image_dir,key)
                os.makedirs(query_img_dir,exist_ok=True)
                pcd1=o3d.io.read_point_cloud(os.path.join(wz_path,key+'.pcd'))
                point1=np.array(pcd1.points).astype('float32')
                point1_cloud=o3d.geometry.PointCloud()
                point1_cloud.points = o3d.utility.Vector3dVector(point1)
                for value in values:
                    pcd2=o3d.io.read_point_cloud(os.path.join(wz_path,value+'.pcd'))
                    point2=np.array(pcd2.points).astype('float32')
                    point2_cloud = o3d.geometry.PointCloud()
                    point2_cloud.points = o3d.utility.Vector3dVector(point2)

                    point1_cloud.paint_uniform_color([1, 0, 0])
                    point2_cloud.paint_uniform_color([0, 1, 0])
                    all_point=point1_cloud+point2_cloud


                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    vis.add_geometry(all_point)
                    vis.update_geometry(all_point)
                    vis.poll_events()
                    vis.update_renderer()

                    save_name=key+'_'+value
                    save_file=os.path.join(query_img_dir,save_name+'.png')
                    vis.capture_screen_image(save_file)
                    vis.destroy_window()
                    time.sleep(0.2)

    # print('gt_map',gt_id_map)
    # print('retrieved_map',retrieved_map)

    metric=mean_metric(gt_id_map,retrieved_map)
    print('metric',metric)
    if auto_del:
        shutil.rmtree(wz_path)
    end_time=time.time()
    total_time=end_time-methoe_time
    print(model,' running time= ',total_time)
    return

if __name__ == "__main__":

    data_folder=os.path.join(ROOT,'data')
    xml='Reisch.xml'
    model='pointnet2'
    dienst_number=0## 1 training_similarity;2 predict torch pose; 3 training LUT
    matching(data_folder, xml, model,dienst_number,save_image=False,auto_del=False)

