import xml.etree.ElementTree as ET
import open3d.core as o3c
import json
import argparse
from pointnn.save_pn_feature import save_feature
from pointnn.cossim import pointnn
from ICP_RMSE import ICP
from PCReg import Benchmark, IterativeBenchmark, icp, fgr
import os.path
from util import npy2pcd
from single_spot_table.obj_geo_based_classification import PFE,main
from lut import LookupTable
from tools import points2pcd,get_distance,get_weld_info,WeldScene
import open3d as o3d
import numpy as np
import time
import torch
import sys
import shutil
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(CURRENT_PATH)
# ROOT = os.path.dirname(BASE)

def evaluate_benchmark(args,src,tgt,model):
    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    model.eval()

    centroid1 = np.mean(src, axis=0)
    src = src - centroid1
    m1 = np.max(np.sqrt(np.sum(src ** 2, axis=1)))
    src = src / m1

    centroid2 = np.mean(tgt, axis=0)
    tgt = tgt - centroid2
    m2 = np.max(np.sqrt(np.sum(tgt ** 2, axis=1)))
    tgt = tgt / m2

    with torch.no_grad():
        src_cloud = torch.from_numpy(src).cuda()
        tgt_cloud = torch.from_numpy(tgt).cuda()
        src_cloud=torch.unsqueeze(src_cloud,0)
        tgt_cloud=torch.unsqueeze(tgt_cloud,0)
        tic = time.time()
        R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                tgt_cloud.permute(0, 2, 1).contiguous())
        # toc = time.time()
        # dura.append(toc - tic)
        # cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        # cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        # r_mse.append(cur_r_mse)
        # r_mae.append(cur_r_mae)
        # t_mse.append(cur_t_mse)
        # t_mae.append(cur_t_mae)
        # r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        # t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

        if args.show:
            tgt_cloud = torch.squeeze(tgt_cloud).cpu().numpy()
            src_cloud = torch.squeeze(src_cloud).cpu().numpy()
            result_cloud = torch.squeeze(pred_ref_cloud[-1]).cpu().numpy()
            pcd1 = npy2pcd(src_cloud*m1+centroid1, 0)
            pcd2 = npy2pcd(tgt_cloud*m2+centroid2, 1)
            pcd3 = npy2pcd(result_cloud*m1+centroid1, 2)
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    # r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
    #     summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return R, t, pred_ref_cloud

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

    if model == 'icp':
        ICP(SNahts,wz_path,tree,xml_path)

    elif model == 'pointnn':
        save_feature(wz_path)
        pointnn(SNahts,tree,xml_path)

    if auto_del:
        shutil.rmtree(wz_path)
    end_time=time.time()
    print('total time=',end_time-start_time)
    return

if __name__ == "__main__":
    model = 'icp'
    matching(os.path.join(ROOT,'data','Aehn3TestJob1.xml'),model)

