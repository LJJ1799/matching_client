import xml.etree.ElementTree as ET
import argparse
from models import Benchmark, IterativeBenchmark, icp, fgr
import os.path
from util import npy2pcd
from single_spot_table.obj_geo_based_classification import PFE,main
from lut import LookupTable
from tools import WeldScene,points2pcd,get_distance,get_weld_info
import open3d as o3d
import numpy as np
import time
import torch
import sys
from create_pc import split,convert

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH)
ROOT = os.path.dirname(BASE)

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--infer_npts', type=int, default=-1,
                        help='the points number of each pc for training')
    parser.add_argument('--in_dim', type=int, default=3,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--checkpoint', default='test_min_loss.pth',
                        help='the path to the trained checkpoint')
    parser.add_argument('--method', default='Benchmark',
                        help='choice=[benchmark, icp]')
    parser.add_argument('--cuda', default='True',action='store_true',
                        help='whether to use the cuda')
    parser.add_argument('--show',action='store_true',
                        help='whether to visualize')
    parser.add_argument('--xml',default='Reisch',
                        help='the points number of each pc for training')
    args = parser.parse_args()
    return args
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

def matching(args):
    start_time=time.time()
    xml_path=os.path.join('xml',args.xml+'.xml')
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
        if os.path.exists(os.path.join(wz_path,slice_name+'.pcd'))==False:
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
        src=point1
        for SNaht_tgt in SNahts:
            tgt_name=SNaht_tgt.attrib.get('Name')
            if src_name==tgt_name:
                continue
            tgt_path=wz_path + '/' + tgt_name + '.pcd'
            if os.path.exists(tgt_path)!=True:
                continue
            seam_length_tgt,seam_vec_tgt=get_distance(SNaht_tgt)
            pcd2 = o3d.io.read_point_cloud(tgt_path)
            point2 = np.array(pcd2.points).astype('float32')
            target = point2

            seam_vec_diff=seam_vec_src-seam_vec_tgt
            if abs(seam_vec_diff[0])>3 or abs(seam_vec_diff[1])>3 or abs(seam_vec_diff[2])>3:
                continue
            if abs(seam_length_src-seam_length_tgt)>5:
                continue

            if args.method == 'Benchmark':
                R, t, pred_ref_cloud=evaluate_benchmark(args, src, target, model)
                R_value = np.around(torch.mean(R).cpu().numpy(), decimals=4)
                if(R_value<0.3331):
                    continue

            elif args.method == 'icp':
                centroid1 = np.mean(src, axis=0)
                src = src - centroid1
                m1 = np.max(np.sqrt(np.sum(src ** 2, axis=1)))
                src = src / m1

                centroid2 = np.mean(target, axis=0)
                target = target - centroid2
                m2 = np.max(np.sqrt(np.sum(target ** 2, axis=1)))
                target = target / m2

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
    args=config_params()
    if args.method == 'Benchmark':
        model = IterativeBenchmark(in_dim=args.in_dim,
                                   niters=args.niters,
                                   gn=args.gn)
        if args.cuda:
            model = model.cuda()
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    matching(args)
