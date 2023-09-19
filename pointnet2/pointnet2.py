"""
Author: Benny
Date: Nov 2019
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch.nn as nn
import sys
import importlib
from pointnet2.data_utils.tools_dataset import *
import time
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
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
    return distance,seam_vector.astype(int)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg_siamese', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--dataset', default='data', type=str, help='pu1k or pugan')
    parser.add_argument('--input_num', default=2048, type=str, help='optimizer, adam or sgd')
    parser.add_argument('--file_path', default=os.path.join(ROOT_DIR,'data','Aehn3Test_welding_zone'), help='model name')
    parser.add_argument('--model_path', default=os.path.join(BASE_DIR,'checkpoints','best_model.pth'), help='model name')

    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def process_pc(file_path,pcs):
    datas = []
    names = []
    for pc in pcs:
        if pc.endswith('pcd'):
            tmp_name = os.path.join(file_path,pc)
            pcd=o3d.io.read_point_cloud(tmp_name)#路径需要根据实际情况设置
            input=np.asarray(pcd.points)#A已经变成n*3的矩阵

            lens = len(input)

            if lens < 2048:
                ratio = int(2048 /lens + 1)
                tmp_input = np.tile(input, (ratio, 1))
                input = tmp_input[:2048 ]

            if lens > 2048 :
                np.random.shuffle(input) # 每次取不一样的1024个点
                input = farthest_point_sampling(input,2048)

            datas.append(input)
            names.append(pc)

    return datas,names



def pointnet2(file_path,SNahts,tree,xml_path):
    args = parse_args()

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    siamese_model = MODEL.get_model().cuda()
    # criterion = MODEL.get_loss().cuda()
    siamese_model.apply(inplace_relu)

    sig = nn.Sigmoid()

    checkpoint = torch.load(args.model_path)
    start_epoch = checkpoint['epoch']
    siamese_model.load_state_dict(checkpoint['model_state_dict'])

    name_id = {}
    for Snaht in SNahts:
        Name = Snaht.attrib.get('Name')
        ID = Snaht.attrib.get('ID')
        name_id[Name] = ID
    print(name_id)
    pc_list = os.listdir(file_path)
    all_datas, all_names = process_pc(file_path, pc_list)
    retrieved_map = {}
    # query_pcs = ['PgmDef_260_0.pcd']
    tic = time.time()

    with torch.no_grad():
        # retrieved_map = {}
        # for SNaht_src in SNahts:
        #     dict = {}
        #     similar_list=[]
        #     similar_str = ''
        #     calculate_list=[]
        #     src_ID=SNaht_src.attrib.get('ID')
        #     src_name = SNaht_src.attrib.get('Name')
        #     src_path = file_path + '/' + src_name + '.pcd'
        #     if os.path.exists(src_path) != True:
        #         continue
        #     seam_length_src, seam_vec_src = get_distance(SNaht_src)
        #     pcd1 = o3d.io.read_point_cloud(src_path)
        #     point1 = np.array(pcd1.points).astype('float32')
        #     src = pc_normalize(point1)
        #     query_data = torch.from_numpy(src)[None, ...]
        #     query_data = query_data.float().cuda()
        #     query_data = query_data.transpose(2, 1)
        #     for SNaht_tgt in SNahts:
        #         tgt_ID=SNaht_tgt.attrib.get('ID')
        #         tgt_name = SNaht_tgt.attrib.get('Name')
        #         if src_name == tgt_name:
        #             continue
        #         tgt_path = file_path + '/' + tgt_name + '.pcd'
        #         if os.path.exists(tgt_path) != True:
        #             continue
        #         seam_length_tgt, seam_vec_tgt = get_distance(SNaht_tgt)
        #         pcd2 = o3d.io.read_point_cloud(tgt_path)
        #         point2 = np.array(pcd2.points).astype('float32')
        #         target = pc_normalize(point2)
        #
        #         seam_vec_diff = seam_vec_src - seam_vec_tgt
        #         if abs(seam_vec_diff[0]) > 3 or abs(seam_vec_diff[1]) > 3 or abs(seam_vec_diff[2]) > 3:
        #             continue
        #         if abs(seam_length_src - seam_length_tgt) > 5:
        #             continue
        #
        #         calculate_list.append(tgt_name)
        for pc_1 in pc_list:
            similar_list=[]
            query_id = all_names.index(pc_1)
            query_data = all_datas[query_id]
            query_data = pc_normalize(query_data)
            query_data = torch.from_numpy(query_data)[None, ...]
            query_data = query_data.float().cuda()
            query_data = query_data.transpose(2, 1)
            all_sim = []
            cat_data = query_data
            for pc_2 in pc_list:
                compare_id  =  all_names.index(pc_2)
                compare_data = all_datas[compare_id]
                compare_data = pc_normalize(compare_data)
                compare_data = torch.from_numpy(compare_data)[None,...]
                compare_data = compare_data.float().cuda()
                compare_data = compare_data.transpose(2, 1)
                cat_data=torch.cat([cat_data,compare_data],0)
            pc_sim = siamese_model(query_data, cat_data)
            for i in range(len(pc_sim)):
                score = sig(pc_sim[i])
                all_sim.append(score.item())
            toc=time.time()
            print('DL processing time:',toc-tic)
            st = np.argsort(all_sim)[::-1]
            for s in st:
                if all_sim[s]<0.95:
                    continue
                if all_names[s] == pc_1:
                    continue
                similar_list.append(name_id[all_names[s].split('.')[0]])
                string = '点云: ' + all_names[s] + ', 相似度: {}'.format(all_sim[s])
                print(string)
                print('similar_list',similar_list)
            retrieved_map[name_id[pc_1.split('.')[0]]]=similar_list
        print('retrieved_map',retrieved_map)
        for SNaht in SNahts:
            attr_dict={}
            for key, value in SNaht.attrib.items():
                if key == 'ID':
                    print(retrieved_map[value])
                    attr_dict[key] = value
                    attr_dict['Naht_ID'] = ','.join(retrieved_map[value])
                elif key == 'Naht_ID':
                    continue
                else:
                    attr_dict[key] = value
            SNaht.attrib.clear()
            for key, value in attr_dict.items():
                SNaht.set(key, value)
        tree.write(xml_path)
    return retrieved_map
if __name__ == '__main__':
    pointnet2()
