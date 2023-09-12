from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import open3d as o3d
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def pointnn(SNahts,tree,xml_path):
    name_id_dict={}
    for SNaht in SNahts:
        slice_name = SNaht.attrib['Name']
        name_id_dict[slice_name] = SNaht.attrib['ID']


    feature_path = os.path.join(BASE_DIR, 'cnn_feature')
    tmp = np.load(os.path.join(feature_path, 'pnn_tpc_cnn_feature.npz'), allow_pickle=True)
    features = tmp['cnn_feature'].squeeze()
    names = [str(tname) for tname in tmp['name']]

    similar_dict={}

    for query_name in names:
        similar_slice = ''

    # query_name = os.path.join(parent_DIR,'data','Aehn3Test_welding_zone','AutoDetect_11_15.pcd')
        print(query_name)
        for ii, nn in enumerate(names):
            if query_name in nn:
                query_id = ii
                break
        query_pcf = features[query_id]
        query_pcn = names[query_id]


        similarity = cosine_similarity(features, np.reshape(query_pcf, (1, -1))).reshape(-1)

        sorted_idx = np.argsort(similarity)[::-1]

        for idx in sorted_idx:
            cname = str(names[idx])
            if idx == query_id:
                print(cname + ' query pcd! ' + str(idx) + ':' + str(similarity[idx]))
                # 读取pcd文件
                pcd = o3d.io.read_point_cloud(cname)
                # 可视化点云
                # o3d.visualization.draw_geometries_with_editing([pcd], window_name="Open3D", width=800, height=600)
            else:
                if similarity[idx] > 0.9:
                    if similar_slice=='':
                        similar_slice+=name_id_dict[cname.split('/')[-1].split('.')[0]]
                    else:
                        similar_slice+=(','+name_id_dict[cname.split('/')[-1].split('.')[0]])
                    print(cname + ' sim pcd! ' + str(idx) + ':' + str(similarity[idx]))
        similar_dict[query_name.split('/')[-1].split('.')[0]]=similar_slice

    for SNaht in SNahts:
        src_name=SNaht.attrib.get('Name')
        dict={}
        for key, value in SNaht.attrib.items():
            if key == 'ID':
                dict[key] = value
                dict['Naht_IDs'] = similar_dict[src_name]
            elif key == 'Naht_IDs':
                continue
            else:
                dict[key] = value
        SNaht.attrib.clear()
        for key, value in dict.items():
            SNaht.set(key, value)
        tree.write(xml_path)

if __name__ == "__main__":
    pointnn()
