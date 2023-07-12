import os.path
import xml.etree.ElementTree as ET
import open3d as o3d
import numpy as np
# import json

# def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
#     if element:  # 判断element是否有子元素
#         if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
#             element.text = newline + indent * (level + 1)
#         else:
#             element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
#             # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
#             # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
#     temp = list(element)  # 将element转成list
#
#     for subelement in temp:
#         if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
#             subelement.tail = newline + indent * (level + 1)
#         else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
#             subelement.tail = newline + indent * level
#         pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作



# def run_one_pointcloud(src, target, net):
#     if len(src.shape) == 2 and len(target.shape) == 2:  ##  (N,3)
#
#         # print("src/target shape:", src.shape, target.shape)
#
#         src = np.expand_dims(src[:, :3], axis=0)
#         src = np.transpose(src, [0, 2, 1])  ##  (1, 3, 1024)
#         target = np.expand_dims(target[:, :3], axis=0)
#         target = np.transpose(target, [0, 2, 1])  ##  (1, 3, 1024)
#
#     net.eval()
#
#     src = torch.from_numpy(src).cuda()
#     target = torch.from_numpy(target).cuda()
#
#     rotation_ab_pred, translation_ab_pred, \
#         rotation_ba_pred, translation_ba_pred = net(src, target)
#     target_pred = transform_point_cloud(src, rotation_ab_pred,
#                                         translation_ab_pred)
#
#     src_pred = transform_point_cloud(target, rotation_ba_pred,
#                                      translation_ba_pred)
#
#     mse_s_t = torch.mean((target_pred - target) ** 2, dim=[0, 1, 2]).item()
#     mae_s_t = torch.mean(torch.abs(target_pred - target), dim=[0, 1, 2]).item()
#
#     mse_t_s = torch.mean((src_pred - src) ** 2, dim=[0, 1, 2]).item()
#     mae_t_s = torch.mean(torch.abs(src_pred - src), dim=[0, 1, 2]).item()
#     # put on cpu and turn into numpy
#     src_pred = src_pred.detach().cpu().numpy()
#     src_pred = np.transpose(src_pred[0], [1, 0])
#
#     target_pred = target_pred.detach().cpu().numpy()
#     target_pred = np.transpose(target_pred[0], [1, 0])
#
#     rotation_ab_pred = rotation_ab_pred.detach().cpu().numpy()
#     translation_ab_pred = translation_ab_pred.detach().cpu().numpy()
#
#     rotation_ba_pred = rotation_ba_pred.detach().cpu().numpy()
#     translation_ba_pred = translation_ba_pred.detach().cpu().numpy()
#
#     return src_pred, target_pred, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred,mse_s_t,mae_s_t,mse_t_s,mae_t_s

def matching(xml_file):
    file_path='Reisch_pc_seam'
    xml_path=os.path.join('xml',xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    SNahts = root.findall("SNaht")
    for SNaht_src in SNahts:
        dict={}
        dict_rmse= {}
        similar_str=''
        src_name=SNaht_src.attrib.get('Name')
        print(src_name)
        src_path=file_path + '/' + src_name + '.pcd'
        if os.path.exists(src_path)!=True:
            continue
        pcd1 = o3d.io.read_point_cloud(src_path)
        point1 = np.array(pcd1.points).astype('float32')
        centroid1 = np.mean(point1, axis=0)
        m1 = np.max(np.sqrt(np.sum(point1 ** 2, axis=1)))
        point1 = (point1 - centroid1) / m1
        src=point1
        for SNaht_tgt in SNahts:
            tgt_name=SNaht_tgt.attrib.get('Name')
            print(tgt_name)
            if src_name==tgt_name:
                continue
            tgt_path=file_path + '/' + tgt_name + '.pcd'
            if os.path.exists(tgt_path)!=True:
                continue
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

            if mean_distance_s_t > 0.03 or rmse_s_t > 0.03 or correspondence_s_t < 2000 or mean_distance_t_s > 0.03 or rmse_t_s > 0.03 or correspondence_t_s < 2000:
                continue
            if similar_str=='':
                similar_str+=SNaht_tgt.attrib.get('ID')
            else:
                similar_str += (','+SNaht_tgt.attrib.get('ID'))
            dict_rmse[tgt_name] = rmse_s_t
        # with open('metric/' + src_name, 'w') as f:
        #     f.write(json.dumps(dict_rmse, indent=1))
        for key,value in SNaht_src.attrib.items():
            if key=='ID':
                dict[key]=value
                dict['Naht_IDs']=similar_str
            else:
                dict[key]=value
        SNaht_src.attrib.clear()
        for key,value in dict.items():
            SNaht_src.set(key,value)
        print('got one')
        tree.write(xml_path)
    return



    # for i in slice_list:
    #     for j in slice_list:
    #         if i!=j:
    #             pcd2 = o3d.io.read_point_cloud(file_path + '/' + j + '.pcd')
    #             point2 = np.array(pcd2.points).astype('float32')
    #             centroid2 = np.mean(point2, axis=0)
    #             point2 = point2 - centroid2
    #             m2 = np.max(np.sqrt(np.sum(point2 ** 2, axis=1)))
    #             point2 = point2 / m2
    #             target = point2
    #
    #             src_cloud = o3d.geometry.PointCloud()
    #             src_cloud.points = o3d.utility.Vector3dVector(src)
    #             tgt_cloud = o3d.geometry.PointCloud()
    #             tgt_cloud.points = o3d.utility.Vector3dVector(target)
    #             icp_s_t = o3d.pipelines.registration.registration_icp(source=src_cloud, target=tgt_cloud,
    #                                                                   max_correspondence_distance=0.2,
    #                                                                   estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #             icp_t_s = o3d.pipelines.registration.registration_icp(source=tgt_cloud, target=src_cloud,
    #                                                                   max_correspondence_distance=0.2,
    #                                                                   estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    #
    #             mean_distance_s_t = np.mean(src_cloud.compute_point_cloud_distance(tgt_cloud))
    #             fitness_s_t = icp_s_t.fitness
    #             rmse_s_t = icp_s_t.inlier_rmse
    #             correspondence_s_t = len(np.asarray(icp_s_t.correspondence_set))
    #
    #             mean_distance_t_s = np.mean(tgt_cloud.compute_point_cloud_distance(src_cloud))
    #             fitness_t_s = icp_t_s.fitness
    #             rmse_t_s = icp_t_s.inlier_rmse
    #             correspondence_t_s = len(np.asarray(icp_t_s.correspondence_set))
    #
    #             if mean_distance_s_t > 0.03 or rmse_s_t > 0.03 or correspondence_s_t < 2000 or mean_distance_t_s > 0.03 or rmse_t_s > 0.03 or correspondence_t_s < 2000:
    #                 continue
    #             similar_list.append(j)
    #
    # return dict3

if __name__ == "__main__":
    matching('Reisch.xml')