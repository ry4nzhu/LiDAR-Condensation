import argparse
import numpy as np
import os
import sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)
from loss_mapping import LinearLossMap, LinearLossMapRanking
from util import *
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)

def random_downsample_loss(data_root, loss_map, td_logs, training_ids):
    train_file_path = os.path.join(data_root, 'training/velodyne')
    original_size, downsampled_size = [], []
    for id in training_ids[-300:]:
        train_file = os.path.join(train_file_path, "%06d.bin"%id)
        pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)

        self_loss = td_logs[id][0]
        downsample_ratio = loss_map.get_downsample_percentage(self_loss)
        print(self_loss, downsample_ratio)
        # random_indices = np.random.randint(pcd.shape[0], size=int(pcd.shape[0] * (1 - downsample_ratio)))
        random_indices = np.random.choice(pcd.shape[0], size=int(pcd.shape[0] * (1 - downsample_ratio)), replace=False)
        pcd_new = np.copy(pcd[random_indices])
        print(pcd.shape, pcd_new.shape)
        original_size.append(pcd.shape[0])
        downsampled_size.append(pcd_new.shape[0])
        save_to_bin(train_file, pcd_new)

    print("downsampeld ratio:", np.sum(downsampled_size)/np.sum(original_size))


def random_downsample_loss_rank(data_root, loss_map, td_logs, training_ids):
    train_file_path = os.path.join(data_root, 'training/velodyne')
    original_size, downsampled_size = [], []
    for idx, id in enumerate(training_ids):
        train_file = os.path.join(train_file_path, "%06d.bin"%id)
        pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)

        self_loss = td_logs[id][0]
        downsample_ratio = loss_map.get_downsample_percentage_ranking(idx)
        print(self_loss, downsample_ratio)
        # random_indices = np.random.randint(pcd.shape[0], size=int(pcd.shape[0] * (1 - downsample_ratio)))
        random_indices = np.random.choice(pcd.shape[0], size=int(pcd.shape[0] * (1 - downsample_ratio)), replace=False)
        random_indices = np.sort(random_indices)
        pcd_new = np.copy(pcd[random_indices])
        print(pcd.shape, pcd_new.shape)
        original_size.append(pcd.shape[0])
        downsampled_size.append(pcd_new.shape[0])
        save_to_bin(train_file, pcd_new)

    print("downsampeld ratio:", np.sum(downsampled_size)/np.sum(original_size))


def object_downsample_loss_rank(data_root, loss_map, td_logs, training_ids):
    lidar_path = os.path.join(data_root, 'training/velodyne')
    label_path = os.path.join(data_root, 'training/label_2')
    calib_path = os.path.join(data_root, 'training/calib')
    original_size, downsampled_size = [], []
    for idx, id in enumerate(training_ids):
        # print("sample id", id)
        train_file = os.path.join(lidar_path, "%06d.bin"%id)
        label_file = os.path.join(label_path, '%06d.txt'%id)
        calib_file = os.path.join(calib_path, '%06d.txt'%id)
        calib = load_kitti_calib(calib_file)
        boxes_velo, objs_type = read_objs2velo(label_file, calib['Tr_velo2cam'])
        pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)
        lidar = np.copy(pcd)
        print(idx, lidar_path, pcd.shape)
        obj_points, non_obj_points, sorted_object_indices, sorted_non_object_indices, boxes_3d, boxes_old = seperate_obj_points(lidar, boxes_velo, objs_type)
        # print(pcd.shape[0], sorted_object_indices.shape[0], sorted_non_object_indices.shape[0])
        # print("after chop", pcd.shape)

        self_loss = td_logs[id][0]
        downsample_ratio = loss_map.get_downsample_percentage_ranking(idx)
        # downsample_ratio = loss_map.get_downsample_percentage_ranking_rev(idx)
        print(self_loss, downsample_ratio)

        ## remove non-pedestrian/cyclist/car points
        downsampled_shape = int(pcd.shape[0] * (1 - downsample_ratio))

        # if obj_points is not None:
        remaining_target_point_num = downsampled_shape - obj_points.shape[0]
        # else:
        #     remaining_target_point_num = downsampled_shape
        non_obj_indices, obj_indices = None, sorted_object_indices
        if remaining_target_point_num > 0:
            random_indices = np.random.choice(non_obj_points.shape[0], size=remaining_target_point_num, replace=False)
            non_obj_indices = sorted_non_object_indices[random_indices]
            points_remain = non_obj_points[random_indices]
        else:
            random_indices = np.random.choice(obj_points.shape[0], size=downsampled_shape, replace=False)
            obj_indices = obj_indices[random_indices]
            obj_points = obj_points[random_indices]
        if non_obj_indices is not None:
            obj_indices = np.concatenate((obj_indices, non_obj_indices), axis=None)
        obj_indices = np.sort(obj_indices)

        # pcd_new = np.copy(pcd)
        # pcd_new = pcd_new[obj_indices]

        # print(obj_indices)
        # print(pcd.shape[0], np.unique(obj_indices).shape[0], downsampled_shape)
        # if idx == len(training_ids) - 2:
        # draw_open3d([pcd_new], gt_bboxes=boxes_3d, pred_bboxes=boxes_old) # , obj_points+0.01

        if points_remain is not None:
            obj_points = np.vstack([obj_points, points_remain])
        pcd_new = np.vstack(obj_points)
        # print(pcd.shape, pcd_new.shape)
        original_size.append(pcd.shape[0])
        downsampled_size.append(pcd_new.shape[0])
        save_to_bin(train_file, pcd_new)
        # break

    print("downsampeld ratio:", np.sum(downsampled_size)/np.sum(original_size))


if __name__ == "__main__":
    td_loss = load_pickle('./sorted_loss_td.pickle')
    training_sample_ids = list(td_loss.keys())
    min_loss_vec = td_loss[training_sample_ids[0]]
    max_loss_vec = td_loss[training_sample_ids[-1]]
    min_loss = min_loss_vec[0]
    max_loss = max_loss_vec[0]
    linearLossMap = LinearLossMap(0.25, min_loss, max_loss)
    linearLossMapRanking = LinearLossMapRanking(0.4, len(training_sample_ids))

    linearLossMapRanking = LinearLossMapRanking(0.2, len(training_sample_ids), min_downsample_ratio=0.2)
    # linearLossMapRanking = LinearLossMapRanking(0.25, len(training_sample_ids), min_downsample_ratio=0.15)

    print(min_loss, max_loss)

    # random_downsample_loss(
    #     "/y/datasets/kitti-downsample/KITTI-0.8-loss-rand/",
    #     linearLossMap, td_loss, training_sample_ids
    # )

    # random_downsample_loss_rank(
    #     "/scratch/zmao_root/zmao0/ryanzhu/KITTI-0.8-loss-rand-rank/",
    #     linearLossMapRanking, td_loss, training_sample_ids
    # )


    object_downsample_loss_rank(
        "/scratch/zmao_root/zmao0/ryanzhu/KITTI-0.8-loss-obj-rank/",
        linearLossMapRanking, td_loss, training_sample_ids
    )
    # /scratch/zmao_root/zmao0/ryanzhu/KITTI-0.8-loss-obj-rank/
    # "/nfs/turbo/coe-zmao/ryanzhu/KITTI-0.8-rand-loss-obj-0.15/"
    # /nfs/turbo/coe-zmao/ryanzhu/KITTI-0.8-rand-loss-ped/
    # "/nfs/turbo/coe-zmao/ryanzhu/KITTI-0.8-loss-obj-0.2-fixed/"