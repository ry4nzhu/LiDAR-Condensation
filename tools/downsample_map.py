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
        random_indices = np.random.randint(pcd.shape[0], size=int(pcd.shape[0] * (1 - downsample_ratio)))
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
        random_indices = np.random.randint(pcd.shape[0], size=int(pcd.shape[0] * (1 - downsample_ratio)))
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
        train_file = os.path.join(lidar_path, "%06d.bin"%id)
        label_file = os.path.join(label_path, '%06d.txt'%id)
        calib_file = os.path.join(calib_path, '%06d.txt'%id)
        calib = load_kitti_calib(calib_file)
        boxes_velo, objs_type = read_objs2velo(label_file, calib['Tr_velo2cam'])
        pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)
        print(idx, lidar_path, pcd.shape)
        obj_points, non_obj_points = seperate_obj_points(pcd, boxes_velo, objs_type)

        self_loss = td_logs[id][0]
        # downsample_ratio = loss_map.get_downsample_percentage_ranking(idx)
        downsample_ratio = loss_map.get_downsample_percentage_ranking_rev(idx)
        print(self_loss, downsample_ratio)

        ## remove non-pedestrian, cyclist points
        downsampled_shape = int(pcd.shape[0] * (1 - downsample_ratio))

        if obj_points is not None:
            remaining_target_point_num = downsampled_shape - obj_points.shape[0]
        else:
            remaining_target_point_num = downsampled_shape
        if remaining_target_point_num > 0:
            random_indices = np.random.randint(non_obj_points.shape[0], size=remaining_target_point_num)
            points_remain = non_obj_points[random_indices]
        else:
            random_indices = np.random.randint(obj_points.shape[0], size=downsampled_shape)
            obj_points = obj_points[random_indices]
        if points_remain is not None:
            obj_points = np.vstack([obj_points, points_remain])
        # random_indices = np.random.randint(pcd.shape[0], size=downsampled_shape)
        pcd_new = np.vstack(obj_points)
        print(pcd.shape, pcd_new.shape)
        original_size.append(pcd.shape[0])
        downsampled_size.append(pcd_new.shape[0])
        save_to_bin(train_file, pcd_new)

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

    print(min_loss, max_loss)

    # random_downsample_loss(
    #     "/y/datasets/kitti-downsample/KITTI-0.8-loss-rand/",
    #     linearLossMap, td_loss, training_sample_ids
    # )

    # random_downsample_loss_rank(
    #     "/y/datasets/kitti-downsample/KITTI-0.8-loss-rand-rank/",
    #     linearLossMapRanking, td_loss, training_sample_ids
    # )

    object_downsample_loss_rank(
        "/y/datasets/kitti-downsample/KITTI-0.8-loss-obj-rank-rev/",
        linearLossMapRanking, td_loss, training_sample_ids
    )