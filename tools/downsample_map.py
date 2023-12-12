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
        # save_to_bin(train_file, pcd_new)

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


    random_downsample_loss_rank(
        "/y/datasets/kitti-downsample/KITTI-0.8-loss-rand-rank/",
        linearLossMapRanking, td_loss, training_sample_ids
    )