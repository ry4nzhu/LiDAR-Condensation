import argparse
import numpy as np
import os
import sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)
import open3d as o3d

data_root = '/y/datasets/kitti-downsample/'
data_root = '/scratch/zmao_root/zmao0/ryanzhu/'
dataset_folder = 'KITTI-0.8/'
RANDOM_SEED = 1000
np.random.seed(RANDOM_SEED)

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--aug',
        action='store_true',
        help='Whether to visualize augmented datasets or original dataset.')
    args = parser.parse_args()
    return args



def random_downsample(ratio=0.8):
    train_file_path = os.path.join(data_root, dataset_folder, 'training/velodyne')
    with open(os.path.join(data_root, dataset_folder, 'ImageSets/train.txt'), 'r') as f:
        train_ids = f.readlines()
        for train_id in train_ids:
            # print(train_id, type(train_id))
            train_file = os.path.join(train_file_path, "%06d.bin" % int(train_id))
            pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)

            random_indices = np.random.choice(pcd.shape[0], size=int(ratio*pcd.shape[0]), replace=False)
            random_indices = np.sort(random_indices) # maintain the same order as original point cloud
            print(random_indices.shape, np.unique(random_indices).shape)
            pcd_new = np.copy(pcd)
            pcd_new = pcd_new[random_indices]

            print("before: ", pcd.shape[0], " after", pcd_new.shape[0], pcd.dtype, pcd_new.dtype)
            with open(train_file, 'wb') as f:
                print("save to ", train_file, pcd_new.dtype)
                pcd_new.tofile(f)


if __name__ == "__main__":
    
    # random_non_object(ratio=0.8)
    random_downsample(ratio=0.8)