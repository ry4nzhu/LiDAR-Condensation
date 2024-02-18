import argparse
import numpy as np
import os
import sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)
import open3d as o3d
from collections import defaultdict
from util import load_kitti_calib, read_objs2velo, seperate_obj_points, draw_open3d

data_root = '/y/datasets/kitti-downsample/'
data_root = '/nfs/turbo/coe-zmao/ryanzhu/'
dataset_folder = 'KITTI-0.8-rand-obj-ped+cyc/'
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
    stats = defaultdict(list)
    with open(os.path.join(data_root, dataset_folder, 'ImageSets/train.txt'), 'r') as f:
        train_ids = f.readlines()
        for train_id in train_ids:
            # print(train_id, type(train_id))
            train_file = os.path.join(train_file_path, "%06d.bin" % int(train_id))
            pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)
            # draw_open3d([pcd])
            random_indices_old = np.random.randint(pcd.shape[0], size=int(ratio*pcd.shape[0]))
            random_indices = np.random.choice(pcd.shape[0], size=int(ratio*pcd.shape[0]), replace=False)
            random_indices = np.sort(random_indices) # maintain the same order as original point cloud
            print(random_indices.shape, np.unique(random_indices).shape, np.unique(random_indices_old).shape)
            pcd_new = np.copy(pcd)
            pcd_new = pcd_new[random_indices]
            stats['original_size'].append(pcd.shape[0])
            stats['condensed_size'].append(pcd_new.shape[0])
            break
            # print("before: ", pcd.shape[0], " after", pcd_new.shape[0])
            # with open(train_file, 'wb') as f:
            #     print("save to ", train_file, pcd_new.dtype)
            #     pcd_new.tofile(f)

    print(np.mean(stats['original_size']), np.mean(stats['condensed_size']), np.mean(stats['condensed_size'])/np.mean(stats['original_size']))


def random_domsample_objects(ratio=0.8):
    lidar_path = os.path.join(data_root, dataset_folder, 'training/velodyne')
    label_path = os.path.join(data_root, dataset_folder, 'training/label_2')
    calib_path = os.path.join(data_root, dataset_folder, 'training/calib')
    with open(os.path.join(data_root, dataset_folder, 'ImageSets/train.txt'), 'r') as f:
        train_ids = f.readlines()
        for train_id in train_ids:
            # print(train_id, type(train_id))
            id = int(train_id)
            train_file = os.path.join(lidar_path, "%06d.bin"%id)
            label_file = os.path.join(label_path, '%06d.txt'%id)
            calib_file = os.path.join(calib_path, '%06d.txt'%id)
            calib = load_kitti_calib(calib_file)
            boxes_velo, objs_type = read_objs2velo(label_file, calib['Tr_velo2cam'])
            pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)
            lidar = np.copy(pcd)
            obj_points, non_obj_points, sorted_object_indices, sorted_non_object_indices, boxes_3d, boxes_old = seperate_obj_points(lidar, boxes_velo, objs_type)
            target_size = int(ratio*pcd.shape[0])
            remaining_size = target_size - sorted_object_indices.shape[0]
            obj_indices = np.copy(sorted_object_indices)
            if remaining_size > 0:
                random_indices = np.random.choice(sorted_non_object_indices.shape[0], size=remaining_size, replace=False)
                non_obj_indices = np.copy(sorted_non_object_indices[random_indices])
            else:
                print("only cutting object points", train_file)
                random_indices = np.random.choice(sorted_object_indices.shape[0], size=target_size, replace=False)
                obj_indices = np.copy(sorted_object_indices[random_indices])
            
            # draw_open3d([pcd[obj_indices], pcd[non_obj_indices]], gt_bboxes=boxes_old)
            if non_obj_indices is not None:
                obj_indices = np.concatenate((obj_indices, non_obj_indices), axis=None)
            rst_indices = np.sort(obj_indices).copy()

            pcd_new = np.copy(pcd)
            pcd_new = pcd_new[rst_indices]

            with open(train_file, 'wb') as f:
                print("save to ", train_file, pcd_new.dtype)
                pcd_new.tofile(f)



if __name__ == "__main__":
    
    random_downsample(ratio=0.8)
    random_domsample_objects(ratio=0.8)