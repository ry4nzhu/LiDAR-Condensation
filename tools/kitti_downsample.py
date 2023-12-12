import argparse
import numpy as np
import os
import sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)
import open3d as o3d
from mmdet3d.registry import DATASETS
from mmdet3d.utils import replace_ceph_backend
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar, mkdir_or_exist
from mmdet3d.visualization.vis_utils import to_depth_mode
from mmengine.visualization.utils import (check_type, color_val_matplotlib,
                                          tensor2ndarray)

data_root = '/y/datasets/kitti-downsample/'
dataset_folder = 'KITTI-0.1/'
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


def build_data_cfg(config_path, aug):
    """Build data config for loading visualization data."""

    cfg = Config.fromfile(config_path)
    # if cfg_options is not None:
    #     cfg.merge_from_dict(cfg_options)

    # extract inner dataset of `RepeatDataset` as
    # `cfg.train_dataloader.dataset` so we don't
    # need to worry about it later
    if cfg.train_dataloader.dataset['type'] == 'RepeatDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.train_dataloader.dataset['type'] == 'ConcatDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.datasets[0]
    if cfg.train_dataloader.dataset['type'] == 'CBGSDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.dataset

    train_data_cfg = cfg.train_dataloader.dataset

    if aug:
        show_pipeline = cfg.train_pipeline
    else:
        show_pipeline = cfg.test_pipeline
        for i in range(len(cfg.train_pipeline)):
            if cfg.train_pipeline[i]['type'] == 'LoadAnnotations3D':
                show_pipeline.insert(i, cfg.train_pipeline[i])
            # Collect data as well as labels
            if cfg.train_pipeline[i]['type'] == 'Pack3DDetInputs':
                if show_pipeline[-1]['type'] == 'Pack3DDetInputs':
                    show_pipeline[-1] = cfg.train_pipeline[i]
                else:
                    show_pipeline.append(cfg.train_pipeline[i])

    train_data_cfg['pipeline'] = show_pipeline

    return cfg


def random_downsample(ratio=0.8):
    train_file_path = os.path.join(data_root, dataset_folder, 'training/velodyne')
    for file in os.listdir(train_file_path):
        train_file = os.path.join(train_file_path, file)
        pcd = np.fromfile(train_file, dtype=np.float32).reshape(-1, 4)

        random_indices = np.random.randint(pcd.shape[0], size=int(ratio*pcd.shape[0]))
        print(random_indices.shape)
        pcd_new = np.copy(pcd)
        pcd_new = pcd_new[random_indices]

        print("before: ", pcd.shape[0], " after", pcd_new.shape[0])

        with open(train_file, 'wb') as f:
            print("save to ", train_file, pcd_new.dtype)
            pcd_new.tofile(f)


def remove_ground():
    pass


def random_non_object(ratio=0.8):
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)
    
    cfg = build_data_cfg(args.config, args.aug)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    dataset = DATASETS.build(
            cfg.train_dataloader.dataset,
            default_args=dict(filter_empty_gt=False))

    output_dir = args.output_dir
    # print(type(dataset))

    for i, item in enumerate(dataset):
        data_input = item['inputs']
        points = tensor2ndarray(data_input['points'])
        points_np = np.copy(points).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        data_sample = item['data_samples'].numpy()
        # print(data_sample.lidar_path, points.shape)

        target_point_num = int(ratio*points.shape[0])

        if 'gt_instances_3d' in data_sample:
            instances = data_sample.gt_instances_3d
            # print("gt instance", type(data_sample.gt_instances_3d))

            bboxes_3d = instances.bboxes_3d
            labels_3d = instances.labels_3d
            points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            bboxes_3d = tensor2ndarray(bboxes_3d_depth.tensor)
            # print(bboxes_3d_depth)
            all_obj_indices = []
            for i in range(len(bboxes_3d)):
                center = bboxes_3d[i, 0:3]
                dim = bboxes_3d[i, 3:6]
                yaw = np.zeros(3)
                rot_axis = 2
                yaw[rot_axis] = bboxes_3d[i, 6]
                rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
                center[rot_axis] += dim[rot_axis] / 2
                box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)
                indices = box3d.get_point_indices_within_bounding_box(pcd.points)
                all_obj_indices += indices
            
            # 004234.bin
            
            if len(all_obj_indices) > 0:
                obj_points = points_np[np.array(all_obj_indices)]
                # print(data_sample.lidar_path)
                filepath = data_sample.lidar_path.split('/')[-3]
                filepath_train = data_sample.lidar_path.split('/')[-2]
                filename = data_sample.lidar_path.split('/')[-1]
                save_path = os.path.join(output_dir, filepath, filepath_train, filename)
                # print(data_sample.lidar_path)
                # print(save_path)
                
                # random filling the object points with other points
                points_remain = None
                remaining_target_point_num = target_point_num - len(all_obj_indices)
                if remaining_target_point_num > 0:
                    points_remain = np.delete(points_np, np.array(all_obj_indices), axis=0)
                    # print("remaining point shape", points_remain.shape)
                    random_indices = np.random.randint(points_remain.shape[0], size=remaining_target_point_num)
                    points_remain = points_remain[random_indices]
                elif remaining_target_point_num < 0:
                    print("oversize, ", data_sample.lidar_path)
                    random_indices = np.random.randint(obj_points.shape[0], size=target_point_num)
                    obj_points = obj_points[random_indices]

                if points_remain is not None:
                    obj_points = np.vstack([obj_points, points_remain])
                    # print("remaining point shape", obj_points.shape)
            else:
                filepath = data_sample.lidar_path.split('/')[-3]
                filepath_train = data_sample.lidar_path.split('/')[-2]
                filename = data_sample.lidar_path.split('/')[-1]
                save_path = os.path.join(output_dir,filepath,filepath_train, filename)
                obj_points = np.copy(points_np)
                random_indices = np.random.randint(obj_points.shape[0], size=target_point_num)
                obj_points = obj_points[random_indices]


            print("point shape", points_np.shape, obj_points.shape)
            with open(save_path, 'wb') as f:
                # print("obj shape", obj_points.dtype)
                obj_points.tofile(f)



if __name__ == "__main__":
    
    random_non_object(ratio=0.8)
    # random_downsample(ratio=0.1)