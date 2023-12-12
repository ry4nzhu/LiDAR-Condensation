import os
import sys
sys.path.append('./')
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from util import *
    
def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    assert data is not None
    return data

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_num_object(keys, td):
    obj_nums = []
    for k in keys:
        obj_nums.append(td[k][-1])
    return obj_nums


def get_num_pedestrains(keys, data_path='/y/datasets/KITTI/training/'):
    ped_nums = []
    for frame in keys:
        lidar_path = os.path.join(data_path, 'velodyne')
        label_path = os.path.join(data_path, 'label_2')
        calib_path = os.path.join(data_path, 'calib')

        label_file = os.path.join(label_path, '%06d.txt'%frame)
        calib_file = os.path.join(calib_path, '%06d.txt'%frame)

        # get calibration
        calib = load_kitti_calib(calib_file)
        boxes_velo, objs_type = read_objs2velo(label_file, calib['Tr_velo2cam'])
        # print(objs_type)
        peds = [i for i in objs_type if i==3] # 3 is pedestrian
        ped_nums.append(len(peds)) 
    return ped_nums
        

if __name__ == "__main__":
    td_loss_cls = load_pickle('./sorted_loss_cls_td.pickle')
    td_loss_loc = load_pickle('./sorted_loss_loc_td.pickle')
    td_loss = load_pickle('./sorted_loss_td.pickle')

    cls_keys = list(td_loss_cls.keys())
    loc_keys = list(td_loss_loc.keys())
    loss_keys = list(td_loss.keys())

    num_frames = len(loss_keys)
    # selected = int(0.1 * num_frames)
    # cls_keys = cls_keys[-selected:]
    # loc_keys = loc_keys[-selected:]
    # loss_keys = loss_keys[-selected:]

    # print(len(intersection(cls_keys, loss_keys)), len(intersection(loc_keys, loss_keys)),
    #         len(intersection(cls_keys, loc_keys)), selected)


    ## Object number distribution (pedestrain) ? 
    obj_num_dist = []
    for i in range(0, num_frames, int(num_frames / 10)):
        selected = loss_keys[i:i + int(num_frames / 10)]
        print(i, i + int(num_frames / 10))
        num_objs = get_num_object(selected, td_loss)
        num_peds = get_num_pedestrains(selected)
        # print(num_objs)
        obj_num_dist.append(np.mean(num_peds))
    
    print(obj_num_dist)