import os
import sys
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_loss_change(frame_id, frame_td_list, loss_index=1, save_path=None):
    epochs = frame_td_list[:, 0]
    losses = frame_td_list[:, loss_index]
    object_num = frame_td_list[0, 2]
    
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(epochs, losses)
    
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Frame %d, Obj num %d'%(frame_id, object_num))
    
    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    
    
def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    assert data is not None
    return data


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def preprocess_td_data(td_data):
    collections = defaultdict(list)
    for td_tuple in td_data:
        frame_id = int(td_tuple['index'])
        gt_box_number = td_tuple['gt_bboxes'].cpu().numpy()[0].shape[0]
        loss = td_tuple['loss']
        epoch = td_tuple['epoch']
        loss_cls = td_tuple['loss_cls']
        loss_loc = td_tuple['loss_loc']
        loss_dir = td_tuple['loss_dir']
        print(frame_id, epoch, loss, gt_box_number) # , td_tuple['gt_bboxes'].cpu().numpy()
        collections[frame_id].append(np.array([epoch, loss, gt_box_number, loss_cls, loss_loc, loss_dir]))
    return collections        


def get_avg_std_loss(frame_td_list, from_epoch=0, to_epoch=20):
    avg_loss_list =  [np.average(frame_td_list[:, 1][from_epoch:to_epoch]), np.average(frame_td_list[:, 3][from_epoch:to_epoch]), 
                      np.average(frame_td_list[:, 4][from_epoch:to_epoch]), np.average(frame_td_list[:, 5][from_epoch:to_epoch]),
                      int(np.average(frame_td_list[:, 2][from_epoch:to_epoch]))]
    std_loss_list =  [np.std(frame_td_list[:, 1][from_epoch:to_epoch]), np.std(frame_td_list[:, 3][from_epoch:to_epoch]), 
                      np.std(frame_td_list[:, 4][from_epoch:to_epoch]), np.std(frame_td_list[:, 5][from_epoch:to_epoch])]
    return np.array(avg_loss_list), np.array(std_loss_list)



def analyze_loss_dist(td_collection):
    avg_loss_dict = {}
    for frame_id, frame_td in td_collection.items():
        avg_loss, avg_std = get_avg_std_loss(np.array(frame_td))
        avg_loss_dict[frame_id] = avg_loss
    
    result_dict = dict(sorted(avg_loss_dict.items(), key=lambda item: item[1][0]))

    dump_pickle(result_dict, 'sorted_loss_td.pickle')
    result_dict = dict(sorted(avg_loss_dict.items(), key=lambda item: item[1][1]))
    dump_pickle(result_dict, 'sorted_loss_cls_td.pickle')
    result_dict = dict(sorted(avg_loss_dict.items(), key=lambda item: item[1][2]))
    dump_pickle(result_dict, 'sorted_loss_loc_td.pickle')

if __name__ == "__main__":
    td_data = load_pickle('./td.pickle')
    # print(type(td_data['training_dynamics'][0]['loss_cls']))
    
    training_dynamics = td_data['training_dynamics']
    print(len(training_dynamics))
    
    td_colection = preprocess_td_data(training_dynamics)

    analyze_loss_dist(td_colection)
    
    # print(td_colection.keys())
    
    # for frame_id, frame_td in td_colection.items():
    #     plot_loss_change(frame_id, np.array(frame_td), loss_index=1, save_path='./visual_utils/result/loss/%d.png'%frame_id)
    #     plot_loss_change(frame_id, np.array(frame_td), loss_index=3, save_path='./visual_utils/result/loss_cls/%d.png'%frame_id)
    #     plot_loss_change(frame_id, np.array(frame_td), loss_index=4, save_path='./visual_utils/result/loss_loc/%d.png'%frame_id)
    #     plot_loss_change(frame_id, np.array(frame_td), loss_index=5, save_path='./visual_utils/result/loss_dir/%d.png'%frame_id)