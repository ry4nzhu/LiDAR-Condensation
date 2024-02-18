import numpy as np

filename = 'voxel_output.txt'
with open(filename, 'r') as f:
    lines = f.readlines()
    i = 0
    oversized_voxel_training, size_xovel_training = [], []
    oversized_voxel_testing, size_xovel_testing = [], []
    for line in lines:
        if line.startswith('oversized voxel'):
            if len(oversized_voxel_training) < 3712:
                oversized_voxel_training.append(int(line.split()[-1]))
                size_xovel_training.append(int(line.split()[-2]))
            else:
                size_xovel_testing.append(int(line.split()[-2]))
                oversized_voxel_testing.append(int(line.split()[-1]))

print(np.mean(size_xovel_training), np.std(size_xovel_training))
print(np.mean(size_xovel_testing), np.std(size_xovel_testing))
print(np.mean(oversized_voxel_training), np.std(oversized_voxel_training))
print(np.mean(oversized_voxel_testing), np.std(oversized_voxel_testing))