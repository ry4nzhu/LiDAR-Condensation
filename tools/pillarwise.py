import numpy as np
# from spconv.utils import VoxelGenerator

class Pillars:
    def __init__(self, pointcloud, point_cloud_range, voxel_size):
        self.range = point_cloud_range
        self.voxel_size = voxel_size
        self.pcd = pointcloud
        # self.voxel_generator = None
    

    def create_pillars(self):
        # get indices of points to pillar
        # map pillar to indices
        cropped_pcd = self.crop_pointcloud_by_range()
        print("cropped shape", cropped_pcd.shape)
        

    def crop_pointcloud_by_range(self):
        mask = (self.pcd[:, 0] >= self.range[0]) & (self.pcd[:, 0] <= self.range[3]) \
            & (self.pcd[:, 1] >= self.range[1]) & (self.pcd[:, 1] <= self.range[4]) \
            & (self.pcd[:, 2] >= self.range[2]) & (self.pcd[:, 2] <= self.range[5])
        return self.pcd[mask]