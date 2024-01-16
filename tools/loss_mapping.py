import numpy as np


class LinearLossMap():
    def __init__(self, max_downsample_ratio, loss_min=0.0, loss_max=0.0):
        self.max_downsample_ratio = max_downsample_ratio
        self.loss_min = loss_min
        self.loss_max = loss_max
        

    
    def get_downsample_percentage(self, loss):
        # simple linear mapping model
        ratio = (self.loss_max - loss) / self.loss_max * self.max_downsample_ratio
        return ratio
    

class LinearLossMapRanking():
    def __init__(self, max_downsample_ratio, max_rank, min_rank=0):
        self.max_downsample_ratio = max_downsample_ratio
        self.min_rank = min_rank
        self.max_rank = max_rank

        self.downsample_ratio = np.linspace(max_downsample_ratio, 0, max_rank)

    def get_downsample_percentage_ranking(self, loss_rank):
        return self.downsample_ratio[loss_rank - self.min_rank]


    def get_downsample_percentage_ranking_rev(self, loss_rank):
        return self.downsample_ratio[self.max_rank - loss_rank - 1]
