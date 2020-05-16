import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PointDataSet(Dataset):
    def __init__(self, data_fold):

        self.feature_matrix = data_fold.feature_matrix
        self.labels = data_fold.label_vector
    
    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, index):
        feature_vector = self.feature_matrix[index]
        label = self.labels[index]

        return feature_vector, label