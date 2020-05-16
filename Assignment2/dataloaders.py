import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, data):

        self.features = data[0]
        self.labels = data[1]
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        return feature, label