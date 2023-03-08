import torch
import numpy as np


class SingleClassSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, target_class):
        self.dataset = dataset
        self.indices = np.where(np.array(dataset.targets) == target_class)[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.target_class = target_class

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)