import numpy as np
from typing import Iterable
import torch as t
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class IdentityDataset(Dataset):
    def __init__(self, shape: Iterable[int], n_batches: int):
        self.data = [t.randn(shape) for _ in range(n_batches)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        X = self.data[idx]
        return (X, X)


class CIFAR10_label_noise(CIFAR10):
    def __init__(self, label_noise=0, **kwargs):
        super().__init__(**kwargs)
        self.label_noise = label_noise

        self.noise_idxs = np.where(np.random.rand(super().__len__()) < self.label_noise)[0]

        self.noise_vals = np.empty(len(self.noise_idxs), dtype=int)

        self.num_classes = 10

        for i, idx in enumerate(self.noise_idxs):
            new_label = np.random.randint(self.num_classes - 1)
            if new_label == (super().__getitem__(int(idx))[1]):
                # if a randomized thing is assigned the same thing, then like don't do that
                new_label = self.num_classes - 1

            self.noise_vals[i] = new_label

    def __getitem__(self, idx):
        if idx in self.noise_idxs:
            new_label = self.noise_vals[self.noise_idxs == idx][0]
            return (super().__getitem__(idx)[0], new_label)
        else:
            return super().__getitem__(idx)
