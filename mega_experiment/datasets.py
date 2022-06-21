import torch as t
from torch.utils.data import Dataset

class IdentityDataset(Dataset):
    def __init__(self, shape, n_batches):
        self.data = [t.randn(shape) for _ in range(n_batches)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        return (X, X)