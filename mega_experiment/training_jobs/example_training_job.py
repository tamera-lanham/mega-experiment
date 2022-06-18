from dataclasses import dataclass
from inspect import Parameter
from pyexpat import model
from mega_experiment.training_jobs.training_job import TrainingJob, create_job_hyperparams_class, HyperparamsBase
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Iterable, Union

@dataclass
class Hyperparams(HyperparamsBase):
    n_inputs: int = 16
    batch_size: int = 32
    n_batches: int = 500
    epochs: int = 5
    lr: float = 1e-3

JobHyperparams = create_job_hyperparams_class(Hyperparams)


class IdentityDataset(Dataset):
    def __init__(self, shape, n_batches):
        self.data = [t.randn(shape) for _ in range(n_batches)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        return (X, X)

class ExampleTrainingJob(TrainingJob):

    def model(self, hyperparams: Hyperparams) -> t.nn.Module:
        input_size = hyperparams.n_inputs
        hidden_size = hyperparams.n_inputs * 4
        output_size = hyperparams.n_inputs

        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        return model

    def data_loaders(self, hyperparams: Hyperparams) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
        
        train_loader = DataLoader(
            IdentityDataset(
                (hyperparams.batch_size, hyperparams.n_inputs), hyperparams.n_batches
            )
        )
        val_loader = DataLoader(
            IdentityDataset(
                (hyperparams.batch_size, hyperparams.n_inputs), hyperparams.n_batches // 5
            )
        )
        return train_loader, val_loader

    def optimizer(self, hyperparams: Hyperparams, model: t.nn.Module) -> t.optim.Optimizer:
        return t.optim.SGD(model.parameters(), lr=hyperparams.lr)

    def train_step(self, hyperparams: Hyperparams, model: t.nn.Module, batch: tuple[t.Tensor, t.Tensor]) -> tuple[t.Tensor, t.Tensor]:
        return self.get_loss_and_y_pred(model, batch, F.mse_loss)




