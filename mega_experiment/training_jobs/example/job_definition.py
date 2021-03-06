from dataclasses import dataclass
from mega_experiment.job_runner.training_job import (
    TrainingJob,
    create_job_hyperparams_class,
    HyperparamsBase,
)
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from mega_experiment.datasets import IdentityDataset


@dataclass
class Hyperparams(HyperparamsBase):
    n_inputs: int = 16
    batch_size: int = 32
    n_batches: int = 500
    n_epochs: int = 5
    lr: float = 1e-3


JobHyperparams = create_job_hyperparams_class(Hyperparams)  # Please keep this line in your training job definition!


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

    def data_loaders(self, hyperparams: Hyperparams) -> tuple[DataLoader, Optional[DataLoader]]:

        train_loader = DataLoader(
            IdentityDataset((hyperparams.batch_size, hyperparams.n_inputs), hyperparams.n_batches)
        )
        val_loader = DataLoader(
            IdentityDataset(
                (hyperparams.batch_size, hyperparams.n_inputs),
                hyperparams.n_batches // 5,
            )
        )
        return train_loader, val_loader

    def optimizer(self, hyperparams: Hyperparams, model: t.nn.Module) -> t.optim.Optimizer:
        return t.optim.SGD(model.parameters(), lr=hyperparams.lr)

    def loss_function(self, hyperparams: Hyperparams):
        return F.mse_loss

    def stop_condition(self, hyperparams: Hyperparams, epoch: int, all_metrics: list[dict[str, object]]) -> bool:
        if hyperparams.n_epochs == epoch:
            return True
