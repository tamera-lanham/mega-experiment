from dataclasses import dataclass
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from mega_experiment.datasets import CIFAR10_label_noise

from mega_experiment.job_runner.training_job import (
    TrainingJob,
    create_job_hyperparams_class,
    HyperparamsBase,
)
from mega_experiment.models.resnet import make_resnet18k


@dataclass
class Hyperparams(HyperparamsBase):
    n_epochs: int = 5
    label_noise: float = 0.2
    batch_size: int = 128
    resnet_width: int = 64
    lr: float = 1e-3


JobHyperparams = create_job_hyperparams_class(Hyperparams)  # Please keep this line in your training job definition!


class CifarDoubleDescentJob(TrainingJob):
    def model(self, hyperparams: Hyperparams) -> t.nn.Module:
        return make_resnet18k(k=hyperparams.resnet_width)

    def data_loaders(self, hyperparams: Hyperparams) -> tuple[DataLoader, Optional[DataLoader]]:

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        data_path = "./.data/CIFAR10/"
        train_set_ln = CIFAR10_label_noise(
            label_noise=hyperparams.label_noise,
            root=data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
        train_loader_ln = DataLoader(train_set_ln, batch_size=hyperparams.batch_size, shuffle=True, num_workers=1)

        val_set = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        val_loader = DataLoader(val_set, batch_size=hyperparams.batch_size, shuffle=False, num_workers=1)

        return train_loader_ln, val_loader

    def optimizer(self, hyperparams: Hyperparams, model: t.nn.Module) -> t.optim.Optimizer:
        return t.optim.SGD(model.parameters(), lr=hyperparams.lr)

    def loss_function(self, hyperparams: Hyperparams):
        return F.cross_entropy

    def stop_condition(self, hyperparams: Hyperparams, epoch: int, all_metrics: list[dict[str, object]]) -> bool:
        if hyperparams.n_epochs == epoch:
            return True
