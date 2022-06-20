from dataclasses import dataclass, fields, make_dataclass
from datetime import datetime
import subprocess
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union, TypeVar


# All methods on this class are overwriteable by subclasses! In fact, subclasses are
# encouraged to overwrite these methods to suit their needs.
class TrainingJob:
    def __init__(self, settings: Optional["TrainingJobSettings"] = None):
        self.settings = TrainingJobSettings() if settings is None else settings

    def model(self, hyperparams: "HyperparamsBase") -> t.nn.Module:
        raise NotImplementedError("Subclasses should implement this method.")

    def data_loaders(self, hyperparams: "HyperparamsBase") -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
        raise NotImplementedError("Subclasses should implement this method.")

    def optimizer(self, hyperparams: "HyperparamsBase", model: t.nn.Module) -> t.optim.Optimizer:
        return t.optim.SGD(model.parameters())

    def get_loss_and_y_pred(
        self, model: t.nn.Module, batch: tuple[t.Tensor, t.Tensor], loss_fn: callable
    ) -> tuple[t.Tensor, t.Tensor]:
        X, y = batch
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        return loss, y_pred

    def train_step(
        self,
        hyperparams: "HyperparamsBase",
        model: t.nn.Module,
        batch: tuple[t.Tensor, t.Tensor],
    ) -> tuple[t.Tensor, t.Tensor]:
        return self.get_loss_and_y_pred(model, batch, F.cross_entropy)

    def validation_step(
        self,
        hyperparams: "HyperparamsBase",
        model: t.nn.Module,
        batch: tuple[t.Tensor, t.Tensor],
    ) -> tuple[t.Tensor, t.Tensor]:
        return self.train_step(hyperparams, model, batch)


@dataclass
class TrainingJobSettings:
    job_type: str = "TrainingJob"
    job_description: str = ""
    n_instance_repeats: int = 1
    save_metrics_every: tuple[int, Optional[int]] = (1, None)  # n_epochs, n_batches
    save_parameters_every: tuple[int, Optional[int]] = (1, None)  # n_epochs, n_batches
    n_processes: int = 4
    output_dirname: str = ""
    most_recent_commit_hash: str = ""

    def __post_init__(self):
        self.most_recent_commit_hash = self._current_commit_hash()
        if not self.output_dirname:
            self.output_dirname = self._output_dirname()

    def _current_commit_hash(self) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    def _output_dirname(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M")
        return self.job_type + "_" + timestamp


@dataclass
class HyperparamsBase:
    instance_id: Optional[int] = None


class HyperparamOptions(list[TypeVar("T")]):
    def __init__(self, *args, **kwargs):
        super().__init__(args[0])


def create_job_hyperparams_class(hyperparams_class: type):
    """
    Create a new dataclass based on an original, where the new class is the same as the original
    but with a HyperparamOptions type hint option added to each of the attibutes.
    """
    enhance_type = lambda type_: Union[type_, HyperparamOptions[type_]]
    job_hyperparams_class_fields = [
        (field.name, enhance_type(field.type), field)
        for field in fields(hyperparams_class)
        if field.name != "instance_id"
    ]
    job_hyperparams_class = make_dataclass("JobHyperparams", job_hyperparams_class_fields)
    job_hyperparams_class.hyperparams_class = hyperparams_class
    return job_hyperparams_class
