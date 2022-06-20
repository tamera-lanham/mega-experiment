# In charge of:
# - Saving hyperparams
# - Tracing and saving the model
# - Training the model
# - Saving the params, metrics, val outputs and such while training
# - Saving to GCP
# - Reporting progress in the console??

from dataclasses import asdict
from itertools import count
from typing import Optional
from mega_experiment.training_jobs.training_job import TrainingJob, HyperparamsBase
import json
import os
from pathlib import Path
import torch as t


class InstanceRunner:
    def __init__(
        self,
        training_job: TrainingJob,
        hyperparams: HyperparamsBase,
        job_output_dir: Path,
        device: t.device,
    ):
        self.training_job = training_job
        self.hyperparams = hyperparams
        self.job_output_dir = job_output_dir
        self.output_dir = job_output_dir / "instances" / str(self.hyperparams.instance_id)
        self.device = device

    def run(self):
        model = self.pre_training_setup()
        self.train(model)

    def pre_training_setup(self) -> t.nn.Module:
        # Make the output dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Save hyperparams
        with open(self.job_output_dir / "instance_metadata.jsonl", "a") as f:
            f.write(json.dumps(asdict(self.hyperparams)) + "\n")

        # Trace and save model
        model = self.training_job.model(self.hyperparams)
        scripted_model = t.jit.script(model)
        t.jit.save(scripted_model, self.output_dir / "model_torchscript.pt")
        return model

    def train(self, model: t.nn.Module):

        # Set up callbacks
        metrics = SaveMetrics(1, None, self.output_dir)
        parameters = SaveParameters(1, 100, self.output_dir)
        callbacks = Callbacks(metrics, parameters)

        # Set up data loaders, model, and optimizer
        train_loader, val_loader = self.training_job.data_loaders(self.hyperparams)
        val_loader = [] if val_loader is None else val_loader
        model.to(self.device)
        optimizer = self.training_job.optimizer(self.hyperparams, model)

        for epoch in count():  # Counts up from 0 indefinitely

            # Check traning stop condition
            if self.training_job.stop_condition(self.hyperparams, epoch, metrics.stored_metrics):
                return

            # Training phase
            for i, (X, y) in enumerate(train_loader):
                X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss, y_pred = self.training_job.train_step(self.hyperparams, model, (X, y))
                loss.backward()
                optimizer.step()

                callbacks.run_callbacks(model, epoch, i, X, y, y_pred, loss, False)

            # Validation phase
            for i, (X, y) in enumerate(val_loader):
                X.to(self.device), y.to(self.device)
                with t.no_grad():
                    # TODO: don't throw away all the val data maybe??
                    loss, y_pred = self.training_job.validation_step(self.hyperparams, model, (X, y))

            callbacks.run_callbacks(model, epoch, None, X, y, y_pred, loss, True)


class Callbacks:
    def __init__(self, *callbacks: "Callback"):
        self.callbacks = callbacks

    def run_callbacks(self, model, epoch, batch_index, X, y, y_pred, loss, validation=False):
        for callback in self.callbacks:
            if callback.run_condition(epoch, batch_index, validation):
                callback.run(model, epoch, batch_index, X, y, y_pred, loss, validation)


class Callback:
    def __init__(self, every_n_epochs: int, every_n_batches: Optional[int], output_dir: Optional[Path] = None):
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches
        self.output_dir = output_dir

    def run_condition(self, epoch, batch_index, validation=False):
        # TODO: Handle case where self.every_n_batches is None appropriately (likely requires SaveMetrics refactor)
        if self.every_n_batches and batch_index and batch_index % self.every_n_batches == 0:
            return True
        if self.every_n_epochs and not batch_index and epoch % self.every_n_epochs == 0:
            return True
        return False

    def run(
        self,
        model: t.nn.Module,
        epoch: int,
        batch_index: Optional[int],
        X: t.Tensor,
        y: t.Tensor,
        y_pred: t.Tensor,
        loss: t.Tensor,
        validation: bool = False,
    ) -> None:
        raise NotImplementedError("Callback run method not implemented")


class SaveParameters(Callback):
    def __init__(self, every_n_epochs: int, every_n_batches: Optional[int], output_dir: Optional[Path] = None):
        super().__init__(every_n_epochs, every_n_batches, output_dir)
        self.parameters_dir = self.output_dir / "parameter_checkpoints"
        os.makedirs(self.parameters_dir, exist_ok=True)

    def run(
        self,
        model: t.nn.Module,
        epoch: int,
        batch_index: Optional[int],
        X: t.Tensor,
        y: t.Tensor,
        y_pred: t.Tensor,
        loss: t.Tensor,
        validation: bool = False,
    ):
        state_name = f"epoch_{epoch}" + (f"_batch_{batch_index}" if batch_index is not None else "_end")
        t.save(
            model.state_dict(),
            self.parameters_dir / (state_name + ".pt"),
        )


class SaveMetrics(Callback):
    def __init__(self, every_n_epochs: int, every_n_batches: Optional[int], output_dir: Optional[Path] = None):
        super().__init__(every_n_epochs, every_n_batches, output_dir)
        self.stored_metrics = []

    def run(
        self,
        model: t.nn.Module,
        epoch: int,
        batch_index: Optional[int],
        X: t.Tensor,
        y: t.Tensor,
        y_pred: t.Tensor,
        loss: t.Tensor,
        validation: bool = False,
    ) -> None:
        raw_metrics = self.calculate_metrics(model, X, y, y_pred, loss)
        metrics = self.combine_metrics(raw_metrics, epoch, batch_index, validation)
        self.stored_metrics.append(metrics)
        self.write_metrics(metrics)

    def calculate_metrics(self, model, X, y, y_pred, loss) -> dict[str, float]:
        # User-defined metrics called from here (if any)
        return {"loss": loss.item()}

    def most_recent_metrics(self) -> dict:
        if not self.stored_metrics:
            return {}
        return self.stored_metrics[-1]

    def combine_metrics(self, metrics, epoch, batch_index, validation=False):
        prefix = "val_" if validation else "train_"
        metrics_prefixed = {prefix + key: value for key, value in metrics.items()}
        new_metrics = {
            **self.most_recent_metrics(),
            **metrics_prefixed,
            "epoch": epoch,
            "batch": "end" if batch_index is None else batch_index,
        }
        return new_metrics

    def write_metrics(self, combined_metrics):
        with open(self.output_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(combined_metrics) + "\n")
