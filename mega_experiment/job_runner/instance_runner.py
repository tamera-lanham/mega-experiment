from dataclasses import asdict
from itertools import count
from turtle import back
from typing import Optional
from mega_experiment.job_runner.training_job import TrainingJob, HyperparamsBase
import json
import os
from pathlib import Path
import torch as t
from tqdm import tqdm

# In charge of:
# - Saving hyperparams
# - Tracing and saving the model
# - Training the model
# - Saving the params, metrics, and such while training
# - Saving to GCP?


class InstanceRunner:
    def __init__(
        self,
        training_job: TrainingJob,
        hyperparams: HyperparamsBase,
        job_output_dir: Path,
        device: Optional[t.device] = None,
        process_num: Optional[int] = None,
    ):
        self.training_job = training_job
        self.hyperparams = hyperparams
        self.job_output_dir = job_output_dir
        self.output_dir = job_output_dir / "instances" / str(self.hyperparams.instance_id)
        self.device = device
        self.process_num = process_num

    def run(self):
        model = self.pre_training_setup()
        self.train(model)

    def pre_training_setup(self) -> t.nn.Module:
        # Make the output dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Save hyperparams
        with open(self.job_output_dir / "instance_hyperparams.jsonl", "a") as f:
            f.write(json.dumps(asdict(self.hyperparams)) + "\n")

        # Trace and save model
        model = self.training_job.model(self.hyperparams)
        scripted_model = t.jit.script(model)
        t.jit.save(scripted_model, self.output_dir / "model_torchscript.pt")
        return model

    def train(self, model: t.nn.Module):

        # Set up callbacks
        metrics = SaveMetrics(self.output_dir)
        parameters = SaveParameters(self.training_job.settings.save_parameters_every_n_batches, self.output_dir)
        callbacks = Callbacks(metrics, parameters)

        # Set up data loaders, model, loss function, and optimizer
        train_loader, val_loader = self.training_job.data_loaders(self.hyperparams)
        val_loader = [] if val_loader is None else val_loader
        model.to(self.device)
        loss_fn = self.training_job.loss_function(self.hyperparams)
        optimizer = self.training_job.optimizer(self.hyperparams, model)

        progress_bar = tqdm(
            count(),  # Counts up from 0 indefinitely
            unit=" epochs",
            desc=f"Instance {self.hyperparams.instance_id} on device {self.device if self.device else 'cpu'} ",
            position=self.process_num,
            leave=False,
        )

        for epoch in progress_bar:
            # Check training stop condition
            if self.training_job.stop_condition(self.hyperparams, epoch, metrics.stored_metrics):
                return

            # Training phase
            for i, (X, y) in enumerate(train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_pred = model(X)
                train_loss = loss_fn(y_pred, y)
                if t.cuda.is_available: gpu_mem_usage = t.cuda.memory_allocated(self.device) 
                train_loss.backward()
                optimizer.step()

                callbacks.run_callbacks(model, epoch, i, X, y, y_pred, train_loss, False)

            callbacks.run_callbacks(model, epoch, None, X, y, y_pred, train_loss, False)

            # Validation phase
            for i, (X, y) in enumerate(val_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                with t.no_grad():
                    # TODO: don't throw away all the val data maybe??
                    y_pred = model(X)
                    val_loss = loss_fn(y_pred, y)

            callbacks.run_callbacks(model, epoch, None, X, y, y_pred, val_loss, True)

            postfix = {"train_loss": train_loss.item(), "val_loss": val_loss.item()}
            if t.cuda.is_available():
                postfix = {
                    **postfix,
                    "gpu memory usage (GiB)": gpu_mem_usage / 2**30,
                    "gpu utilization %": t.cuda.utilization(self.device),
                }
            progress_bar.set_postfix(postfix)


class Callbacks:
    def __init__(self, *callbacks: "Callback"):
        self.callbacks = callbacks

    def run_callbacks(self, model, epoch, batch_index, X, y, y_pred, loss, validation=False):
        for callback in self.callbacks:
            if callback.run_condition(epoch, batch_index, validation):
                callback.run(model, epoch, batch_index, X, y, y_pred, loss, validation)


class Callback:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def run_condition(self, epoch: int, batch_index: Optional[int], validation=False):
        return True

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
    def __init__(self, every_n_batches: Optional[int], output_dir: Optional[Path] = None):
        super().__init__(output_dir)
        self.every_n_batches = every_n_batches
        self.parameters_dir = self.output_dir / "parameter_checkpoints"
        os.makedirs(self.parameters_dir, exist_ok=True)

    def run_condition(self, epoch: int, batch_index: Optional[int], validation=False):
        if validation or batch_index is None:
            return False

        if not self.every_n_batches and batch_index == 0:
            return True

        if self.every_n_batches and batch_index % self.every_n_batches == 0:
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
    ):
        state_name = f"epoch_{epoch}" + (f"_batch_{batch_index}" if batch_index is not None else "_end")
        t.save(
            model.state_dict(),
            self.parameters_dir / (state_name + ".pt"),
        )


class SaveMetrics(Callback):
    def __init__(self, output_dir: Optional[Path] = None):
        super().__init__(output_dir)
        self.current_metrics = {}
        self.stored_metrics = []

    def run_condition(self, epoch: int, batch_index: Optional[int], validation=False):
        if batch_index is None:
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

        raw_metrics = self.calculate_metrics(model, X, y, y_pred, loss)
        metrics = self.combine_metrics(raw_metrics, epoch, batch_index, validation)

        if not validation:
            self.stored_metrics.append(metrics)

        else:
            self.stored_metrics[-1] = metrics
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
