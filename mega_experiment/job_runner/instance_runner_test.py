import json
from mega_experiment.job_runner.instance_runner import *
from mega_experiment.training_jobs.example.job_definition import ExampleTrainingJob, Hyperparams
from pathlib import Path
import shutil
import torch as t
from typing import OrderedDict


def test_pre_training_setup():
    training_job = ExampleTrainingJob()
    hyperparams = Hyperparams(instance_id=42)
    base_output_dir = Path("outputs/instance-runner-test")

    instance_runner = InstanceRunner(training_job, hyperparams, base_output_dir)
    instance_runner.pre_training_setup()

    assert (instance_runner.output_dir).is_dir()
    assert (instance_runner.output_dir / "model_torchscript.pt").is_file()
    assert (base_output_dir / "instance_hyperparams.jsonl").is_file()

    shutil.rmtree(base_output_dir)


def test_instance_runner():
    training_job = ExampleTrainingJob()
    hyperparams = Hyperparams(instance_id=42)
    base_output_dir = Path("outputs/instance-runner-test")

    instance_runner = InstanceRunner(training_job, hyperparams, base_output_dir)
    instance_runner.run()

    parameters = t.load(instance_runner.output_dir / "parameter_checkpoints" / "epoch_0_end.pt")
    assert isinstance(parameters, OrderedDict)
    assert isinstance(next(iter(parameters.values())), t.Tensor)

    with open(instance_runner.output_dir / "metrics.jsonl") as f:
        metrics = [json.loads(line) for line in f]
    assert all(key in metrics[0] for key in ["train_loss", "epoch", "batch"])

    shutil.rmtree(base_output_dir)
