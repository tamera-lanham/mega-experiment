from mega_experiment.job_runner.instance_runner import *
from mega_experiment.training_jobs.example_training_job import (
    ExampleTrainingJob,
    Hyperparams,
)
import os
from pathlib import Path
import shutil
import torch as t


def test_pre_training_setup():
    training_job = ExampleTrainingJob()
    hyperparams = Hyperparams(instance_id=42)
    base_output_dir = Path("outputs/instance-runner-test")

    instance_runner = InstanceRunner(training_job, hyperparams, base_output_dir)
    instance_runner.pre_training_setup()

    assert (base_output_dir / "instances" / str(hyperparams.instance_id)).is_dir()
    assert (base_output_dir / "instances" / str(hyperparams.instance_id) / "model_torchscript.pt").is_file()
    assert (base_output_dir / "instance_metadata.jsonl").is_file()

    shutil.rmtree(base_output_dir)
