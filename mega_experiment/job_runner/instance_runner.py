# In charge of:
# - Saving hyperparams
# - Tracing and saving the model
# - Training the model
# - Saving the params, metrics, val outputs and such while training
# - Saving to GCP
# - Reporting progress in the console??

from dataclasses import asdict
from mega_experiment.training_jobs.training_job import TrainingJob, HyperparamsBase
import json
import os
from pathlib import Path
import torch as t


class InstanceRunner:

    def __init__(self, training_job: TrainingJob, hyperparams: HyperparamsBase, job_output_dir: Path):
        self.training_job = training_job
        self.job_ouput_dir = job_output_dir
        self.hyperparams = hyperparams

    def run(self):
        self.pre_training_setup()
        # Train
        # Save the params, metrics, val outputs and such while training
        # Save to GCP

    def pre_training_setup(self):
        # Make the output dir
        output_dir = self.job_ouput_dir / "instances" / str(self.hyperparams.instance_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save hyperparams
        with open(self.job_ouput_dir / "instance_metadata.jsonl", "a") as f:
            f.write(json.dumps(asdict(self.hyperparams)) + "\n")

        # Trace and save model
        model = self.training_job.model(self.hyperparams)
        scripted_model = t.jit.script(model)
        t.jit.save(scripted_model, output_dir / "model_torchscript.pt")

