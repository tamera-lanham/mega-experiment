from dataclasses import asdict
from itertools import product
import json
import os
from mega_experiment.job_runner.instance_runner import InstanceRunner
from mega_experiment.training_jobs.training_job import HyperparamOptions, HyperparamsBase, TrainingJob
from pathlib import Path
from typing import Iterable


class JobRunner:
    def __init__(self, training_job: TrainingJob, job_hyperparams: object):
        self.training_job = training_job
        self.settings = self.training_job.settings
        self.job_hyperparams = job_hyperparams

        self.job_ouput_dir = Path("outputs") / self.settings.output_dirname

    def run(self):
        self.save_settings_and_job_hyperparams()

        for instance_hyperparams in self.generate_instance_hyperparams():
            self.run_instance(instance_hyperparams)

    def save_settings_and_job_hyperparams(self):
        os.makedirs(self.job_ouput_dir, exist_ok=True)  # TODO: remove exists ok
        with open(self.job_ouput_dir / "job_settings.json", "w") as f:
            json.dump(asdict(self.settings), f)
        with open(self.job_ouput_dir / "job_hyperparams.json", "w") as f:
            json.dump(asdict(self.job_hyperparams), f)

    def run_instance(self, hyperparams):
        instance_runner = InstanceRunner(self.training_job, hyperparams, self.job_ouput_dir)
        instance_runner.run()

    def generate_instance_hyperparams(self) -> Iterable[HyperparamsBase]:

        hyperparams_with_options = {
            k: v for k, v in asdict(self.job_hyperparams).items() if isinstance(v, HyperparamOptions)
        }
        as_tuples = [[(label, value) for value in options] for label, options in hyperparams_with_options.items()]
        instance_hyperparam_values = (
            {**asdict(self.job_hyperparams), **dict(hyperparam_values)} for hyperparam_values in product(*as_tuples)
        )

        i = 0
        for instance_hyperparams in instance_hyperparam_values:
            for _ in range(self.settings.n_instance_repeats):
                yield self.job_hyperparams.hyperparams_class(instance_id=i, **instance_hyperparams)
                i += 1
