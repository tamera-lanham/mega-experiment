from dataclasses import asdict
from itertools import product
import json
import os
from math import prod
from mega_experiment.job_runner.instance_runner import InstanceRunner
from mega_experiment.job_runner.training_job import HyperparamOptions, HyperparamsBase, TrainingJob
from pathlib import Path
import torch as t
from tqdm.contrib.concurrent import process_map
from typing import Iterable, Optional
import multiprocessing


class JobRunner:
    def __init__(self, training_job: TrainingJob):
        self.training_job = training_job
        self.settings = self.training_job.settings
        self.job_ouput_dir = Path("outputs") / self.settings.output_dirname
        self.gpus = t.cuda.device_count()

    def run(self, job_hyperparams: object):
        # job_hyperparams is an arg here and not on __init__ for pickleability reasons

        self.save_settings_and_job_hyperparams(job_hyperparams)
        n_instances = self.get_n_instances(job_hyperparams)
        hyperparams_generator = self.generate_instance_hyperparams(job_hyperparams)

        multiprocessing.set_start_method("spawn")
        process_map(  # This is tqdm's version of multiprocessing.map, which also creates a progress bar
            self.run_instance,
            hyperparams_generator,
            max_workers=self.settings.n_processes,
            total=n_instances,
            desc=f"{self.settings.output_dirname} progress",
            unit="instance",
            position=self.settings.n_processes + 2,
        )

    def save_settings_and_job_hyperparams(self, job_hyperparams):
        os.makedirs(self.job_ouput_dir)
        with open(self.job_ouput_dir / "job_settings.json", "w") as f:
            json.dump(asdict(self.settings), f)
        with open(self.job_ouput_dir / "job_hyperparams.json", "w") as f:
            json.dump(asdict(job_hyperparams), f)

    def run_instance(self, hyperparams: HyperparamsBase):
        process_num = int(multiprocessing.current_process().name.split("-")[-1])
        device = self.choose_device(process_num)
        instance_runner = InstanceRunner(self.training_job, hyperparams, self.job_ouput_dir, device, process_num)
        instance_runner.run()

    def get_n_instances(self, job_hyperparams) -> int:
        option_counts = [len(v) for v in asdict(job_hyperparams).values() if isinstance(v, HyperparamOptions)]
        return self.settings.n_instance_repeats * prod(option_counts)

    def generate_instance_hyperparams(self, job_hyperparams) -> Iterable[HyperparamsBase]:

        hyperparams_with_options = {
            k: v for k, v in asdict(job_hyperparams).items() if isinstance(v, HyperparamOptions)
        }
        as_tuples = [[(label, value) for value in options] for label, options in hyperparams_with_options.items()]
        instance_hyperparam_values = (
            {**asdict(job_hyperparams), **dict(hyperparam_values)} for hyperparam_values in product(*as_tuples)
        )

        i = 0
        for instance_hyperparams in instance_hyperparam_values:
            for _ in range(self.settings.n_instance_repeats):
                yield job_hyperparams.hyperparams_class(instance_id=i, **instance_hyperparams)
                i += 1

    def choose_device(self, process_num) -> Optional[t.device]:
        if not self.gpus:
            return None
        gpu_index = process_num % self.gpus
        return t.device(f"cuda:{gpu_index}")
