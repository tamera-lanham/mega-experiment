# In charge of
# - Saving job settings and JobHyperparams
# - Generating hyperparam instances from JobHyperparams
# - Parallelizing calls to InstanceRunner
from dataclasses import asdict
from typing import Iterable, Type
from mega_experiment.job_runner.instance_runner import InstanceRunner
from mega_experiment.training_jobs.training_job import HyperparamsBase, TrainingJob
from pathlib import Path


class JobRunner:
    def __init__(self, training_job: TrainingJob, job_hyperparams: object):
        self.training_job = training_job
        self.settings = self.training_job.settings
        self.job_hyperparams = job_hyperparams
        self.Hyperparams = self.job_hyperparams.hyperparams_class

        self.job_ouput_dir = Path("outputs") / self.settings.output_dirname

    def run(self):
        # Save training_job.settings
        # Save job_hyperparams
        for instance_hyperparams in self.generate_instance_hyperparams():
            self.run_instance(instance_hyperparams)

    def run_instance(self, hyperparams):

        instance_runner = InstanceRunner(self.training_job, hyperparams, self.job_ouput_dir)
        instance_runner.run()

    def generate_instance_hyperparams(self) -> Iterable[HyperparamsBase]:
        # TODO: the cartesian product thing
        for i in range(self.settings.n_init_repeats):
            yield self.Hyperparams(instance_id=i, **asdict(self.job_hyperparams))
