from mega_experiment.training_jobs.example.job_definition import *
from mega_experiment.job_runner.training_job import HyperparamOptions, TrainingJobSettings
from mega_experiment.job_runner.job_runner import JobRunner

if __name__ == "__main__":
    settings = TrainingJobSettings(
        job_type="ExampleTrainingJob",
        job_description="Example training job for demo and testing purposes",
        n_instance_repeats=5,
        save_parameters_every_n_batches=250,
        n_processes=4,
    )

    job_hyperparams = JobHyperparams(
        n_inputs=HyperparamOptions([8, 16, 32]),
        batch_size=32,
        n_batches=500,
        n_epochs=10,
        lr=HyperparamOptions([1e-3, 1e-5]),
    )

    job = ExampleTrainingJob(settings)
    job_runner = JobRunner(job).run(job_hyperparams)
