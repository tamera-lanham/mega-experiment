from mega_experiment.training_jobs.example_training_job import *
from mega_experiment.job_runner.training_job import HyperparamOptions, TrainingJobSettings
from mega_experiment.job_runner.job_runner import JobRunner

if __name__ == "__main__":
    settings = TrainingJobSettings(
        job_type="ExampleTrainingJob",
        job_description="Example training job for demo and testing purposes",
        n_instance_repeats=2,
        save_parameters_every=(1, 250),
        save_metrics_every=(1, 250),
    )

    job_hyperparams = JobHyperparams(
        **{
            "n_inputs": HyperparamOptions([8, 16, 32]),
            "batch_size": 32,
            "n_batches": 500,
            "n_epochs": 10,
            "lr": HyperparamOptions([1e-3, 1e-5]),
        }
    )

    training_job = ExampleTrainingJob(settings)
    job_runner = JobRunner(training_job)
    job_runner.run(job_hyperparams)
