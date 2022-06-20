from mega_experiment.training_jobs.example_training_job import *
from mega_experiment.training_jobs.training_job import TrainingJobSettings
from mega_experiment.job_runner import job_runner

if __name__ == "__main__":
    settings = TrainingJobSettings(
        **{
            "job_type": "ExampleTrainingJob",
            "job_description": "Example training job for demo and testing purposes",
            "n_init_repeats": 5,
        }
    )

    job_hyperparams = JobHyperparams(**{"n_inputs": 16, "batch_size": 32, "n_batches": 500, "n_epochs": 5, "lr": 0.001})

    training_job = ExampleTrainingJob(settings)
    job_runner = job_runner.JobRunner(training_job, job_hyperparams)
    job_runner.run()
