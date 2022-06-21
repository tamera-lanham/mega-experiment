from mega_experiment.training_jobs.cifar_double_descent.job_definition import *
from mega_experiment.job_runner.training_job import HyperparamOptions, TrainingJobSettings
from mega_experiment.job_runner.job_runner import JobRunner

if __name__ == "__main__":
    settings = TrainingJobSettings(
        job_type="CifarDoubleDescent",
        job_description="Deep double descent replication using ResNet on CIFAR-10",
        n_instance_repeats=5,
        save_parameters_every_n_batches=None,
    )

    job_hyperparams = JobHyperparams(n_epochs=5, label_noise=0.2, batch_size=128, resnet_width=64)

    job = CifarDoubleDescentJob(settings)
    job_runner = JobRunner(job).run(job_hyperparams)
