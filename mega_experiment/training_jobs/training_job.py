# Metadata thoughts and questions:
# - What should the two metadata objects be like?
#   - One for the training job, another for each init
#   - Mainitaining typing between the two, so the user only has to define it once (should probably be defined on the init metadata, and then the training job class comes from that)
# - Should there be different metadata used to construct the model for torchscripting? Should the model just be torchscripted on each init?

from dataclasses import dataclass, fields, make_dataclass
from typing import Optional, Union, TypeVar

@dataclass
class TheseShouldProbablyBeConstructorArgsForATrainingJob:
    job_type: str = "TrainingJob"
    description: str = ""
    n_init_repeats: int = 1
    gcs_bucket: Optional[str] = None
    n_processes: int = 4



class HyperparamOptions(list[TypeVar("T")]):
    def __init__(self, *args, **kwargs):
        super().__init__(args[0])

def create_job_hyperparams_class(hyperparams_class: type):
    """
    Create a new dataclass based on an original, where the new class is the same as the original
    but with a HyperparamOptions type hint option on each of the attibutes.
    """

    enhance_type = lambda type_: Union[type_, HyperparamOptions[type_]]

    job_metadata_class_fields = []
    for field in fields(hyperparams_class):
        job_metadata_class_fields.append((field.name, enhance_type(field.type), field))

    return make_dataclass('JobHyperparams', job_metadata_class_fields)
