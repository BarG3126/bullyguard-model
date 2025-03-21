from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from bullyguard.config_schemas.evaluation import model_selector_schemas
from bullyguard.config_schemas.base_schemas import TaskConfig
from bullyguard.config_schemas.config_schema import Config
from bullyguard.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig
from bullyguard.config_schemas.evaluation.evaluation_task_schemas import DefaultCommonEvaluationTaskConfig


@dataclass
class LocalBertExperiment(Config):
    tasks: dict[str, TaskConfig] = field(
        default_factory=lambda: {
            "binary_text_classification_task": DefaultCommonTrainingTaskConfig(),
            "binary_text_evaluation_task": DefaultCommonEvaluationTaskConfig(),
        }
    )

    model_selector: Optional[
        model_selector_schemas.ModelSelectorConfig
    ] = model_selector_schemas.BullyingDetectionModelSelectorConfig()
    registered_model_name: Optional[str] = "bert_tiny"


FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist([
        "tasks.binary_text_evaluation_task.tar_model_path=${tasks.binary_text_classification_task.tar_model_export_path}",
        "tasks.binary_text_evaluation_task.data_module=${tasks.binary_text_classification_task.data_module}",
        "tasks.binary_text_evaluation_task.trainer=${tasks.binary_text_classification_task.trainer}",
    ]),
)

cs = ConfigStore.instance()
cs.store(name="local_bert", group="experiment/bert", node=FinalLocalBertExperiment, package="_global_")
