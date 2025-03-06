from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from bullyguard.config_schemas.base_schemas import TaskConfig
from bullyguard.config_schemas.config_schema import Config
from bullyguard.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig


@dataclass
class LocalBertExperiment(Config):
    tasks: dict[str, TaskConfig] = field(
        default_factory=lambda: {
            "binary_text_classification_task": DefaultCommonTrainingTaskConfig(),
        }
    )


FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist([]),
)

cs = ConfigStore.instance()
cs.store(name="local_bert", group="experiment/bert", node=FinalLocalBertExperiment, package="_global_")
