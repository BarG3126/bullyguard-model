from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from bullyguard.config_schemas.base_schemas import TaskConfig
from bullyguard.config_schemas.infrastructure import infrastructure_schema
from bullyguard.config_schemas.training import training_task_schemas
from bullyguard.config_schemas.evaluation import model_selector_schemas
from typing import Optional


@dataclass
class Config:
    infrastructure: infrastructure_schema.InfrastructureConfig = infrastructure_schema.InfrastructureConfig()
    save_last_checkpoint_every_n_train_steps: int = 500
    seed: int = 1234
    tasks: dict[str, TaskConfig] = MISSING
    model_selector: Optional[model_selector_schemas.ModelSelectorConfig] = None
    registered_model_name: Optional[str] = None


def setup_config() -> None:
    infrastructure_schema.setup_config()
    training_task_schemas.setup_config()
    model_selector_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
