from dataclasses import dataclass

from omegaconf import MISSING

from bullyguard.config_schemas.data_module_schemas import DataModuleConfig
from bullyguard.config_schemas.trainer.trainer_schemas import TrainerConfig


@dataclass
class LightningModuleConfig:
    _target_: str = MISSING


@dataclass
class TaskConfig:
    _target_: str = MISSING
    name: str = MISSING
    data_module: DataModuleConfig = MISSING
    lightning_module: LightningModuleConfig = MISSING
    trainer: TrainerConfig = MISSING
