from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from bullyguard.config_schemas.base_schemas import LightningModuleConfig
from bullyguard.config_schemas.models.model_schemas import BertTinyBinaryTextClassificationModelConfig, ModelConfig
from bullyguard.config_schemas.training import loss_schemas, optimizer_schemas, scheduler_schemas


@dataclass
class TrainingLightningModuleConfig(LightningModuleConfig):
    _target_: str = MISSING
    model: ModelConfig = MISSING
    loss: loss_schemas.LossFunctionConfig = MISSING
    optimizer: optimizer_schemas.OptimizerConfig = MISSING
    scheduler: Optional[scheduler_schemas.LightningSchedulerConfig] = None


@dataclass
class BinaryTextClassificationTrainingLightningModuleConfig(TrainingLightningModuleConfig):
    _target_: str = "bullyguard.training.lightning_modules.binary_text_classification.BinaryTextClassificationTrainingLightningModule"


@dataclass
class BullyguardBinaryTextClassificationTrainingLightningModuleConfig(
    BinaryTextClassificationTrainingLightningModuleConfig
):
    model: ModelConfig = BertTinyBinaryTextClassificationModelConfig()
    loss: loss_schemas.LossFunctionConfig = loss_schemas.BCEWithLogitsLossConfig()
    optimizer: optimizer_schemas.OptimizerConfig = optimizer_schemas.AdamWOptimizerConfig()
    scheduler: Optional[
        scheduler_schemas.LightningSchedulerConfig
    ] = scheduler_schemas.ReduceLROnPlateauLightningSchedulerConfig()


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_training_lightning_module_schema",
        group="tasks/lightning_module",
        node=BinaryTextClassificationTrainingLightningModuleConfig,
    )
