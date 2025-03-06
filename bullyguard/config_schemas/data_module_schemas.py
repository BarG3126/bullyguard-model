from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI

from bullyguard.config_schemas import transformations_schemas


@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    batch_size: int = MISSING
    shuffle: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False


@dataclass
class TextClassificationDataModuleConfig(DataModuleConfig):
    _target_: str = "bullyguard.data_modules.data_modules.TextClassificationDataModule"
    train_df_path: str = MISSING
    dev_df_path: str = MISSING
    test_df_path: str = MISSING
    transformation: transformations_schemas.TransformationConfig = MISSING
    text_column_name: str = "cleaned_text"
    label_column_name: str = "label"


@dataclass
class ScrappedDataTextClassificationDataModuleConfig(TextClassificationDataModuleConfig):
    batch_size: int = 64
    train_df_path: str = "gs://bullyguard/data/processed/rebalanced_splits/train.parquet"
    dev_df_path: str = "gs://bullyguard/data/processed/rebalanced_splits/dev.parquet"
    test_df_path: str = "gs://bullyguard/data/processed/rebalanced_splits/test.parquet"
    transformation: transformations_schemas.TransformationConfig = SI(
        "${..lightning_module.model.backbone.transformation}"
    )


def setup_config() -> None:
    transformations_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema",
        group="tasks/data_module",
        node=TextClassificationDataModuleConfig,
    )
