import os
from abc import abstractmethod
from typing import Any, Callable, Iterable, Optional, Union

from lightning.pytorch import LightningModule
from torch import Tensor
import torch
from torch.optim import Optimizer

from bullyguard.models.transformations import Transformation
from bullyguard.models.models import Model
from bullyguard.training.loss_functions import LossFunction
from bullyguard.training.schedulers import LightningScheduler
from bullyguard.utils.io_utils import open_file
from bullyguard.utils.utils import get_logger

PartialOptimizerType = Callable[[Union[Iterable[Tensor], dict[str, Iterable[Tensor]]]], Optimizer]


class TrainingLightningModule(LightningModule):
    def __init__(
        self,
        model: Model,
        loss: LossFunction,
        optimizer: PartialOptimizerType,
        scheduler: Optional[LightningScheduler] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.partial_optimizer = optimizer
        self.scheduler = scheduler

        self.logging_logger = get_logger(self.__class__.__name__)

    def configure_optimizers(self) -> Union[Optimizer, tuple[list[Optimizer], list[dict[str, Any]]]]:
        optimizer = self.partial_optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler.configure_scheduler(
                optimizer=optimizer, estimated_stepping_batches=self.trainer.estimated_stepping_batches
            )
            return [optimizer], [scheduler]

        return optimizer

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        ...

    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...


class ModelStateDictExportingTrainingLightningModule(TrainingLightningModule):
    @abstractmethod
    def export_model_state_dict(self, checkpoint_path: str) -> str:
        """
        Export model state dict from LightningModule checkpoint and save it
        to the same location as the checkpoint_path, and return the save path
        """

    def common_export_model_state_dict(self, checkpoint_path: str) -> str:
        with open_file(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"), weights_only=False)["state_dict"]

        model_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("loss."):
                model_state_dict[key.replace("model.", "", 1)] = value

        model_state_dict_save_path = os.path.join(os.path.dirname(checkpoint_path), "model_state_dict.pth")

        with open_file(model_state_dict_save_path, "wb") as f:
            torch.save(model_state_dict, f)

        return model_state_dict_save_path
