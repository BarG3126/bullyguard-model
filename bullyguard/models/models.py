from abc import abstractmethod
from typing import Optional

from torch import Tensor, nn
from transformers import BatchEncoding

from bullyguard.models.transformations import Transformation
from bullyguard.models.adapters import Adapter
from bullyguard.models.backbones import Backbone
from bullyguard.models.heads import Head


class Model(nn.Module):
    @abstractmethod
    def get_transformation(self) -> Transformation:
        ...


class BinaryTextClassificationModel(Model):
    def __init__(self, backbone: Backbone, head: Head, adapter: Optional[Adapter]) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, encodings: BatchEncoding) -> Tensor:
        output = self.backbone(encodings)
        if self.adapter is not None:
            output = self.adapter(output)
        output = self.head(output)
        assert isinstance(output, Tensor)
        return output

    def get_transformation(self) -> Transformation:
        return self.backbone.get_transformation()
