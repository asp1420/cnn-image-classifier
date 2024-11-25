import torchvision
import torch.nn as nn

from networks.basenetwork import BaseNetwork
from torch import Tensor


class Resnext50(BaseNetwork):

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        if self.pretrained:
            pretrained = 'DEFAULT'
        self.model = torchvision.models.resnext50_32x4d(weights=pretrained)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.num_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, tensors: Tensor) -> Tensor:
        return self.model(tensors)
