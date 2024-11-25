
import torchvision
import torch.nn as nn

from networks.basenetwork import BaseNetwork
from torch import Tensor


class InceptionV3(BaseNetwork):

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        pretrained = None
        if self.pretrained:
            pretrained = 'DEFAULT'
        self.model = torchvision.models.inception_v3(weights=pretrained)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.num_classes),
            nn.LogSoftmax(1)
        )
        self.model.AuxLogits.fc = nn.Linear(768, self.num_classes)

    def forward(self, tensors: Tensor) -> Tensor:
        y = self.model(tensors)
        if self.model.training:
            y = y[0]
        return y
