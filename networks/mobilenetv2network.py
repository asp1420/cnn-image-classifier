import torchvision
import torch.nn as nn

from networks.basenetwork import BaseNetwork
from torch import Tensor


class MobileNetV2(BaseNetwork):
    
    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        pretrained = None
        if self.pretrained:
            pretrained = 'DEFAULT'
        self.model = torchvision.models.mobilenet_v2(weights='DEFAULT')
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, self.num_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, tensors: Tensor) -> Tensor:
        return self.model(tensors)
