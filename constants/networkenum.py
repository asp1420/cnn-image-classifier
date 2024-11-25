from enum import Enum, auto


class Network(Enum):
    INCEPTION = auto()
    RESNET50 = auto()
    RESNEXT50 = auto()
    EFFICIENTNETV2 = auto()
    SWINGTRANSFORMERV2 = auto()
