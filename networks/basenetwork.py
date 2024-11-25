import torch.nn as nn


class BaseNetwork(nn.Module):

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
