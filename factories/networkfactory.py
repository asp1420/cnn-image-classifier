from constants.networkenum import Network
from networks import (
    BaseNetwork,
    InceptionV3,
    Resnet50,
    Resnext50,
    EfficientnetV2,
    SwinTransformerV2
)



class NetworkFactory:

    @staticmethod
    def create(ntype: Network, num_classes:int, pretrained: bool=True) -> BaseNetwork:
        network = None
        if ntype == Network.INCEPTION:
            network = InceptionV3(num_classes=num_classes, pretrained=pretrained)
        elif ntype == Network.RESNET50:
            network = Resnet50(num_classes=num_classes, pretrained=pretrained)
        elif ntype == Network.RESNEXT50:
            network = Resnext50(num_classes=num_classes, pretrained=pretrained)
        elif ntype == Network.EFFICIENTNETV2:
            network = EfficientnetV2(num_classes=num_classes, pretrained=pretrained)
        elif ntype == Network.SWINGTRANSFORMERV2:
            network = SwinTransformerV2(num_classes=num_classes, pretrained=pretrained)
        return network
