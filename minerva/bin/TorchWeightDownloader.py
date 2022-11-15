from minerva.models import get_torch_weights
from torchvision.models._api import WeightsEnum

resnets = [
    "ResNet101_Weights.IMAGENET1K_V1",
    "ResNet152_Weights.IMAGENET1K_V1",
    "ResNet18_Weights.IMAGENET1K_V1",
    "ResNet34_Weights.IMAGENET1K_V1",
    "ResNet50_Weights.IMAGENET1K_V1",
]

for resnet in resnets:
    weights: WeightsEnum = get_torch_weights(resnet)
    weight_dict = weights.get_state_dict(True)
