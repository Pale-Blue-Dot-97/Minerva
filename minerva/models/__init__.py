from .core import (
    MinervaModel as MinervaModel,
    MinervaDataParallel as MinervaDataParallel,
    MinervaBackbone as MinervaBackbone,
    get_torch_weights as get_torch_weights,
    get_output_shape as get_output_shape,
    bilinear_init as bilinear_init,
)

from .fcn import (
    FCN8ResNet18 as FCN8ResNet18,
    FCN8ResNet34 as FCN8ResNet34,
    FCN8ResNet50 as FCN8ResNet50,
    FCN8ResNet101 as FCN8ResNet101,
    FCN8ResNet152 as FCN8ResNet152,
    FCN16ResNet18 as FCN16ResNet18,
    FCN16ResNet34 as FCN16ResNet34,
    FCN16ResNet50 as FCN16ResNet50,
    FCN32ResNet18 as FCN32ResNet18,
    FCN32ResNet34 as FCN32ResNet34,
    FCN32ResNet50 as FCN32ResNet50,
)

from .resnet import (
    ResNet18 as ResNet18,
    ResNet34 as ResNet34,
    ResNet50 as ResNet50,
    ResNet101 as ResNet101,
    ResNet152 as ResNet152,
)

from .siamese import (
    SimCLR18 as SimCLR18,
    SimCLR34 as SimCLR34,
    SimCLR50 as SimCLR50,
    SimSiam18 as SimSiam18,
    SimSiam34 as SimSiam34,
    SimSiam50 as SimSiam50,
)

from .unet import UNetR18 as UNetR18

from .__depreciated import MLP, CNN
