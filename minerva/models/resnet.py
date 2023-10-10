# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
#
"""Module containing ResNets adapted for use in :mod:`minerva`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "ResNet",
    "ResNetX",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn.modules as nn
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from .core import MinervaModel, get_torch_weights


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class ResNet(MinervaModel):
    """Modified version of the ResNet network to handle multi-spectral inputs and cross-entropy.

    Attributes:
        encoder_on (bool): Whether to initialise the :class:`ResNet` as an encoder or end-to-end classifier.
            If ``True``, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If ``False``, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
        inplanes (int): Number of input feature maps. Initially set to ``64``.
        dilation (int): Dilation factor of convolutions. Initially set to ``1``.
        groups (int): Number of convolutions in grouped convolutions of Bottleneck Blocks.
        base_width (int): Modifies the number of feature maps in convolutional layers of Bottleneck Blocks.
        conv1 (~torch.nn.Conv2d): Input convolutional layer of Conv1 input block to the network.
        bn1 (~torch.nn.Module): Batch normalisation layer of the Conv1 input block to the network.
        relu (~torch.nn.ReLU): Rectified Linear Unit (ReLU) activation layer to be used throughout ResNet.
        maxpool (~torch.nn.MaxPool2d): 3x3 Max-pooling layer with stride 2 of the Conv1 input block to the network.
        layer1 (~torch.nn.Sequential): Layer 1 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        layer2 (~torch.nn.Sequential): Layer 2 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        layer3 (~torch.nn.Sequential): Layer 3 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        layer4 (~torch.nn.Sequential): Layer 4 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        avgpool (~torch.nn.AdaptiveAvgPool2d): Global average pooling layer taking the output from the last block.
            Only initialised if ``encoder_on=False``.
        fc (~torch.nn.Linear): Fully connected layer that takes the flattened output from average pooling
            to a classification output. Only initialised if ``encoder_on=False``.

    .. warning::
        Layers using :class:`BasicBlock` are not compatible with anything other than the default values for
        ``groups`` and ``width_per_group``.

    Args:
        block (~torchvision.models.resnet.BasicBlock | ~torchvision.models.resnet.Bottleneck): Type of block operations
            to use throughout network.
        layers (list[int] | tuple[int, int, int, int]): Number of blocks in each of the 4 ``layers``.
        in_channels (int): Optional; Number of channels (or bands) in the input imagery.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If ``True``, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Optional; Number of convolutions in grouped convolutions of Bottleneck Blocks.
            Not compatible with Basic Block!
        width_per_group (int): Optional; Modifies the number of feature maps in convolutional layers
            of Bottleneck Blocks. Not compatible with Basic Block!
        replace_stride_with_dilation (tuple[bool, bool, bool]): Optional; Each element in the tuple indicates
            whether to replace the ``2x2`` stride with a dilated convolution instead.
            Must be a three element tuple of bools.
        norm_layer (~typing.Callable[..., ~torch.nn.Module]): Optional; Normalisation layer to use in each block.
            Typically, :class:`~torch.nn.BatchNorm2d`.
        encoder (bool): Optional; Whether to initialise the :class:`ResNet` as an encoder or end-to-end classifier.
            If ``True``, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If ``False``, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.

    Raises:
        ValueError: If ``replace_stride_with_dilation`` is not None or a 3-element tuple.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: Union[List[int], Tuple[int, int, int, int]],
        in_channels: int = 3,
        n_classes: int = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        encoder: bool = False,
    ) -> None:
        super(ResNet, self).__init__()

        # Inits normalisation layer for use in each block.
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Specifies if this network is to be configured as an encoder backbone or an end-to-end classifier.
        self.encoder_on = encoder

        # Sets the number of input feature maps to an init of 64.
        self.inplanes = 64

        # Init dilation of convolutions set to 1.
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = (False, False, False)

        # Raises ValueError if replace_stride_with_dilation is not a 3-element tuple of bools.
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        # Sets the number of convolutions in groups and the base width of convolutions.
        self.groups = groups
        self.base_width = width_per_group

        # --- CONV1 LAYER =============================================================================================
        # Adds the input convolutional layer to the network.
        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3,
            bias=False,
        )
        # Adds the batch norm layer for the Conv1 layer.
        self.bn1 = norm_layer(self.inplanes)

        # Inits the ReLU to be use in Conv1 and throughout the network.
        self.relu = nn.ReLU(inplace=True)

        # Adds the max pooling layer to complete the Conv1 layer.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- LAYERS 1-4 ==============================================================================================
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        # =============================================================================================================

        # Adds average pooling and classification layer to network if this is an end-to-end classifier.
        if not self.encoder_on:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, n_classes)

        # Performs weight initialisation across network.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # If set to, zero-initialise the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        try:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample,
                    self.groups,
                    self.base_width,
                    previous_dilation,
                    norm_layer,
                )
            )

        except ValueError as err:
            print(err.args)
            print("Setting groups=1, base_width=64 and trying again")
            self.groups = 1
            self.base_width = 64
            try:
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        stride,
                        downsample,
                        self.groups,
                        self.base_width,
                        previous_dilation,
                        norm_layer,
                    )
                )

            except ValueError as err:  # pragma: no cover
                print(err.args)

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0: Tensor = self.maxpool(x)

        x1: Tensor = self.layer1(x0)
        x2: Tensor = self.layer2(x1)
        x3: Tensor = self.layer3(x2)
        x4: Tensor = self.layer4(x3)

        if self.encoder_on:
            return x4, x3, x2, x1, x0

        else:
            x5 = self.avgpool(x4)
            x5 = torch.flatten(x5, 1)  # type: ignore[attr-defined]
            x5 = self.fc(x5)

            assert isinstance(x5, Tensor)
            return x5

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. ``model.forward``) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor | tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor]: If
            initialised as an encoder, returns a tuple of outputs from each ``layer`` 1-4. Else,
            returns :class:`~torch.Tensor` of the likelihoods the network places on the
            input ``x`` being of each class.
        """
        return self._forward_impl(x)


class ResNetX(MinervaModel):
    """Helper class to allow for easy creation of ResNet variant classes of the base :class:`ResNet` class.

    Example:
        To build a :class:`ResNet` variant class, just simply set the appropiate attributes in the definition of
        your new variant class that inherits from :class:`ResNetX`.

        >>> from minerva.models import ResNetX
        >>> from torchvision.models.resnet import Bottleneck
        >>>
        >>> class MyResNet101(ResNetX):
        >>>     layer_struct = [3, 4, 23, 3]
        >>>     block_type = BottleNeck
        >>>     weights_name = "ResNet101_Weights.IMAGENET1K_V1"

        You can then construct an instance of your new class like any other :class:`ResNet` with the added bonus
        of being able to use pre-trained torch weights:

        >>> model = ResNet101(*args, **kwargs, torch_weights=True)

    Attributes:
        block_type (~torchvision.models.resnet.BasicBlock | ~torchvision.models.resnet.Bottleneck): Type of the *block*
            used to construct the :class:`ResNet` layers.
        layer_struct (list[int]): Number of layers per block in the :class:`ResNet`.
        weights_name (str): Name of the :mod:`torch` pre-trained weights to use if ``torch_weights==True``.
        network (ResNet): :class:`ResNet` network.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int, int, int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If ``True``, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        replace_stride_with_dilation (tuple[bool, bool, bool]): Optional; Each element in the tuple indicates whether
            to replace the ``2x2`` stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (~typing.Callable[..., ~torch.nn.Module]): Optional; Normalisation layer to use in each block.
            Typically :class:`~torch.nn.BatchNorm2d`.
        encoder (bool): Optional; Whether to initialise the :class:`ResNet` as an encoder or end-to-end classifier.
            If ``True``, forward method returns the output of each layer block.
            ``avgpool`` and ``fc`` are not initialised.
            If ``False``, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
        torch_weights (bool): Optional; Whether to use the pre-trained weights from ``torchvision``. See note.

    Note:
        If using ``torch_weights``, the weight ``state_dict`` is modified to remove incompatible layers
        (such as the ``conv1`` layer) if ``input_size`` is non-RGB (i.e not 3-channel) and/or images
        smaller than ``224x224`` with the randomly initialised ``conv1`` that is appropiate for ``input_size``.
    """

    __metaclass__ = abc.ABCMeta
    block_type: Union[Type[BasicBlock], Type[Bottleneck]] = BasicBlock
    layer_struct: List[int] = [2, 2, 2, 2]
    weights_name = "ResNet18_Weights.IMAGENET1K_V1"

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        n_classes: int = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        encoder: bool = False,
        torch_weights: bool = False,
    ) -> None:
        super(ResNetX, self).__init__(
            criterion=criterion, input_size=input_size, n_classes=n_classes
        )

        self.network = ResNet(
            self.block_type,
            self.layer_struct,
            in_channels=input_size[0],
            n_classes=n_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            encoder=encoder,
        )

        if torch_weights:
            print(f"{self.weights_name=}")
            self.network = _preload_weights(
                self.network, get_torch_weights(self.weights_name), input_size, encoder
            )

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor | tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor]: If
            initialised as an encoder, returns a tuple of outputs from each ``layer`` 1-4. Else, returns
            :class:`~torch.Tensor` of the likelihoods the network places on the input ``x`` being of each class.
        """
        z: Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = self.network(
            x
        )
        if isinstance(z, Tensor):
            return z
        elif isinstance(z, tuple):
            assert all(isinstance(n, Tensor) for n in z)
            return z


class ResNet18(ResNetX):
    """ResNet18 modified from source to have customisable number of input channels and to be used as a backbone
    by stripping classification layers away.
    """

    layer_struct: List[int] = [2, 2, 2, 2]
    weights_name = "ResNet18_Weights.IMAGENET1K_V1"


class ResNet34(ResNetX):
    """ResNet34 modified from source to have customisable number of input channels and to be used as a backbone
    by stripping classification layers away.
    """

    layer_struct: List[int] = [3, 4, 6, 3]
    weights_name = "ResNet34_Weights.IMAGENET1K_V1"


class ResNet50(ResNetX):
    """ResNet50 modified from source to have customisable number of input channels and to be used as a backbone
    by stripping classification layers away.
    """

    block_type = Bottleneck
    layer_struct = [3, 4, 6, 3]
    weights_name = "ResNet50_Weights.IMAGENET1K_V1"


class ResNet101(ResNetX):
    """ResNet101 modified from source to have customisable number of input channels and to be used as a backbone
    by stripping classification layers away.
    """

    block_type = Bottleneck
    layer_struct = [3, 4, 23, 3]
    weights_name = "ResNet101_Weights.IMAGENET1K_V1"


class ResNet152(ResNetX):
    """ResNet152 modified from source to have customisable number of input channels and to be used as a backbone
    by stripping classification layers away.
    """

    block_type = Bottleneck
    layer_struct = [3, 8, 36, 3]
    weights_name = "ResNet152_Weights.IMAGENET1K_V1"


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def _preload_weights(
    resnet: ResNet,
    weights: Optional[Union[WeightsEnum, Any]],
    input_shape: Tuple[int, int, int],
    encoder_on: bool,
) -> ResNet:
    if not weights:
        print("Weights are None! The original resnet will be used")
        return resnet

    if isinstance(weights, WeightsEnum):
        weights = weights.get_state_dict(True)

    if input_shape[0] != 3 or input_shape[1] <= 224 or input_shape[2] <= 224:
        weights["conv1.weight"] = resnet.conv1.state_dict()["weight"]  # type: ignore[attr-defined]

    if encoder_on:
        del weights["fc.weight"]  # type: ignore[attr-defined]
        del weights["fc.bias"]  # type: ignore[attr-defined]

    resnet.load_state_dict(weights)

    return resnet
