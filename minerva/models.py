# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.

# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
#
# TODO: Consider removing redundant models.
#
"""Module containing neural network model classes."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Sequence,
    Union,
    overload,
)

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from minerva.utils import utils

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaModel(Module, ABC):
    """Abstract class to act as a base for all Minerva Models.

    Designed to provide inter-compatability with :class:`Trainer`.

    Attributes:
        criterion (Module): PyTorch loss function model will use.
        input_shape (tuple[int, int, int] or list[int]): The shape of the input data in order of
            number of channels, image width, image height.
        n_classes (int): Number of classes in input data.
        output_shape: The shape of the output of the network. Determined and set by determine_output_dim.
        optimiser: PyTorch optimiser model will use, to be initialised with inherited model's parameters.

    Args:
        criterion (Module): Optional; PyTorch loss function model will use.
        input_shape (tuple[int, int, int] or list[int]): Optional; Defines the shape of the input data in order of
            number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in input data.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        n_classes: Optional[int] = None,
    ) -> None:

        super(MinervaModel, self).__init__()

        # Sets loss function
        self.criterion: Optional[Module] = criterion

        self.input_shape = input_shape
        self.n_classes = n_classes

        # Output shape initialised as None. Should be set by calling determine_output_dim.
        self.output_shape: Optional[Union[int, Iterable[int]]] = None

        # Optimiser initialised as None as the model parameters created by its init is required to init a
        # torch optimiser. The optimiser MUST be set by calling set_optimiser before the model can be trained.
        self.optimiser: Optional[Optimizer] = None

    def set_optimiser(self, optimiser: Optimizer) -> None:
        """Sets the optimiser used by the model.

        .. warning::
            *MUST* be called after initialising a model and supplied with a PyTorch optimiser
            using this model's parameters.

        Args:
            optimiser (Optimizer): PyTorch optimiser model will use, initialised with this model's parameters.
        """
        self.optimiser = optimiser

    def determine_output_dim(self, sample_pairs: bool = False) -> None:
        """Uses get_output_shape to find the dimensions of the output of this model and sets to attribute."""

        assert self.input_shape is not None

        self.output_shape = get_output_shape(
            self, self.input_shape, sample_pairs=sample_pairs
        )

    @overload
    def step(
        self, x: Tensor, y: Tensor, train: bool = False
    ) -> Tuple[_Loss, Union[Tensor, Tuple[Tensor, ...]]]:
        ...

    @overload
    def step(
        self, x: Tensor, *, train: bool = False
    ) -> Tuple[_Loss, Union[Tensor, Tuple[Tensor, ...]]]:
        ...

    def step(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        train: bool = False,
    ) -> Tuple[_Loss, Union[Tensor, Tuple[Tensor, ...]]]:
        """Generic step of model fitting using a batch of data.

        Raises:
            NotImplementedError: If ``self.optimiser`` is None.
            NotImplementedError: If ``self.criterion`` is None.

        Args:
            x (Tensor): Batch of input data to network.
            y (Tensor): Either a batch of ground truth labels or generated labels/ pairs.
            train (bool): Sets whether this shall be a training step or not. True for training step which will then
                clear the optimiser, and perform a backward pass of the network then update the optimiser.
                If False for a validation or testing step, these actions are not taken.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """

        if self.optimiser is None:
            raise NotImplementedError("Optimiser has not been set!")

        if self.criterion is None:
            raise NotImplementedError("Criterion has not been set!")

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        # Forward pass.
        z: Union[Tensor, Tuple[Tensor, ...]] = self.forward(x)

        # Compute Loss.
        loss: _Loss = self.criterion(z, y)

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, z


class MinervaBackbone(ABC):
    """Abstract class to mark a model for use as a backbone."""

    __metaclass__ = abc.ABCMeta

    def __init__(self) -> None:
        super().__init__()

        self.backbone: MinervaModel

    def get_backbone(self) -> Module:
        """Gets the backbone network of the model.
        Returns:
            Module: The backbone of the model.
        """
        return self.backbone


class MinervaDataParallel(Module):
    """Custom wrapper for DataParallel that automatically fetches the attributes of the wrapped model.

    Attributes:
        model (Module): PyTorch Model to be wrapped by :class:`DataParallel`.

    Args:
        model (Module): PyTorch Model to be wrapped by :class:`DataParallel`.
    """

    def __init__(self, model: Module, Paralleliser: Module, *args, **kwargs) -> None:
        super(MinervaDataParallel, self).__init__()
        self.model = Paralleliser(model, *args, **kwargs).cuda()

    def forward(self, *input: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Ensures a forward call to the model goes to the actual wrapped model.

        Args:
            input (Tuple[Tensor, ...]): Input of tensors to be parsed to the model forward.
        Returns:
            Tuple[Tensor, ...]: Output of model.
        """
        z = self.model(*input)
        assert isinstance(z, tuple) and list(map(type, z)) == [Tensor] * len(z)
        return z

    def __call__(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

    def __repr__(self) -> Any:
        return self.model.__repr__()


class MLP(MinervaModel):
    """Simple class to construct a Multi-Layer Perceptron (MLP).

    Inherits from :class:`torch.nn.Module` and :class:`MinervaModel`. Designed for use with PyTorch functionality.

    Should be used in tandem with :class:`Trainer`.

    Attributes:
        input_size (int): Size of the input vector to the network.
        output_size (int): Size of the output vector of the network.
        hidden_sizes (tuple[int] or list[int]): Series of values for the size of each hidden layers within the network.
            Also determines the number of layers other than the required input and output layers.
        network (torch.nn.Sequential): The actual neural network of the model.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (int): Optional; Size of the input vector to the network.
        n_classes (int): Optional; Number of classes in input data.
            Determines the size of the output vector of the network.
        hidden_sizes (tuple[int] or list[int]): Optional; Series of values for the size of each hidden layers
            within the network. Also determines the number of layers other than the required input and output layers.
    """

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: int = 288,
        n_classes: int = 8,
        hidden_sizes: Union[Tuple[int, ...], List[int], int] = (256, 144),
    ) -> None:

        super(MLP, self).__init__(
            criterion=criterion, input_shape=(input_size,), n_classes=n_classes
        )

        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        self.hidden_sizes = hidden_sizes

        self._layers = OrderedDict()

        # Constructs layers of the network based on the input size, the hidden sizes and the number of classes.
        for i in range(len(hidden_sizes)):
            if i == 0:
                self._layers["Linear-0"] = torch.nn.Linear(input_size, hidden_sizes[i])
            else:
                self._layers[f"Linear-{i}"] = torch.nn.Linear(
                    hidden_sizes[i - 1], hidden_sizes[i]
                )

            # Adds ReLu activation after every linear layer.
            self._layers[f"ReLu-{i}"] = torch.nn.ReLU()

        # Adds the final classification layer.
        self._layers["Classification"] = torch.nn.Linear(hidden_sizes[-1], n_classes)

        # Constructs network from the OrderedDict of layers
        self.network = torch.nn.Sequential(self._layers)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the network.

        Can be called directly as a method of :class:`MLP` (e.g. ``model.forward()``)
        or when data is parsed to :class:`MLP` (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Tensor of the likelihoods the network places on the input ``x`` being of each class.
        """
        z = self.network(x)
        assert isinstance(z, Tensor)
        return z


class CNN(MinervaModel, ABC):
    """Simple class to construct a Convolutional Neural Network (CNN).

    Inherits from :class:`torch.nn.Module` and :class:`MinervaModel`. Designed for use with PyTorch functionality.

    Should be used in tandem with :class:`Trainer`.

    Attributes:
        flattened_size (int): Length of the vector resulting from the flattening of the output from the convolutional
            network.
        conv_net (torch.nn.Sequential): Convolutional network of the model.
        fc_net (torch.nn.Sequential): Fully connected network of the model.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in input data.
        features (tuple[int] or list[int]): Optional; Series of values defining the number of feature maps.
            The length of the list is also used to determine the number of convolutional layers in conv_net.
        conv_kernel_size (int or tuple[int]): Optional; Size of all convolutional kernels for all channels and layers.
        conv_stride (int or tuple[int]): Optional; Size of all convolutional stride lengths for all channels and layers.
        max_kernel_size (int or tuple[int]): Optional; Size of all max-pooling kernels for all channels and layers.
        max_stride (int or tuple[int]): Optional; Size of all max-pooling stride lengths for all channels and layers.
    """

    def __init__(
        self,
        criterion,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        n_classes: int = 8,
        features: Union[Tuple[int, ...], List[int]] = (2, 1, 1),
        fc_sizes: Union[Tuple[int, ...], List[int]] = (128, 64),
        conv_kernel_size: Union[int, Tuple[int, ...]] = 3,
        conv_stride: Union[int, Tuple[int, ...]] = 1,
        max_kernel_size: Union[int, Tuple[int, ...]] = 2,
        max_stride: Union[int, Tuple[int, ...]] = 2,
        conv_do: bool = True,
        fc_do: bool = True,
        p_conv_do: float = 0.1,
        p_fc_do: float = 0.5,
    ) -> None:

        super(CNN, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        self._conv_layers = OrderedDict()
        self._fc_layers = OrderedDict()

        # Checks that the kernel sizes and strides match the number of layers defined by features.
        _conv_kernel_size: Sequence[int] = utils.check_len(conv_kernel_size, features)
        _conv_stride: Sequence[int] = utils.check_len(conv_stride, features)

        # Constructs the convolutional layers determined by the number of input channels and the features of these.
        assert self.input_shape is not None
        for i in range(len(features)):
            if i == 0:
                self._conv_layers["Conv-0"] = torch.nn.Conv2d(
                    self.input_shape[0],
                    features[i],
                    _conv_kernel_size[0],
                    stride=_conv_stride[0],
                )
            else:
                self._conv_layers[f"Conv-{i}"] = torch.nn.Conv2d(
                    features[i - 1],
                    features[i],
                    _conv_kernel_size[i],
                    stride=_conv_stride[i],
                )

            # Each convolutional layer is followed by max-pooling layer and ReLu activation.
            self._conv_layers[f"MaxPool-{i}"] = torch.nn.MaxPool2d(
                kernel_size=max_kernel_size, stride=max_stride
            )
            self._conv_layers[f"ReLu-{i}"] = torch.nn.ReLU()

            if conv_do:
                self._conv_layers[f"DropOut-{i}"] = torch.nn.Dropout(p_conv_do)

        # Construct the convolutional network from the dict of layers.
        self.conv_net = torch.nn.Sequential(self._conv_layers)

        # Calculate the input of the Linear layer by sending some fake data through the network
        # and getting the shape of the output.
        out_shape = get_output_shape(self.conv_net, self.input_shape)

        if type(out_shape) is int:
            self.flattened_size = out_shape
        elif isinstance(out_shape, Iterable):
            # Calculate the flattened size of the output from the convolutional network.
            self.flattened_size = int(np.prod(list(out_shape)))

        # Constructs the fully connected layers determined by the number of input channels and the features of these.
        for i in range(len(fc_sizes)):
            if i == 0:
                self._fc_layers["Linear-0"] = torch.nn.Linear(
                    self.flattened_size, fc_sizes[i]
                )
            else:
                self._fc_layers[f"Linear-{i}"] = torch.nn.Linear(
                    fc_sizes[i - 1], fc_sizes[i]
                )

            # Each fully connected layer is followed by a ReLu activation.
            self._fc_layers[f"ReLu-{i}"] = torch.nn.ReLU()

            if fc_do:
                self._fc_layers[f"DropOut-{i}"] = torch.nn.Dropout(p_fc_do)

        # Add classification layer.
        self._fc_layers["Classification"] = torch.nn.Linear(
            fc_sizes[-1], self.n_classes
        )

        # Create fully connected network.
        self.fc_net = torch.nn.Sequential(self._fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the convolutional network and then the fully connected network.

        Can be called directly as a method (e.g. ``model.forward()``)
        or when data is parsed to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Tensor of the likelihoods the network places on the input ``x`` being of each class.
        """
        # Inputs the data into the convolutional network.
        conv_out = self.conv_net(x)

        # Output from convolutional network is flattened and input to the fully connected network for classification.
        z = self.fc_net(conv_out.view(-1, self.flattened_size))
        assert isinstance(z, Tensor)
        return z


class ResNet(MinervaModel, ABC):
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
        conv1 (torch.nn.Conv2d): Input convolutional layer of Conv1 input block to the network.
        bn1 (Module): Batch normalisation layer of the Conv1 input block to the network.
        relu (torch.nn.ReLU): Rectified Linear Unit (ReLU) activation layer to be used throughout ResNet.
        maxpool (torch.nn.MaxPool2d): 3x3 Max-pooling layer with stride 2 of the Conv1 input block to the network.
        layer1 (torch.nn.Sequential): Layer 1 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        layer2 (torch.nn.Sequential): Layer 2 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        layer3 (torch.nn.Sequential): Layer 3 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        layer4 (torch.nn.Sequential): Layer 4 of the :class:`ResNet` comprising number and type of blocks defined
            by ``layers``.
        avgpool (torch.nn.AdaptiveAvgPool2d): Global average pooling layer taking the output from the last block.
            Only initialised if ``encoder_on=False``.
        fc (torch.nn.Linear): Fully connected layer that takes the flattened output from average pooling
            to a classification output. Only initialised if ``encoder_on=False``.

    .. warning::
        Layers using :class:`BasicBlock` are not compatible with anything other than the default values for
        ``groups`` and ``width_per_group``.

    Args:
        block (BasicBlock or Bottleneck): Type of block operations to use throughout network.
        layers (list): Number of blocks in each of the 4 `layers`.
        in_channels (int): Optional; Number of channels (or bands) in the input imagery.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If ``True``, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Optional; Number of convolutions in grouped convolutions of Bottleneck Blocks.
            Not compatible with Basic Block!
        width_per_group (int): Optional; Modifies the number of feature maps in convolutional layers
            of Bottleneck Blocks. Not compatible with Basic Block!
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block.
            Typically, :class:`torch.nn.BatchNorm2d`.
        encoder (bool): Optional; Whether to initialise the :class:`ResNet` as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
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
            norm_layer = torch.nn.BatchNorm2d
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
        self.conv1 = torch.nn.Conv2d(
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
        self.relu = torch.nn.ReLU(inplace=True)

        # Adds the max pooling layer to complete the Conv1 layer.
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512 * block.expansion, n_classes)

        # Performs weight initialisation across network.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
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
    ) -> torch.nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
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

            except ValueError as err:
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

        return torch.nn.Sequential(*layers)

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

        if not self.encoder_on:
            x5 = self.avgpool(x4)
            x5 = torch.flatten(x5, 1)
            x5 = self.fc(x5)

            assert isinstance(x5, Tensor)
            return x5

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: If initialised as an encoder,
            returns a tuple of outputs from each `layer` 1-4. Else, returns :class:`Tensor` of the likelihoods the
            network places on the input `x` being of each class.
        """
        return self._forward_impl(x)


class DCN(MinervaModel, ABC):
    """Generic DCN defined by the FCN paper. Can construct the DCN32, DCN16 or DCN8 variants defined in the paper.

    Based on the example found here: https://github.com/haoran1062/FCN-pytorch/blob/master/FCN.py

    Attributes:
        variant (str): Defines which DCN variant this object is, altering the layers constructed
            and the computational graph. Will be either '32', '16' or '8'.
            See the FCN paper for details on these variants.
        n_classes (int): Number of classes in dataset. Defines number of output classification channels.
        relu (torch.nn.ReLU): Rectified Linear Unit (ReLU) activation layer to be used throughout the network.
        Conv1x1 (torch.nn.Conv2d): First Conv1x1 layer acting as input to the network from the final output of
            the encoder and common to all variants.
        bn1 (torch.nn.BatchNorm2d): First batch norm layer common to all variants that comes after Conv1x1.
        DC32 (torch.nn.ConvTranspose2d): De-convolutional layer with stride 32 for DCN32 variant.
        dbn32 (torch.nn.BatchNorm2d): Batch norm layer after DC32.
        Conv1x1_x3 (torch.nn.Conv2d): Conv1x1 layer acting as input to the network taking the output from the
            third layer from the ResNet encoder.
        DC2 (torch.nn.ConvTranspose2d): De-convolutional layer with stride 2 for DCN16 & DCN8 variants.
        dbn2 (torch.nn.BatchNorm2d): Batch norm layer after DC2.
        DC16 (torch.nn.ConvTranspose2d): De-convolutional layer with stride 16 for DCN16 variant.
        dbn16 (torch.nn.BatchNorm2d): Batch norm layer after DC16.
        Conv1x1_x2 (torch.nn.Conv2d): Conv1x1 layer acting as input to the network taking the output from the
            second layer from the ResNet encoder.
        DC4 (torch.nn.ConvTranspose2d): De-convolutional layer with stride 2 for DCN8 variant.
        dbn4 (torch.nn.BatchNorm2d): Batch norm layer after DC4.
        DC8 (torch.nn.ConvTranspose2d): De-convolutional layer with stride 8 for DCN8 variant.
        dbn8 (torch.nn.BatchNorm2d): Batch norm layer after DC8.

    Args:
        in_channel (int): Optional; Number of channels in the input layer of the network.
            Should match the number of output channels (likely feature maps) from the encoder.
        n_classes (int): Optional; Number of classes in dataset. Defines number of output classification channels.
        variant (str): Optional; Flag for which DCN variant to construct. Must be either '32', '16' or '8'.
            See the FCN paper for details on these variants.
    """

    def __init__(
        self,
        in_channel: int = 512,
        n_classes: int = 21,
        variant: Literal["32", "16", "8"] = "32",
    ) -> None:

        super(DCN, self).__init__(n_classes=n_classes)
        self.variant: Literal["32", "16", "8"] = variant

        assert type(self.n_classes) is int

        # Common to all variants.
        self.relu = torch.nn.ReLU(inplace=True)
        self.Conv1x1 = torch.nn.Conv2d(in_channel, self.n_classes, kernel_size=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(self.n_classes)

        if variant == "32":
            self.DC32 = torch.nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(64, 64),
                stride=(32, 32),
                dilation=1,
                padding=(16, 16),
            )
            self.DC32.weight.data = bilinear_init(self.n_classes, self.n_classes, 64)
            self.dbn32 = torch.nn.BatchNorm2d(self.n_classes)

        if variant in ("16", "8"):
            self.Conv1x1_x3 = torch.nn.Conv2d(
                int(in_channel / 2), self.n_classes, kernel_size=(1, 1)
            )
            self.DC2 = torch.nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(4, 4),
                stride=(2, 2),
                dilation=1,
                padding=(1, 1),
            )
            self.DC2.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn2 = torch.nn.BatchNorm2d(self.n_classes)

        if variant == "16":
            self.DC16 = torch.nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(32, 32),
                stride=(16, 16),
                dilation=1,
                padding=(8, 8),
            )
            self.DC16.weight.data = bilinear_init(self.n_classes, self.n_classes, 32)
            self.dbn16 = torch.nn.BatchNorm2d(self.n_classes)

        if variant == "8":
            self.Conv1x1_x2 = torch.nn.Conv2d(
                int(in_channel / 4), self.n_classes, kernel_size=(1, 1)
            )

            self.DC4 = torch.nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(4, 4),
                stride=(2, 2),
                dilation=1,
                padding=(1, 1),
            )
            self.DC4.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn4 = torch.nn.BatchNorm2d(self.n_classes)

            self.DC8 = torch.nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(16, 16),
                stride=(8, 8),
                dilation=1,
                padding=(4, 4),
            )
            self.DC8.weight.data = bilinear_init(self.n_classes, self.n_classes, 16)
            self.dbn8 = torch.nn.BatchNorm2d(self.n_classes)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Performs a forward pass of the decoder. Depending on DCN variant, will take multiple inputs
        throughout pass from the encoder.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): Input data to network.
                Should be from a backbone that supports output at multiple points e.g ResNet.

        Returns:
            Tensor segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.
        """
        # Unpack outputs from the ResNet layers.
        x4, x3, x2, *_ = x

        # All DCNs have a common 1x1 Conv input block.
        z = self.bn1(self.relu(self.Conv1x1(x4)))

        # If DCN32, forward pass through DC32 and DBN32 and return output.
        if self.variant == "32":
            z = self.dbn32(self.relu(self.DC32(z)))
            assert isinstance(z, Tensor)
            return z

        # Common Conv1x1 layer to DCN16 & DCN8.
        x3 = self.bn1(self.relu(self.Conv1x1_x3(x3)))
        z = self.dbn2(self.relu(self.DC2(z)))

        z = z + x3

        # If DCN16, forward pass through DCN16 and DBN16 and return output.
        if self.variant == "16":
            z = self.dbn16(self.relu(self.DC16(z)))
            assert isinstance(z, Tensor)
            return z

        # If DCN8, continue through remaining layers to output.
        elif self.variant == "8":
            x2 = self.bn1(self.relu(self.Conv1x1_x2(x2)))
            z = self.dbn4(self.relu(self.DC4(z)))

            z = z + x2

            z = self.dbn8(self.relu(self.DC8(z)))

            assert isinstance(z, Tensor)
            return z


class ResNet18(MinervaModel, ABC):
    """ResNet18 modified from source to have customisable number of input channels and to be used as a backbone
    by stripping classification layers away.

    Attributes:
        network (ResNet): ResNet18 network.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If True, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block. Typically torch.nn.BatchNorm2d.
        encoder (bool): Optional; Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
    """

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        zero_init_residual: bool = False,
        norm_layer: Optional[Callable[..., Module]] = None,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        encoder: bool = False,
    ) -> None:

        super(ResNet18, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        self.network = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            in_channels=input_size[0],
            n_classes=n_classes,
            zero_init_residual=zero_init_residual,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            encoder=encoder,
        )

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: If initialised as an encoder,
            returns a tuple of outputs from each `layer` 1-4. Else, returns :class:`Tensor` of the likelihoods the
            network places on the input `x` being of each class.
        """
        z: Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = self.network(
            x
        )
        if isinstance(z, Tensor):
            return z
        elif isinstance(z, tuple):
            assert all(isinstance(n, Tensor) for n in z)
            return z


class ResNet34(MinervaModel, ABC):
    """ResNet34 modified from source to have customisable number of input channels and to be used as a backbone
        by stripping classification layers away.

    Attributes:
        network (ResNet): ResNet34 network.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If True, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block. Typically torch.nn.BatchNorm2d.
        encoder (bool): Optional; Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
    """

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        zero_init_residual: bool = False,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        encoder: bool = False,
    ) -> None:

        super(ResNet34, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        self.network = ResNet(
            BasicBlock,
            [3, 4, 6, 3],
            in_channels=input_size[0],
            n_classes=n_classes,
            zero_init_residual=zero_init_residual,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            encoder=encoder,
        )

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: If initialised as an encoder,
            returns a tuple of outputs from each `layer` 1-4. Else, returns :class:`Tensor` of the likelihoods the
            network places on the input `x` being of each class.
        """
        z: Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = self.network(
            x
        )
        if isinstance(z, Tensor):
            return z
        elif isinstance(z, tuple):
            assert all(isinstance(n, Tensor) for n in z)
            return z


class ResNet50(MinervaModel, ABC):
    """ResNet50 modified from source to have customisable number of input channels and to be used as a backbone
            by stripping classification layers away.

    Attributes:
        network (ResNet): ResNet50 network.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If True, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Number of convolutions in grouped convolutions of Bottleneck Blocks.
        width_per_group (int): Modifies the number of feature maps in convolutional layers of Bottleneck Blocks.
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block. Typically torch.nn.BatchNorm2d.
        encoder (bool): Optional; Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
    """

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        encoder: bool = False,
    ) -> None:

        super(ResNet50, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        self.network = ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            in_channels=input_size[0],
            n_classes=n_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            encoder=encoder,
        )

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Overwrites :class:`MinervaModel` abstract method.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: If initialised as an encoder,
            returns a tuple of outputs from each `layer` 1-4. Else, returns :class:`Tensor` of the likelihoods the
            network places on the input `x` being of each class.
        """
        z: Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = self.network(
            x
        )
        if isinstance(z, Tensor):
            return z
        elif isinstance(z, tuple):
            assert all(isinstance(n, Tensor) for n in z)
            return z


class ResNet101(MinervaModel, ABC):
    """ResNet101 modified from source to have customisable number of input channels and to be used as a backbone
            by stripping classification layers away.

    Attributes:
        network (ResNet): ResNet50 network.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If True, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Number of convolutions in grouped convolutions of Bottleneck Blocks.
        width_per_group (int): Modifies the number of feature maps in convolutional layers of Bottleneck Blocks.
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block. Typically torch.nn.BatchNorm2d.
        encoder (bool): Optional; Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
    """

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        encoder: bool = False,
    ) -> None:

        super(ResNet101, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        self.network = ResNet(
            Bottleneck,
            [3, 4, 23, 3],
            in_channels=input_size[0],
            n_classes=n_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            encoder=encoder,
        )

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Overwrites :class:`MinervaModel` abstract method.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: If initialised as an encoder,
            returns a tuple of outputs from each `layer` 1-4. Else, returns :class:`Tensor` of the likelihoods the
            network places on the input `x` being of each class.
        """
        z: Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = self.network(
            x
        )
        if isinstance(z, Tensor):
            return z
        elif isinstance(z, tuple):
            assert all(isinstance(n, Tensor) for n in z)
            return z


class ResNet152(MinervaModel, ABC):
    """ResNet152 modified from source to have customisable number of input channels and to be used as a backbone
            by stripping classification layers away.

    Attributes:
        network (ResNet): ResNet50 network.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If True, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Number of convolutions in grouped convolutions of Bottleneck Blocks.
        width_per_group (int): Modifies the number of feature maps in convolutional layers of Bottleneck Blocks.
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block. Typically torch.nn.BatchNorm2d.
        encoder (bool): Optional; Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
    """

    def __init__(
        self,
        criterion: Optional[Any] = None,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        encoder: bool = False,
    ) -> None:

        super(ResNet152, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        self.network = ResNet(
            Bottleneck,
            [3, 8, 36, 3],
            in_channels=input_size[0],
            n_classes=n_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            encoder=encoder,
        )

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Performs a forward pass of the :class:`ResNet`.

        Overwrites :class:`MinervaModel` abstract method.

        Can be called directly as a method (e.g. :func:`model.forward`) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (Tensor): Input data to network.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: If initialised as an encoder,
            returns a tuple of outputs from each `layer` 1-4. Else, returns :class:`Tensor` of the likelihoods the
            network places on the input `x` being of each class.
        """
        z: Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = self.network(
            x
        )
        if isinstance(z, Tensor):
            return z
        elif isinstance(z, tuple):
            assert all(isinstance(n, Tensor) for n in z)
            return z


class _FCN(MinervaModel, ABC):
    """Base Fully Convolutional Network (FCN) class to be subclassed by FCN variants described in the FCN paper.

    Subclasses MinervaModel.

    Attributes:
        backbone (Module): Backbone of the FCN that takes the imagery input and
            extracts learned representations.
        decoder (Module): Decoder that takes the learned representations from the backbone encoder
            and de-convolves to output a classification segmentation mask.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_name (str): Optional; Name of the backbone within this module to use for the FCN.
        decoder_name (str): Optional; Name of the decoder class to use for the FCN. Must be either 'DCN' or 'Decoder'.
        decoder_variant (str): Optional; Flag for which DCN variant to construct. Must be either '32', '16' or '8'.
            See the FCN paper for details on these variants.
        batch_size (int): Optional; Number of samples in each batch supplied to the network.
            Only needed for Decoder, not DCN.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        backbone_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_name: str = "ResNet18",
        decoder_variant: Literal["32", "16", "8"] = "32",
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super(_FCN, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        # Initialises the selected Minerva backbone.
        self.backbone: MinervaModel = globals()[backbone_name](
            input_size=input_size, n_classes=n_classes, encoder=True, **backbone_kwargs
        )

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:
            self.backbone.load_state_dict(torch.load(backbone_weight_path))

            # Freezes the weights of backbone to avoid end-to-end training.
            if freeze_backbone:
                self.backbone.requires_grad_(False)

        # Determines the output shape of the backbone so the correct input shape is known
        # for the proceeding layers of the network.
        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)
        self.decoder = DCN(
            in_channel=backbone_out_shape[0],
            n_classes=n_classes,
            variant=decoder_variant,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the FCN by using the forward methods of the backbone and
        feeding its output into the forward for the decoder.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (FloatTensor): Input data to network.

        Returns:
            z (Tensor): segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.
        """
        z = self.backbone(x)
        z = self.decoder(z)

        assert isinstance(z, Tensor)
        return z


class FCN32ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN32ResNet18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            decoder_variant="32",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN32ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN32ResNet34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet34",
            decoder_variant="32",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN32ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN32ResNet50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet50",
            decoder_variant="32",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN16ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN16ResNet18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            decoder_variant="16",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN16ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN16ResNet34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet34",
            decoder_variant="16",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN16ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet50 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN16ResNet50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet50",
            decoder_variant="16",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet34",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet50 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet50",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet101(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet101 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet101, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet101",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet152(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet152 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet152, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet152",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class _Siam(MinervaModel, MinervaBackbone):
    """Base Siam class to be subclassed by Siam variants.

    Subclasses MinervaModel.

    Attributes:
        backbone (Module): Backbone of Siam that takes the imagery input and
            extracts learned representations.
        proj_head (Module): Projection head that takes the learned representations from the backbone encoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        backbone_name (str): Optional; Name of the backbone within this module to use.
        backbone_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 128,
        backbone_name: str = "ResNet18",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super(_Siam, self).__init__(criterion=criterion, input_shape=input_size)

        self.backbone: MinervaModel = globals()[backbone_name](
            input_size=input_size, encoder=True, **backbone_kwargs
        )

        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)

        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(np.prod(backbone_out_shape), 512, bias=False),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, feature_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of Siam by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).
        """
        f_a: Tensor = torch.flatten(self.backbone(x[0])[0], start_dim=1)
        f_b: Tensor = torch.flatten(self.backbone(x[1])[0], start_dim=1)

        g_a: Tensor = self.proj_head(f_a)
        g_b: Tensor = self.proj_head(f_b)

        z = torch.cat([g_a, g_b], dim=0)

        assert isinstance(z, Tensor)

        return z, g_a, g_b, f_a, f_b

    def step(self, x: Tensor, *, train: bool = False) -> Tuple[_Loss, Tensor]:
        """Overwrites :class:`MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If ``self.optimiser`` is None.
            NotImplementedError: If ``self.criterion`` is None.

        Args:
            x (FloatTensor): Batch of input data to network.
            train (bool): Sets whether this shall be a training step or not. True for training step which will then
                clear the optimiser, and perform a backward pass of the network then update the optimiser.
                If False for a validation or testing step, these actions are not taken.

        Returns:
            Tuple[_Loss, Tensor]: Loss computed by the loss function and a :class:`Tensor`
            with both projection's logits.
        """

        if self.optimiser is None:
            raise NotImplementedError("Optimiser has not been set!")

        if self.criterion is None:
            raise NotImplementedError("Criterion has not been set!")

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        # Forward pass.
        z, z_a, z_b, _, _ = self.forward(x)

        # Compute Loss.
        loss: _Loss = self.criterion(z_a, z_b)

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, z


class Siam18(_Siam):
    """Siam network using a ResNet18 backbone.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 128,
        **resnet_kwargs,
    ) -> None:

        super(Siam18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet18",
            backbone_kwargs=resnet_kwargs,
        )


class Siam34(_Siam):
    """Siam network using a ResNet32 backbone.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 128,
        **resnet_kwargs,
    ) -> None:

        super(Siam34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet34",
            backbone_kwargs=resnet_kwargs,
        )


class Siam50(_Siam):
    """Siam network using a ResNet50 backbone.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 128,
        **resnet_kwargs,
    ) -> None:

        super(Siam50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet50",
            backbone_kwargs=resnet_kwargs,
        )


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_output_shape(
    model: Module,
    image_dim: Union[List[int], Tuple[int, ...]],
    sample_pairs: bool = False,
) -> Union[int, Iterable[int]]:
    """Gets the output shape of a model.

    Args:
        model (Module): Model for which the shape of the output needs to be found.
        image_dim (list[int] or tuple[int]): Expected shape of the input data to the model.

    Returns:
        The shape of the output data from the model.
    """
    try:
        if len(image_dim) == 1:
            image_dim = image_dim[0]
    except TypeError:
        if not hasattr(image_dim, "__len__"):
            pass
        else:
            raise TypeError

    if not hasattr(image_dim, "__len__"):
        random_input = torch.rand([4, image_dim])
    elif sample_pairs:
        random_input = torch.rand([2, 4, *image_dim])
    else:
        random_input = torch.rand([4, *image_dim])

    output: Tensor = model(random_input)

    if len(output[0].data.shape) == 1:
        return output[0].data.shape[0]

    else:
        return output[0].data.shape[1:]


def bilinear_init(in_channels: int, out_channels: int, kernel_size: int) -> Tensor:
    """Constructs the weights for the bi-linear interpolation kernel for use in transpose convolutional layers.

    Source: https://github.com/haoran1062/FCN-pytorch/blob/master/FCN.py

    Args:
        in_channels (int): Number of input channels to the layer.
        out_channels (int): Number of output channels from the layer.
        kernel_size (int): Size of the (square) kernel.

    Returns:
        Tensor of the initialised bi-linear interpolated weights for the transpose convolutional layer's kernels.
    """
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = int(factor - 0.5)

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype="float32"
    )
    weight[range(in_channels), range(out_channels), :, :] = filt

    weights = torch.from_numpy(weight)
    assert isinstance(weights, Tensor)
    return weights
