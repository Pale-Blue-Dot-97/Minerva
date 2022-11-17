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
import os
from nptyping import NDArray
import numpy as np
import torch
from torch import Tensor
import torch.nn.modules as nn
from torch.nn.modules import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torchvision.models._api import WeightsEnum
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

        self._layers: OrderedDict[str, Module] = OrderedDict()

        # Constructs layers of the network based on the input size, the hidden sizes and the number of classes.
        for i in range(len(hidden_sizes)):
            if i == 0:
                self._layers["Linear-0"] = nn.Linear(input_size, hidden_sizes[i])
            else:
                self._layers[f"Linear-{i}"] = nn.Linear(
                    hidden_sizes[i - 1], hidden_sizes[i]
                )

            # Adds ReLu activation after every linear layer.
            self._layers[f"ReLu-{i}"] = nn.ReLU()

        # Adds the final classification layer.
        self._layers["Classification"] = nn.Linear(hidden_sizes[-1], n_classes)

        # Constructs network from the OrderedDict of layers
        self.network = nn.Sequential(self._layers)

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

        self._conv_layers: OrderedDict[str, Module] = OrderedDict()
        self._fc_layers: OrderedDict[str, Module] = OrderedDict()

        # Checks that the kernel sizes and strides match the number of layers defined by features.
        _conv_kernel_size: Sequence[int] = utils.check_len(conv_kernel_size, features)
        _conv_stride: Sequence[int] = utils.check_len(conv_stride, features)

        # Constructs the convolutional layers determined by the number of input channels and the features of these.
        assert self.input_shape is not None
        for i in range(len(features)):
            if i == 0:
                self._conv_layers["Conv-0"] = nn.Conv2d(
                    self.input_shape[0],
                    features[i],
                    _conv_kernel_size[0],
                    stride=_conv_stride[0],
                )
            else:
                self._conv_layers[f"Conv-{i}"] = nn.Conv2d(
                    features[i - 1],
                    features[i],
                    _conv_kernel_size[i],
                    stride=_conv_stride[i],
                )

            # Each convolutional layer is followed by max-pooling layer and ReLu activation.
            self._conv_layers[f"MaxPool-{i}"] = nn.MaxPool2d(
                kernel_size=max_kernel_size, stride=max_stride
            )
            self._conv_layers[f"ReLu-{i}"] = nn.ReLU()

            if conv_do:
                self._conv_layers[f"DropOut-{i}"] = nn.Dropout(p_conv_do)

        # Construct the convolutional network from the dict of layers.
        self.conv_net = nn.Sequential(self._conv_layers)

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
                self._fc_layers["Linear-0"] = nn.Linear(
                    self.flattened_size, fc_sizes[i]
                )
            else:
                self._fc_layers[f"Linear-{i}"] = nn.Linear(fc_sizes[i - 1], fc_sizes[i])

            # Each fully connected layer is followed by a ReLu activation.
            self._fc_layers[f"ReLu-{i}"] = nn.ReLU()

            if fc_do:
                self._fc_layers[f"DropOut-{i}"] = nn.Dropout(p_fc_do)

        # Add classification layer.
        assert self.n_classes is not None
        self._fc_layers["Classification"] = nn.Linear(fc_sizes[-1], self.n_classes)

        # Create fully connected network.
        self.fc_net = nn.Sequential(self._fc_layers)

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


class _SimCLR(MinervaBackbone):
    """Base SimCLR class to be subclassed by SimCLR variants.

    Subclasses MinervaModel.

    Attributes:
        backbone (Module): Backbone of SimCLR that takes the imagery input and
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

        super(_SimCLR, self).__init__(criterion=criterion, input_shape=input_size)

        self.backbone: MinervaModel = globals()[backbone_name](
            input_size=input_size, encoder=True, **backbone_kwargs
        )

        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)

        self.proj_head = nn.Sequential(
            nn.Linear(np.prod(backbone_out_shape), 512, bias=False),  # type: ignore[arg-type]
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of SimCLR by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).
        """
        f_a: Tensor = torch.flatten(self.backbone(x[0])[0], start_dim=1)  # type: ignore[attr-defined]
        f_b: Tensor = torch.flatten(self.backbone(x[1])[0], start_dim=1)  # type: ignore[attr-defined]

        g_a: Tensor = self.proj_head(f_a)
        g_b: Tensor = self.proj_head(f_b)

        z = torch.cat([g_a, g_b], dim=0)  # type: ignore[attr-defined]

        assert isinstance(z, Tensor)

        return z, g_a, g_b, f_a, f_b

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If ``self.optimiser`` is None.
            NotImplementedError: If ``self.criterion`` is None.

        Args:
            x (Tensor): Batch of input data to network.
            train (bool): Sets whether this shall be a training step or not. True for training step which will then
                clear the optimiser, and perform a backward pass of the network then update the optimiser.
                If False for a validation or testing step, these actions are not taken.

        Returns:
            Tuple[Tensor, Tensor]: Loss computed by the loss function and a :class:`Tensor`
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
        loss: Tensor = self.criterion(z_a, z_b)

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, z


class SimCLR18(_SimCLR):
    """SimCLR network using a ResNet18 backbone.

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

        super(SimCLR18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet18",
            backbone_kwargs=resnet_kwargs,
        )


class SimCLR34(_SimCLR):
    """SimCLR network using a ResNet32 backbone.

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

        super(SimCLR34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet34",
            backbone_kwargs=resnet_kwargs,
        )


class SimCLR50(_SimCLR):
    """SimCLR network using a ResNet50 backbone.

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

        super(SimCLR50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet50",
            backbone_kwargs=resnet_kwargs,
        )


class _SimSiam(MinervaBackbone):
    """Base SimSiam class to be subclassed by SimSiam variants.

    Subclasses MinervaModel.

    Attributes:
        backbone (Module): Backbone of SimSiam that takes the imagery input and
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
        feature_dim: int = 2048,
        pred_dim: int = 512,
        backbone_name: str = "ResNet18",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super(_SimSiam, self).__init__(criterion=criterion, input_shape=input_size)

        self.backbone: MinervaModel = globals()[backbone_name](
            input_size=input_size, encoder=True, **backbone_kwargs
        )

        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)

        prev_dim = np.prod(backbone_out_shape)

        self.proj_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),  # type: ignore[arg-type]
            nn.BatchNorm1d(prev_dim),  # type: ignore[arg-type]
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),  # type: ignore[arg-type]
            nn.BatchNorm1d(prev_dim),  # type: ignore[arg-type]
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(prev_dim, feature_dim, bias=False),  # type: ignore[arg-type]
            nn.BatchNorm1d(feature_dim, affine=False),
        )  # output layer
        # self.proj_head[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, feature_dim),
        )  # output layer

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of SimCLR by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).
        """
        z_a: Tensor = self.proj_head(torch.flatten(self.backbone(x[0])[0], start_dim=1))  # type: ignore[attr-defined]
        z_b: Tensor = self.proj_head(torch.flatten(self.backbone(x[1])[0], start_dim=1))  # type: ignore[attr-defined]

        p_a: Tensor = self.predictor(z_a)
        p_b: Tensor = self.predictor(z_b)

        p = torch.cat([p_a, p_b], dim=0)  # type: ignore[attr-defined]

        assert isinstance(p, Tensor)

        return p, p_a, p_b, z_a.detach(), z_b.detach()

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If ``self.optimiser`` is None.
            NotImplementedError: If ``self.criterion`` is None.

        Args:
            x (Tensor): Batch of input data to network.
            train (bool): Sets whether this shall be a training step or not. True for training step which will then
                clear the optimiser, and perform a backward pass of the network then update the optimiser.
                If False for a validation or testing step, these actions are not taken.

        Returns:
            Tuple[Tensor, Tensor]: Loss computed by the loss function and a :class:`Tensor`
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
        p, p_a, p_b, z_a, z_b = self.forward(x)

        # Compute Loss.
        loss: Tensor = 0.5 * (self.criterion(z_a, p_b) + self.criterion(z_b, p_a))

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, p


class SimSiam18(_SimSiam):
    """SimSiam network using a ResNet18 backbone.

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

        super(SimSiam18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet18",
            backbone_kwargs=resnet_kwargs,
        )


class SimSiam34(_SimSiam):
    """SimSiam network using a ResNet32 backbone.

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

        super(SimSiam34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet34",
            backbone_kwargs=resnet_kwargs,
        )


class SimSiam50(_SimSiam):
    """SimSiam network using a ResNet50 backbone.

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

        super(SimSiam50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            feature_dim=feature_dim,
            backbone_name="ResNet50",
            backbone_kwargs=resnet_kwargs,
        )
