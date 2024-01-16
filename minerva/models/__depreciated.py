# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
"""Module for redundant model classes."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch.nn.modules as nn
from torch import Tensor
from torch.nn.modules import Module

from minerva.utils.utils import check_len

from .core import MinervaModel, get_output_shape


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MLP(MinervaModel):
    """Simple class to construct a Multi-Layer Perceptron (MLP).

    Inherits from :class:`~torch.nn.Module` and :class:`MinervaModel`. Designed for use with PyTorch functionality.

    Should be used in tandem with :class:`~trainer.Trainer`.

    Attributes:
        input_size (int): Size of the input vector to the network.
        output_size (int): Size of the output vector of the network.
        hidden_sizes (tuple[int] | list[int]): Series of values for the size of each hidden layers within the network.
            Also determines the number of layers other than the required input and output layers.
        network (torch.nn.Sequential): The actual neural network of the model.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (int): Optional; Size of the input vector to the network.
        n_classes (int): Optional; Number of classes in input data.
            Determines the size of the output vector of the network.
        hidden_sizes (tuple[int] | list[int]): Optional; Series of values for the size of each hidden layers
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
            criterion=criterion, input_size=(input_size,), n_classes=n_classes
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
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor. Tensor of the likelihoods the network places on the input ``x`` being of each class.
        """
        z = self.network(x)
        assert isinstance(z, Tensor)
        return z


class CNN(MinervaModel):
    """Simple class to construct a Convolutional Neural Network (CNN).

    Inherits from :class:`~torch.nn.Module` and :class:`MinervaModel`. Designed for use with :mod:`torch` functionality.

    Should be used in tandem with :class:`~trainer.Trainer`.

    Attributes:
        flattened_size (int): Length of the vector resulting from the flattening of the output from the convolutional
            network.
        conv_net (torch.nn.Sequential): Convolutional network of the model.
        fc_net (torch.nn.Sequential): Fully connected network of the model.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int] | list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in input data.
        features (tuple[int] | list[int]): Optional; Series of values defining the number of feature maps.
            The length of the list is also used to determine the number of convolutional layers
            in ``conv_net``.
        conv_kernel_size (int | tuple[int, ...]): Optional; Size of all convolutional kernels
            for all channels and layers.
        conv_stride (int | tuple[int, ...]): Optional; Size of all convolutional stride lengths
            for all channels and layers.
        max_kernel_size (int | tuple[int, ...]): Optional; Size of all max-pooling kernels
            for all channels and layers.
        max_stride (int | tuple[int, ...]): Optional; Size of all max-pooling stride lengths
            for all channels and layers.
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
            criterion=criterion, input_size=input_size, n_classes=n_classes
        )

        self._conv_layers: OrderedDict[str, Module] = OrderedDict()
        self._fc_layers: OrderedDict[str, Module] = OrderedDict()

        # Checks that the kernel sizes and strides match the number of layers defined by features.
        _conv_kernel_size: Sequence[int] = check_len(conv_kernel_size, features)
        _conv_stride: Sequence[int] = check_len(conv_stride, features)

        # Constructs the convolutional layers determined by the number of input channels and the features of these.
        assert self.input_size is not None
        for i in range(len(features)):
            if i == 0:
                self._conv_layers["Conv-0"] = nn.Conv2d(
                    self.input_size[0],
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
        self.flattened_size = int(
            np.prod(get_output_shape(self.conv_net, self.input_size))
        )

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
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor: Tensor of the likelihoods the network places on the input ``x`` being of each class.
        """
        # Inputs the data into the convolutional network.
        conv_out = self.conv_net(x)

        # Output from convolutional network is flattened and input to the fully connected network for classification.
        z = self.fc_net(conv_out.view(-1, self.flattened_size))
        assert isinstance(z, Tensor)
        return z
