"""Module containing neural network model classes

    Copyright (C) 2021 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker
Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com
Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd

TODO:
    * Complete MLP documentation
    * Add CNN model
    
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import torch
from abc import ABC
import numpy as np
from collections import OrderedDict


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MLP(torch.nn.Module, ABC):
    """
    Simple class to construct a Multi-Layer Perceptron (MLP)
    """

    def __init__(self, criterion, input_size=288, n_classes=8, hidden_sizes=(256, 144)):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = n_classes
        self.hidden_sizes = hidden_sizes
        self.layers = OrderedDict()

        for i in range(len(hidden_sizes)):
            if i is 0:
                self.layers['Linear-0'] = torch.nn.Linear(input_size, hidden_sizes[i])
            elif i > 0:
                self.layers['Linear-{}'.format(i)] = torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            else:
                print('EXCEPTION on Layer {}'.format(i))
            self.layers['ReLu-{}'.format(i)] = torch.nn.ReLU()

        self.layers['Classification'] = torch.nn.Linear(hidden_sizes[-1], n_classes)

        self.network = torch.nn.Sequential(self.layers)

        self.criterion = criterion
        self.optimiser = None

    def set_optimiser(self, optimiser):
        self.optimiser = optimiser

    def forward(self, x):
        """Performs a forward pass of the network

        Args:
            x (torch.Tensor): Data

        Returns:
            y (torch.Tensor): Label
        """
        return self.network(x)

    def step(self, x, y, train):
        if train:
            self.optimiser.zero_grad()

        # Forward pass
        z = self.forward(x)

        # Compute Loss
        loss = self.criterion(z, y)

        if train:
            # Backward pass
            loss.backward()
            self.optimiser.step()

        return loss, z

    def training_step(self, x, y):
        return self.step(x, y, True)

    def validation_step(self, x, y):
        return self.step(x, y, False)

    def testing_step(self, x, y):
        return self.step(x, y, False)


class CNN(torch.nn.Module, ABC):
    """Simple class to construct a Convolutional Neural Network (CNN)
    """

    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes=8, scaling=(2, 1, 1),
                 conv_kernel_size: tuple = 3, conv_stride: tuple = 1, max_kernel_size=2, max_stride=2):
        """

        Args:
            criterion:
            input_shape:
            n_classes:
            scaling:
            conv_kernel_size (int or tuple):
            conv_stride (int or tuple):
            max_kernel_size:
            max_stride:
        """
        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.conv_layers = OrderedDict()
        self.fc_layers = OrderedDict()

        for i in range(len(scaling)):
            if i is 0:
                self.conv_layers['Conv-0'] = torch.nn.Conv2d(input_shape[0], input_shape[0] * scaling[i],
                                                             conv_kernel_size, stride=conv_stride)
            elif i > 0:
                self.conv_layers['Conv-{}'.format(i)] = torch.nn.Conv2d(scaling[i - 1] * input_shape[0],
                                                                        scaling[i] * input_shape[0],
                                                                        conv_kernel_size, stride=conv_stride)
            else:
                print('EXCEPTION on Layer {}'.format(i))
            self.conv_layers['MaxPool-{}'.format(i)] = torch.nn.MaxPool2d(kernel_size=max_kernel_size,
                                                                          stride=max_stride)
            self.conv_layers['ReLu-{}'.format(i)] = torch.nn.ReLU()

        self.conv_net = torch.nn.Sequential(self.conv_layers)

        # Calculate the input of the Linear layer
        out_shape = []
        for i in range(len(scaling)):
            if i is 0:
                out_shape = get_output_shape(self.conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self.conv_layers['Conv-{}'.format(i)], input_shape))
            if i > 0:
                out_shape = get_output_shape(self.conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self.conv_layers['Conv-{}'.format(i)], out_shape))

        self.flattened_size = int(np.prod(list(out_shape)))

        #self.fc_layers['Flatten'] = torch.nn.Linear(self.flattened_size, self.flattened_size)
        self.fc_layers['Classification'] = torch.nn.Linear(self.flattened_size, self.n_classes)

        self.fc_net = torch.nn.Sequential(self.fc_layers)

        self.criterion = criterion
        self.optimiser = None

    def set_optimiser(self, optimiser):
        self.optimiser = optimiser

    def forward(self, x):
        """Performs a forward pass of the network

        Args:
            x (torch.Tensor): Data

        Returns:
            y (torch.Tensor): Label
        """
        conv_out = self.conv_net(x)
        return self.fc_net(conv_out.view(-1, self.flattened_size))

    def step(self, x, y, train):
        if train:
            self.optimiser.zero_grad()

        # Forward pass
        z = self.forward(x)

        # Compute Loss
        loss = self.criterion(z, y)

        if train:
            # Backward pass
            loss.backward()
            self.optimiser.step()

        return loss, z

    def training_step(self, x, y):
        return self.step(x, y, True)

    def validation_step(self, x, y):
        return self.step(x, y, False)

    def testing_step(self, x, y):
        return self.step(x, y, False)


def get_output_shape(model, image_dim):
    return model(torch.rand([1, *image_dim])).data.shape[1:]
