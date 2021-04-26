"""models

Module containing neural network model classes

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
