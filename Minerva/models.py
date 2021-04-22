"""models

Module containing neural network model classes

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