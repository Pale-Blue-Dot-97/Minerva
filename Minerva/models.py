"""Module containing neural network model classes.

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
    * Add more functionality to CNN inputs
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
    """Simple class to construct a Multi-Layer Perceptron (MLP).

    Inherits from torch.nn.Module. Designed for use with PyTorch functionality.

    Should be used in tandem with Trainer.

    Attributes:
        input_size (int): Size of the input vector to the network.
        output_size (int): Size of the output vector of the network.
        hidden_sizes (tuple[int] or list[int]): Size of the hidden layers within the network.
            Can be a tuple[int] or list[int] of values that will also determine the number of layers other than
            the required input and output layers.
        network (torch.nn.Sequential): The actual neural network of the model.
        criterion: PyTorch loss function model will use.
        optimiser: PyTorch optimiser model will use. Initialised as None. Must be set using set_optimiser.
    """

    def __init__(self, criterion, input_size: int = 288, n_classes: int = 8, hidden_sizes: tuple = (256, 144)):
        """Initialises instance of MLP model.

        Args:
            criterion: PyTorch loss function model will use.
            input_size (int): Optional; Size of the input vector to the network.
            n_classes (int): Optional; Number of classes in input data.
                Determines the size of the output vector of the network.
            hidden_sizes (tuple[int] or list[int]): Optional; Size of the hidden layers within the network.
                Can be a tuple[int] or list[int] of values that will also determine the number of layers other than
                the required input and output layers.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = n_classes
        self.hidden_sizes = hidden_sizes
        self._layers = OrderedDict()

        # Constructs layers of the network based on the input size, the hidden sizes and the number of classes.
        for i in range(len(hidden_sizes)):
            if i is 0:
                self._layers['Linear-0'] = torch.nn.Linear(input_size, hidden_sizes[i])
            elif i > 0:
                self._layers['Linear-{}'.format(i)] = torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            else:
                print('EXCEPTION on Layer {}'.format(i))

            # Adds ReLu activation after every linear layer
            self._layers['ReLu-{}'.format(i)] = torch.nn.ReLU()

        # Adds the final classification layer
        self._layers['Classification'] = torch.nn.Linear(hidden_sizes[-1], n_classes)

        # Constructs network from the OrderedDict of layers
        self.network = torch.nn.Sequential(self.layers)

        # Sets loss function
        self.criterion = criterion

        # Optimiser initialised as None as the model parameters created by its init is required to init a
        # torch optimiser. The optimiser MUST be set by calling set_optimiser before the model can be trained.
        self.optimiser = None

    def set_optimiser(self, optimiser):
        """Sets the optimiser used by the model.

        Must be called after initialising a model and supplied with a PyTorch optimiser using this model's parameters.

        Args:
            optimiser: PyTorch optimiser model will use, initialised with this model's parameters.
        """
        self.optimiser = optimiser

    def forward(self, x: torch.FloatTensor):
        """Performs a forward pass of the network.

        Can be called directly as a method of MLP (e.g. model.forward()) or when data is parsed to MLP (e.g. model()).

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        return self.network(x)

    def step(self, x, y, train: bool):
        """Generic step of model fitting using a batch of data.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.
            train (bool): Sets whether this shall be a training step or not. True for training step which will then
                clear the optimiser, and perform a backward pass of the network then update the optimiser.
                If False for a validation or testing step, these actions are not taken.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        # Forward pass.
        z = self.forward(x)

        # Compute Loss.
        loss = self.criterion(z, y)

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, z

    def training_step(self, x, y):
        """Calls step with train=True to perform a training step. See step for more details.

        Designed to be compatible with Trainer and future compatibility with PyTorchLightning.
        Hence the resulting `boilerplate' of this method and validation_step and testing_step.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        return self.step(x, y, True)

    def validation_step(self, x, y):
        """Calls step with train=False to perform a validation step. See step for more details.

        Designed to be compatible with Trainer and future compatibility with PyTorchLightning.
        Hence the resulting `boilerplate' of this method and training_step and testing_step.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        return self.step(x, y, False)

    def testing_step(self, x, y):
        """Calls step with train=False to perform a testing step. See step for more details.

        Designed to be compatible with Trainer and future compatibility with PyTorchLightning.
        Hence the resulting `boilerplate' of this method and validation_step and training_step.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        return self.step(x, y, False)


class CNN(torch.nn.Module, ABC):
    """Simple class to construct a Convolutional Neural Network (CNN).

    Inherits from torch.nn.Module. Designed for use with PyTorch functionality.

    Should be used in tandem with Trainer.

    Attributes:
        input_shape (tuple[int, int, int] or list[int, int, int]): Defines the shape of the input data in order of
            number of channels, image width, image height.
        n_classes (int): Number of classes in input data.
        flattened_size (int): Length of the vector resulting from the flattening of the output from the convolutional
            network.
        conv_net (torch.nn.Sequential): Convolutional network of the model.
        fc_net (torch.nn.Sequential): Fully connected network of the model.
        criterion: PyTorch loss function model will use.
        optimiser: PyTorch optimiser model will use. Initialised as None. Must be set using set_optimiser.
    """

    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes: int = 8, scaling=(2, 1, 1),
                 conv_kernel_size: tuple = 3, conv_stride: tuple = 1, max_kernel_size: int = 2, max_stride: int = 2):
        """Initialises an instance of CNN.

        Args:
            criterion: PyTorch loss function model will use.
            input_shape (tuple[int, int, int] or list[int, int, int]): Optional; Defines the shape of the input data in
                order of number of channels, image width, image height.
            n_classes (int): Optional; Number of classes in input data.
            scaling (tuple[int] or list[int]): Optional; Series of values determining the factor by which the number of
                channels in the input is scaled up by to determine the number of feature maps. The number of entries is
                also used to determine the number of convolutional layers in conv_net.
            conv_kernel_size (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all convolutional kernels for all channels and layers.
            conv_stride (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all convolutional stride lengths for all channels and layers.
            max_kernel_size (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all max-pooling kernels for all channels and layers.
            max_stride (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all max-pooling stride lengths for all channels and layers.
        """
        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes
        self._conv_layers = OrderedDict()
        self._fc_layers = OrderedDict()

        # Constructs the convolutional layers determined by the number of input channels and the scaling of these.
        for i in range(len(scaling)):
            if i is 0:
                self._conv_layers['Conv-0'] = torch.nn.Conv2d(input_shape[0], input_shape[0] * scaling[i],
                                                              conv_kernel_size, stride=conv_stride)
            elif i > 0:
                self._conv_layers['Conv-{}'.format(i)] = torch.nn.Conv2d(scaling[i - 1] * input_shape[0],
                                                                         scaling[i] * input_shape[0],
                                                                         conv_kernel_size, stride=conv_stride)
            else:
                print('EXCEPTION on Layer {}'.format(i))

            # Each convolutional layer is followed by max-pooling layer and ReLu activation.
            self._conv_layers['MaxPool-{}'.format(i)] = torch.nn.MaxPool2d(kernel_size=max_kernel_size,
                                                                           stride=max_stride)
            self._conv_layers['ReLu-{}'.format(i)] = torch.nn.ReLU()

        # Construct the convolutional network from the dict of layers.
        self.conv_net = torch.nn.Sequential(self._conv_layers)

        # Calculate the input of the Linear layer by sending some fake data through the network
        # and getting the shape of the output.
        out_shape = []
        for i in range(len(scaling)):
            if i is 0:
                out_shape = get_output_shape(self._conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self._conv_layers['Conv-{}'.format(i)], input_shape))
            if i > 0:
                out_shape = get_output_shape(self._conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self._conv_layers['Conv-{}'.format(i)], out_shape))

        # Calculate the flattened size of the output from the convolutional network.
        self.flattened_size = int(np.prod(list(out_shape)))

        #self._fc_layers['Flatten'] = torch.nn.Linear(self.flattened_size, self.flattened_size)
        self._fc_layers['Classification'] = torch.nn.Linear(self.flattened_size, self.n_classes)

        self.fc_net = torch.nn.Sequential(self._fc_layers)

        # Set the loss function.
        self.criterion = criterion

        # Initialises the optimiser as None. MUST be set after init of CNN to a torh optimiser with this model's
        # parameters using set_optimiser before training can proceed.
        self.optimiser = None

    def set_optimiser(self, optimiser):
        """Sets the optimiser used by the model.

        Must be called after initialising a model and supplied with a PyTorch optimiser using this model's parameters.

        Args:
            optimiser: PyTorch optimiser model will use, initialised with this model's parameters.
        """
        self.optimiser = optimiser

    def forward(self, x):
        """Performs a forward pass of the convolutional network and then the fully connected network.

        Can be called directly as a method of MLP (e.g. model.forward()) or when data is parsed to MLP (e.g. model()).

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        # Inputs the data into the convolutional network.
        conv_out = self.conv_net(x)

        # Output from convolutional network is flattened and input to the fully connected network for classification.
        return self.fc_net(conv_out.view(-1, self.flattened_size))

    def step(self, x, y, train: bool):
        """Generic step of model fitting using a batch of data.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.
            train (bool): Sets whether this shall be a training step or not. True for training step which will then
                clear the optimiser, and perform a backward pass of the network then update the optimiser.
                If False for a validation or testing step, these actions are not taken.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        # Forward pass.
        z = self.forward(x)

        # Compute Loss.
        loss = self.criterion(z, y)

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, z

    def training_step(self, x, y):
        """Calls step with train=True to perform a training step. See step for more details.

        Designed to be compatible with Trainer and future compatibility with PyTorchLightning.
        Hence the resulting `boilerplate' of this method and validation_step and testing_step.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        return self.step(x, y, True)

    def validation_step(self, x, y):
        """Calls step with train=False to perform a validation step. See step for more details.

        Designed to be compatible with Trainer and future compatibility with PyTorchLightning.
        Hence the resulting `boilerplate' of this method and training_step and testing_step.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        return self.step(x, y, False)

    def testing_step(self, x, y):
        """Calls step with train=False to perform a testing step. See step for more details.

        Designed to be compatible with Trainer and future compatibility with PyTorchLightning.
        Hence the resulting `boilerplate' of this method and validation_step and training_step.

        Args:
            x (torch.FloatTensor): Batch of input data to network.
            y (torch.LongTensor): Batch of ground truth labels for the input data.

        Returns:
            loss: Loss computed by the loss function.
            z: Predicted label for the input data by the network.
        """
        return self.step(x, y, False)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_output_shape(model, image_dim):
    """Gets the output shape of a model.

    Args:
        model: Model for which the shape of the output needs to be found.
        image_dim (list[int] or tuple[int]): Expected shape of the input data to the model.

    Returns:
        The shape of the output data from the model.
    """
    return model(torch.rand([1, *image_dim])).data.shape[1:]
