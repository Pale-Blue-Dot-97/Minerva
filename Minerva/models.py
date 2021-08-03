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
import abc

from Minerva.utils import utils
import torch
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from abc import ABC
import numpy as np
from collections import OrderedDict


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaModel(torch.nn.Module, ABC):
    """Abstract class to act as a base for all Minerva Models. Designed to provide inter-compatability with Trainer

    Attributes:
        criterion: PyTorch loss function model will use.
        optimiser: PyTorch optimiser model will use, to be initialised with inherited model's parameters.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, criterion=None):
        super(MinervaModel, self).__init__()

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

    @abc.abstractmethod
    def forward(self, x: torch.FloatTensor):
        """Abstract method for performing a forward pass. Needs implementing!

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        return x

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


class MLP(MinervaModel, ABC):
    """Simple class to construct a Multi-Layer Perceptron (MLP).

    Inherits from torch.nn.Module and MinervaModel. Designed for use with PyTorch functionality.

    Should be used in tandem with Trainer.

    Attributes:
        input_size (int): Size of the input vector to the network.
        output_size (int): Size of the output vector of the network.
        hidden_sizes (tuple[int] or list[int]): Size of the hidden layers within the network.
            Can be a tuple[int] or list[int] of values that will also determine the number of layers other than
            the required input and output layers.
        network (torch.nn.Sequential): The actual neural network of the model.
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
        super(MLP, self).__init__(criterion=criterion)

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

    def forward(self, x: torch.FloatTensor):
        """Performs a forward pass of the network.

        Can be called directly as a method of MLP (e.g. model.forward()) or when data is parsed to MLP (e.g. model()).

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        return self.network(x)


class CNN(MinervaModel, ABC):
    """Simple class to construct a Convolutional Neural Network (CNN).

    Inherits from torch.nn.Module and MinervaModel. Designed for use with PyTorch functionality.

    Should be used in tandem with Trainer.

    Attributes:
        input_shape (tuple[int, int, int] or list[int, int, int]): Defines the shape of the input data in order of
            number of channels, image width, image height.
        n_classes (int): Number of classes in input data.
        flattened_size (int): Length of the vector resulting from the flattening of the output from the convolutional
            network.
        conv_net (torch.nn.Sequential): Convolutional network of the model.
        fc_net (torch.nn.Sequential): Fully connected network of the model.
    """

    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes: int = 8, features=(2, 1, 1),
                 fc_sizes=(128, 64), conv_kernel_size: tuple = 3, conv_stride: tuple = 1, max_kernel_size: int = 2,
                 max_stride: int = 2, conv_do: bool = True, fc_do: bool = True, p_conv_do: float = 0.1,
                 p_fc_do: float = 0.5):
        """Initialises an instance of CNN.

        Args:
            criterion: PyTorch loss function model will use.
            input_shape (tuple[int, int, int] or list[int, int, int]): Optional; Defines the shape of the input data in
                order of number of channels, image width, image height.
            n_classes (int): Optional; Number of classes in input data.
            features (tuple[int] or list[int]): Optional; Series of values defining the number of feature maps.
                The length of the list is also used to determine the number of convolutional layers in conv_net.
            conv_kernel_size (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all convolutional kernels for all channels and layers.
            conv_stride (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all convolutional stride lengths for all channels and layers.
            max_kernel_size (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all max-pooling kernels for all channels and layers.
            max_stride (int or tuple[int]): Optional; Either a int or tuple but a single value to determine the
                size of all max-pooling stride lengths for all channels and layers.
        """
        super(CNN, self).__init__(criterion=criterion)

        self.input_shape = input_shape
        self.n_classes = n_classes
        self._conv_layers = OrderedDict()
        self._fc_layers = OrderedDict()

        # Checks that the kernel sizes and strides match the number of layers defined by features.
        conv_kernel_size = utils.check_len(conv_kernel_size, features)
        conv_stride = utils.check_len(conv_stride, features)

        # Constructs the convolutional layers determined by the number of input channels and the features of these.
        for i in range(len(features)):
            if i is 0:
                self._conv_layers['Conv-0'] = torch.nn.Conv2d(input_shape[0], features[i],
                                                              conv_kernel_size[0], stride=conv_stride[0])
            elif i > 0:
                self._conv_layers['Conv-{}'.format(i)] = torch.nn.Conv2d(features[i - 1], features[i],
                                                                         conv_kernel_size[i], stride=conv_stride[i])
            else:
                print('EXCEPTION on Layer {}'.format(i))

            # Each convolutional layer is followed by max-pooling layer and ReLu activation.
            self._conv_layers['MaxPool-{}'.format(i)] = torch.nn.MaxPool2d(kernel_size=max_kernel_size,
                                                                           stride=max_stride)
            self._conv_layers['ReLu-{}'.format(i)] = torch.nn.ReLU()

            if conv_do:
                self._conv_layers['DropOut-{}'.format(i)] = torch.nn.Dropout(p_conv_do)

        # Construct the convolutional network from the dict of layers.
        self.conv_net = torch.nn.Sequential(self._conv_layers)

        # Calculate the input of the Linear layer by sending some fake data through the network
        # and getting the shape of the output.
        out_shape = []
        for i in range(len(features)):
            if i is 0:
                out_shape = get_output_shape(self._conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self._conv_layers['Conv-{}'.format(i)], input_shape))
            if i > 0:
                out_shape = get_output_shape(self._conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self._conv_layers['Conv-{}'.format(i)], out_shape))

        # Calculate the flattened size of the output from the convolutional network.
        self.flattened_size = int(np.prod(list(out_shape)))

        # Constructs the fully connected layers determined by the number of input channels and the features of these.
        for i in range(len(fc_sizes)):
            if i is 0:
                self._fc_layers['Linear-0'] = torch.nn.Linear(self.flattened_size, fc_sizes[i])
            elif i > 0:
                self._fc_layers['Linear-{}'.format(i)] = torch.nn.Linear(fc_sizes[i - 1], fc_sizes[i])
            else:
                print('EXCEPTION on Layer {}'.format(i))

            # Each fully connected layer is followed by a ReLu activation.
            self._fc_layers['ReLu-{}'.format(i)] = torch.nn.ReLU()

            if fc_do:
                self._fc_layers['DropOut-{}'.format(i)] = torch.nn.Dropout(p_fc_do)

        # Add classification layer.
        self._fc_layers['Classification'] = torch.nn.Linear(fc_sizes[-1], self.n_classes)

        # Create fully connected network.
        self.fc_net = torch.nn.Sequential(self._fc_layers)

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


class ResNet(MinervaModel, ABC):
    """Modified version of the ResNet network to handle multi-spectral inputs and cross-entropy."""

    def __init__(self, block, layers, in_channels: int = 3, n_classes: int = 8, zero_init_residual: bool = False,
                 groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation=None,
                 norm_layer=None, encoder: bool = False) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.encoder_on = encoder

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = torch.nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = None

        if self.encoder_on:
            self.fc = torch.nn.Linear(512 * block.expansion, 1024)
        else:
            self.fc = torch.nn.Linear(512 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> torch.nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class Decoder(MinervaModel, ABC):
    def __init__(self, batch_size, n_classes, image_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.n_classes = n_classes
        self.image_size = image_size

        self.derl1 = torch.nn.ReLU()

        self.dfc3 = torch.nn.Linear(1024, 4096)
        self.bn3 = torch.nn.BatchNorm1d(4096)
        self.derl2 = torch.nn.ReLU()

        self.dfc2 = torch.nn.Linear(4096, 4096)
        self.bn2 = torch.nn.BatchNorm1d(4096)
        self.derl3 = torch.nn.ReLU()

        self.dfc1 = torch.nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = torch.nn.BatchNorm1d(256 * 6 * 6)
        self.derl4 = torch.nn.ReLU()

        self.upsample1 = torch.nn.Upsample(scale_factor=2)
        self.dconv5 = torch.nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.derl5 = torch.nn.ReLU()

        self.dconv4 = torch.nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.derl6 = torch.nn.ReLU()

        self.dconv3 = torch.nn.ConvTranspose2d(384, 192, 3, padding=1)
        self.derl7 = torch.nn.ReLU()

        self.dconv2 = torch.nn.ConvTranspose2d(192, 64, 5, padding=2)
        self.derl8 = torch.nn.ReLU()

        self.dconv1 = torch.nn.ConvTranspose2d(64, self.n_classes, 12, stride=4, padding=4)
        self.derl9 = torch.nn.ReLU()

        self.upsample2 = torch.nn.Upsample(size=self.image_size)

    def _forward_impl(self, x):
        x = self.derl1(x)
        x = self.dfc3(x)
        x = self.derl2(self.bn3(x))

        x = self.dfc2(x)
        x = self.derl3(self.bn2(x))

        x = self.dfc1(x)
        x = self.derl4(self.bn1(x))

        x = x.view(self.batch_size, 256, 6, 6)

        x = self.upsample1(x)
        x = self.derl5(self.dconv5(x))
        x = self.derl6(self.dconv4(x))
        x = self.derl7(self.dconv3(x))
        x = self.upsample1(x)
        x = self.derl8(self.dconv2(x))
        x = self.upsample1(x)
        x = self.derl9(self.dconv1(x))
        x = self.upsample2(x)
        #x = F.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class ResNet18(ResNet, ABC):

    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes: int = 8, zero_init_residual: bool = False,
                 groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], in_channels=input_shape[0],
                                       n_classes=n_classes,
                                       zero_init_residual=zero_init_residual,
                                       groups=groups,
                                       width_per_group=width_per_group,
                                       replace_stride_with_dilation=replace_stride_with_dilation,
                                       norm_layer=norm_layer,
                                       criterion=criterion)

        self.input_shape = input_shape
        self.n_classes = n_classes


class FCNResNet18(MinervaModel, ABC):
    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes: int = 8, batch_size: int = 16,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super(FCNResNet18, self).__init__(criterion=criterion)

        self.encoder = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=input_shape[0], n_classes=n_classes,
                              zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
                              replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer,
                              encoder=True)

        self.decoder = Decoder(batch_size, n_classes, input_shape[1:])

        self.input_shape = input_shape
        self.n_classes = n_classes

    def forward(self, x: torch.FloatTensor):
        z = self.encoder(x)
        z = self.decoder(z)

        return z


class FCNResNet34(MinervaModel, ABC):
    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes: int = 8, batch_size: int = 16,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super(FCNResNet34, self).__init__(criterion=criterion)

        self.encoder = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=input_shape[0], n_classes=n_classes,
                              zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
                              replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer,
                              encoder=True)

        self.decoder = Decoder(batch_size, n_classes, input_shape[1:])

        self.input_shape = input_shape
        self.n_classes = n_classes

    def forward(self, x: torch.FloatTensor):
        z = self.encoder(x)
        z = self.decoder(z)

        return z


class FCNResNet50(MinervaModel, ABC):
    def __init__(self, criterion, input_shape=(12, 256, 256), n_classes: int = 8, batch_size: int = 16,
                 zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super(FCNResNet50, self).__init__(criterion=criterion)

        self.encoder = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=input_shape[0], n_classes=n_classes,
                              zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
                              replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer,
                              encoder=True)

        self.decoder = Decoder(batch_size, n_classes, input_shape[1:])

        self.input_shape = input_shape
        self.n_classes = n_classes

    def forward(self, x: torch.FloatTensor):
        z = self.encoder(x)
        z = self.decoder(z)

        return z


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
