"""Module containing neural network model classes.

    Copyright (C) 2022 Harry James Baker

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

Created under a project funded by the Ordnance Survey Ltd.

TODO:
    * Consider removing redundant models
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Union, Optional, Tuple, List, Callable, Type, Any, Dict
import abc
from Minerva.utils import utils
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from abc import ABC
import numpy as np
from collections import OrderedDict


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaModel(torch.nn.Module, ABC):
    """Abstract class to act as a base for all Minerva Models. Designed to provide inter-compatability with Trainer.

    Attributes:
        criterion: PyTorch loss function model will use.
        input_shape (tuple[int, int, int] or list[int]): The shape of the input data in order of
            number of channels, image width, image height.
        n_classes (int): Number of classes in input data.
        output_shape: The shape of the output of the network. Determined and set by determine_output_dim.
        optimiser: PyTorch optimiser model will use, to be initialised with inherited model's parameters.

    Args:
        criterion: Optional; PyTorch loss function model will use.
        input_shape (tuple[int, int, int] or list[int]): Optional; Defines the shape of the input data in order of
            number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in input data.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, criterion=None, input_shape: Optional[Tuple[int, ...]] = None,
                 n_classes: Optional[int] = None) -> None:
        super(MinervaModel, self).__init__()

        # Sets loss function
        self.criterion = criterion

        self.input_shape = input_shape
        self.n_classes = n_classes

        # Output shape initialised as None. Should be set by calling determine_output_dim.
        self.output_shape = None

        # Optimiser initialised as None as the model parameters created by its init is required to init a
        # torch optimiser. The optimiser MUST be set by calling set_optimiser before the model can be trained.
        self.optimiser = None

    def set_optimiser(self, optimiser: Any) -> None:
        """Sets the optimiser used by the model.

        Must be called after initialising a model and supplied with a PyTorch optimiser using this model's parameters.

        Args:
            optimiser: PyTorch optimiser model will use, initialised with this model's parameters.
        """
        self.optimiser = optimiser

    def determine_output_dim(self) -> None:
        """Uses get_output_shape to find the dimensions of the output of this model and sets to attribute."""
        self.output_shape = get_output_shape(self, self.input_shape)

    @abc.abstractmethod
    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """Abstract method for performing a forward pass. Needs implementing!

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        return x

    def step(self, x: torch.FloatTensor, y: torch.LongTensor, train: bool) -> Tuple[Any, torch.Tensor]:
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

    def training_step(self, x: torch.FloatTensor, y: torch.LongTensor) -> Tuple[Any, torch.Tensor]:
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

    def validation_step(self, x: torch.FloatTensor, y: torch.LongTensor) -> Tuple[Any, torch.Tensor]:
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

    def testing_step(self, x: torch.FloatTensor, y: torch.LongTensor) -> Tuple[Any, torch.Tensor]:
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

    def __init__(self, criterion: Any, input_size: int = 288, n_classes: int = 8,
                 hidden_sizes: Union[Tuple[int, ...], List[int]] = (256, 144)) -> None:
        super(MLP, self).__init__(criterion=criterion)

        self.input_size = input_size
        self.output_size = n_classes
        self.hidden_sizes = hidden_sizes
        self._layers = OrderedDict()

        # Constructs layers of the network based on the input size, the hidden sizes and the number of classes.
        for i in range(len(hidden_sizes)):
            if i == 0:
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
        self.network = torch.nn.Sequential(self._layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the network.

        Can be called directly as a method of MLP (e.g. model.forward()) or when data is parsed to MLP (e.g. model()).

        Args:
            x (torch.Tensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        return self.network(x)


class CNN(MinervaModel, ABC):
    """Simple class to construct a Convolutional Neural Network (CNN).

    Inherits from torch.nn.Module and MinervaModel. Designed for use with PyTorch functionality.

    Should be used in tandem with Trainer.

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

    def __init__(self, criterion, input_size: Union[Tuple[int, int, int], List[int]] = (12, 256, 256),
                 n_classes: int = 8, features: Union[Tuple[int, ...], List[int]] = (2, 1, 1),
                 fc_sizes: Union[Tuple[int, ...], List[int]] = (128, 64),
                 conv_kernel_size: Union[int, Tuple[int, ...]] = 3, conv_stride: Union[int, Tuple[int, ...]] = 1,
                 max_kernel_size: Union[int, Tuple[int, ...]] = 2, max_stride: Union[int, Tuple[int, ...]] = 2,
                 conv_do: bool = True, fc_do: bool = True, p_conv_do: float = 0.1, p_fc_do: float = 0.5) -> None:
        super(CNN, self).__init__(criterion=criterion, input_shape=input_size, n_classes=n_classes)

        self._conv_layers = OrderedDict()
        self._fc_layers = OrderedDict()

        # Checks that the kernel sizes and strides match the number of layers defined by features.
        conv_kernel_size = utils.check_len(conv_kernel_size, features)
        conv_stride = utils.check_len(conv_stride, features)

        # Constructs the convolutional layers determined by the number of input channels and the features of these.
        for i in range(len(features)):
            if i == 0:
                self._conv_layers['Conv-0'] = torch.nn.Conv2d(self.input_shape[0], features[i],
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
            if i == 0:
                out_shape = get_output_shape(self._conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self._conv_layers['Conv-{}'.format(i)], self.input_shape))
            if i > 0:
                out_shape = get_output_shape(self._conv_layers['MaxPool-{}'.format(i)],
                                             get_output_shape(self._conv_layers['Conv-{}'.format(i)], out_shape))

        # Calculate the flattened size of the output from the convolutional network.
        self.flattened_size = int(np.prod(list(out_shape)))

        # Constructs the fully connected layers determined by the number of input channels and the features of these.
        for i in range(len(fc_sizes)):
            if i == 0:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the convolutional network and then the fully connected network.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.Tensor): Input data to network.

        Returns:
            torch.Tensor of the likelihoods the network places on the input 'x' being of each class.
        """
        # Inputs the data into the convolutional network.
        conv_out = self.conv_net(x)

        # Output from convolutional network is flattened and input to the fully connected network for classification.
        return self.fc_net(conv_out.view(-1, self.flattened_size))


class ResNet(MinervaModel, ABC):
    """Modified version of the ResNet network to handle multi-spectral inputs and cross-entropy.

    Attributes:
        encoder_on (bool): Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.
        inplanes (int): Number of input feature maps. Initially set to 64.
        dilation (int): Dilation factor of convolutions. Initially set to 1.
        groups (int): Number of convolutions in grouped convolutions of Bottleneck Blocks.
        base_width (int): Modifies the number of feature maps in convolutional layers of Bottleneck Blocks.
        conv1 (torch.nn.Conv2d): Input convolutional layer of Conv1 input block to the network.
        bn1 (torch.nn.Module): Batch normalisation layer of the Conv1 input block to the network.
        relu (torch.nn.ReLU): Rectified Linear Unit (ReLU) activation layer to be used throughout ResNet.
        maxpool (torch.nn.MaxPool2d): 3x3 Max-pooling layer with stride 2 of the Conv1 input block to the network.
        layer1 (torch.nn.Sequential): `Layer' 1 of the ResNet comprising number and type of blocks defined by 'layers'.
        layer2 (torch.nn.Sequential): `Layer' 2 of the ResNet comprising number and type of blocks defined by 'layers'.
        layer3 (torch.nn.Sequential): `Layer' 3 of the ResNet comprising number and type of blocks defined by 'layers'.
        layer4 (torch.nn.Sequential): `Layer' 4 of the ResNet comprising number and type of blocks defined by 'layers'.
        avgpool (torch.nn.AdaptiveAvgPool2d): Global average pooling layer taking the output from the last block.
            Only initialised if encoder_on is False.
        fc (torch.nn.Linear): Fully connected layer that takes the flattened output from average pooling
            to a classification output. Only initialised if encoder_on is False.

    Args:
        block (BasicBlock or Bottleneck): Type of `block operations' to use throughout network.
        layers (list): Number of blocks in each of the 4 `layers'.
        in_channels (int): Optional; Number of channels (or bands) in the input imagery.
        n_classes (int): Optional; Number of classes in data to be classified.
        zero_init_residual (bool): Optional; If True, zero-initialise the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Optional; Number of convolutions in grouped convolutions of Bottleneck Blocks.
            Not compatible with Basic Block!
        width_per_group (int): Optional; Modifies the number of feature maps in convolutional layers
            of Bottleneck Blocks. Not compatible with Basic Block!
        replace_stride_with_dilation (tuple): Optional; Each element in the tuple indicates whether to replace the
            2x2 stride with a dilated convolution instead. Must be a three element tuple of bools.
        norm_layer (function): Optional; Normalisation layer to use in each block. Typically, torch.nn.BatchNorm2d.
        encoder (bool): Optional; Whether to initialise the ResNet as an encoder or end-to-end classifier.
            If True, forward method returns the output of each layer block. avgpool and fc are not initialised.
            If False, adds a global average pooling layer after the last block, flattens the output
            and passes through a fully connected layer for classification output.

    Raises:
        ValueError: If replace_stride_with_dilation is not None or a 3-element tuple.
    """

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: Union[List[int], Tuple[int, int, int, int]],
                 in_channels: int = 3, n_classes: int = 8, zero_init_residual: bool = False, groups: int = 1,
                 width_per_group: int = 64, replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None, encoder: bool = False) -> None:
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
            replace_stride_with_dilation = [False, False, False]

        # Raises ValueError if replace_stride_with_dilation is not a 3-element tuple of bools.
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        # Sets the number of convolutions in groups and the base width of convolutions.
        self.groups = groups
        self.base_width = width_per_group

        # --- CONV1 LAYER =============================================================================================
        # Adds the input convolutional layer to the network.
        self.conv1 = torch.nn.Conv2d(in_channels, self.inplanes, kernel_size=(7, 7), stride=(2, 2),
                                     padding=3, bias=False)
        # Adds the batch norm layer for the Conv1 layer.
        self.bn1 = norm_layer(self.inplanes)

        # Inits the ReLU to be use in Conv1 and throughout the network.
        self.relu = torch.nn.ReLU(inplace=True)

        # Adds the max pooling layer to complete the Conv1 layer.
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- LAYERS 1-4 ==============================================================================================
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # =============================================================================================================

        # Adds average pooling and classification layer to network if this is an end-to-end classifier.
        if not self.encoder_on:
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512 * block.expansion, n_classes)

        # Performs weight initialisation across network.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
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
        try:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))

        except ValueError as err:
            print(err.args)
            print('Setting groups=1, base_width=64 and trying again')
            self.groups = 1
            self.base_width = 64
            try:
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))

            except ValueError as err:
                print(err.args)

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor
                      ) -> Union[torch.Tensor,
                                 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0: torch.Tensor = self.maxpool(x)

        x1: torch.Tensor = self.layer1(x0)
        x2: torch.Tensor = self.layer2(x1)
        x3: torch.Tensor = self.layer3(x2)
        x4: torch.Tensor = self.layer4(x3)

        if self.encoder_on:
            return x4, x3, x2, x1, x0

        if not self.encoder_on:
            x5 = self.avgpool(x4)
            x5 = torch.flatten(x5, 1)
            x5 = self.fc(x5)

            return x5

    def forward(self, x: torch.Tensor
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Performs a forward pass of the ResNet.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.Tensor): Input data to network.

        Returns:
            If inited as an backbone, returns a tuple of outputs from each `layer' 1-4. Else, returns torch.Tensor
                of the likelihoods the network places on the input 'x' being of each class.
        """
        return self._forward_impl(x)


class Decoder(MinervaModel, ABC):
    """Decoder network taken from an autoencoder example:

    https://github.com/arnaghosh/Auto-Encoder/blob/master/resnet.py

    Attributes:
        batch_size (int): Number of samples in each batch supplied to the network.
        n_classes (int): Number of classes in the data to be classified by the network.
        image_size (tuple[int, int, int] or list[int, int, int]): Defines the shape of the input data in order of
            number of channels, image width, image height.
        relu (torch.nn.ReLU): Rectified Linear Unit (ReLU) activation layer to be used throughout the network.
        fc3 (torch.nn.Linear): First fully connected layer of the network that should take the input from the encoder.
        bn3 (torch.nn.BatchNorm1d): First batch norm layer.
        fc2 (torch.nn.Linear): Second fully connected layer of the network.
        bn2 (torch.nn.BatchNorm1d): Second batch norm layer.
        fc1 (torch.nn.Linear): Third fully connected layer of the network.
        bn1 (torch.nn.BatchNorm1d): Third batch norm layer.
        upsample1 (torch.nn.Upsample): 2x factor up-sampling layer used throughout network.
        dconv5 (torch.nn.ConvTranspose2d): First de-convolutional layer with 3x3 kernel and no padding.
        dconv4 (torch.nn.ConvTranspose2d): Second de-convolutional layer with 3x3 kernel and padding=1.
        dconv3 (torch.nn.ConvTranspose2d): Third de-convolutional layer with 3x3 kernel and padding=1.
        dconv2 (torch.nn.ConvTranspose2d): Fourth de-convolutional layer with 5x5 kernel and padding=2.
        dconv1 (torch.nn.ConvTranspose2d): Fifth de-convolutional layer with 12x12 kernel, stride=4 and padding=4.
        upsample2 (torch.nn.Upsample): Final up-sampling layer to ensure the output of the network matches
            the original image size.

    Args:
        batch_size (int): Number of samples in each batch supplied to the network.
        n_classes (int): Number of classes in the data to be classified by the network.
        image_size (tuple[int, int, int] or list[int, int, int]): Defines the shape of the input data in order of
            number of channels, image width, image height.
    """

    def __init__(self, batch_size: int, n_classes: int, image_size: Union[Tuple[int, int, int], List[int]]) -> None:
        super(Decoder, self).__init__(n_classes=n_classes)

        self.batch_size = batch_size
        self.image_size = image_size

        # Init ReLU for use throughout network.
        self.relu = torch.nn.ReLU(inplace=True)

        # First fully connected layer that should take input from an encoder. Followed by batch norm.
        self.fc3 = torch.nn.Linear(1024, 4096)
        self.bn3 = torch.nn.BatchNorm1d(4096)

        # Second fully connected layer and batch norm layer.
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.bn2 = torch.nn.BatchNorm1d(4096)

        # Third and final fully connected layer and batch norm layer.
        self.fc1 = torch.nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = torch.nn.BatchNorm1d(256 * 6 * 6)

        # 2x factor up-sampling operation to use throughout network.
        self.upsample1 = torch.nn.Upsample(scale_factor=2)

        # De-convolutional layers.
        self.dconv5 = torch.nn.ConvTranspose2d(256, 256, (3, 3), padding=(0, 0))
        self.dconv4 = torch.nn.ConvTranspose2d(256, 384, (3, 3), padding=(1, 1))
        self.dconv3 = torch.nn.ConvTranspose2d(384, 192, (3, 3), padding=(1, 1))
        self.dconv2 = torch.nn.ConvTranspose2d(192, 64, (5, 5), padding=(2, 2))
        self.dconv1 = torch.nn.ConvTranspose2d(64, self.n_classes, (12, 12), stride=(4, 4), padding=(4, 4))

        # Up-sampling operation to take output from de-convolutions and match to input size of image.
        self.upsample2 = torch.nn.Upsample(size=self.image_size)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # First block of fully connected layer batch norm and ReLU.
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(self.bn3(x))

        # Second block of fully connected layer batch norm and ReLU.
        x = self.fc2(x)
        x = self.relu(self.bn2(x))

        # Third and final block of fully connected layer batch norm and ReLU.
        x = self.fc1(x)
        x = self.relu(self.bn1(x))

        # Expands 2D tensor of data (batch, vector) into 4D (batch, feature maps, height, width) tensor.
        x = x.view(self.batch_size, 256, 6, 6)

        # Up-samples by 2x.
        x = self.upsample1(x)

        # De-convolutional layers, up-sampling and ReLUs.
        x = self.relu(self.dconv5(x))
        x = self.relu(self.dconv4(x))
        x = self.relu(self.dconv3(x))
        x = self.upsample1(x)
        x = self.relu(self.dconv2(x))
        x = self.upsample1(x)
        x = self.relu(self.dconv1(x))

        # Final up-sampling layer to ensure output matches the dimensions of the input to the encoder.
        x = self.upsample2(x)

        return x

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Performs a forward pass of the decoder.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.Tensor): Input data to network. Should be from a backbone.

        Returns:
            torch.Tensor segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.
        """
        return self._forward_impl(x[0])


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
    def __init__(self, in_channel: int = 512, n_classes: int = 21, variant: str = '32') -> None:
        super(DCN, self).__init__(n_classes=n_classes)
        self.variant = variant

        # Common to all variants.
        self.relu = torch.nn.ReLU(inplace=True)
        self.Conv1x1 = torch.nn.Conv2d(in_channel, self.n_classes, kernel_size=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(self.n_classes)

        if variant == '32':
            self.DC32 = torch.nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(64, 64),
                                                 stride=(32, 32), dilation=1, padding=(16, 16))
            self.DC32.weight.data = bilinear_init(self.n_classes, self.n_classes, 64)
            self.dbn32 = torch.nn.BatchNorm2d(self.n_classes)

        if variant in ('16', '8'):
            self.Conv1x1_x3 = torch.nn.Conv2d(int(in_channel / 2), self.n_classes, kernel_size=(1, 1))
            self.DC2 = torch.nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(4, 4), stride=(2, 2),
                                                dilation=1, padding=(1, 1))
            self.DC2.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn2 = torch.nn.BatchNorm2d(self.n_classes)

        if variant == '16':
            self.DC16 = torch.nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(32, 32), stride=(16, 16),
                                                 dilation=1, padding=(8, 8))
            self.DC16.weight.data = bilinear_init(self.n_classes, self.n_classes, 32)
            self.dbn16 = torch.nn.BatchNorm2d(self.n_classes)

        if variant == '8':
            self.Conv1x1_x2 = torch.nn.Conv2d(int(in_channel / 4), self.n_classes, kernel_size=(1, 1))

            self.DC4 = torch.nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(4, 4), stride=(2, 2),
                                                dilation=1, padding=(1, 1))
            self.DC4.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn4 = torch.nn.BatchNorm2d(self.n_classes)

            self.DC8 = torch.nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=(16, 16), stride=(8, 8),
                                                dilation=1, padding=(4, 4))
            self.DC8.weight.data = bilinear_init(self.n_classes, self.n_classes, 16)
            self.dbn8 = torch.nn.BatchNorm2d(self.n_classes)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Performs a forward pass of the decoder. Depending on DCN variant, will take multiple inputs
        throughout pass from the encoder.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input data to network.
                Should be from a backbone that supports output at multiple points e.g ResNet.

        Returns:
            torch.Tensor segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.
        """
        # Unpack outputs from the ResNet layers.
        x4, x3, x2, *_ = x

        # All DCNs have a common 1x1 Conv input block.
        z = self.bn1(self.relu(self.Conv1x1(x4)))

        # If DCN32, forward pass through DC32 and DBN32 and return output.
        if self.variant == '32':
            z = self.dbn32(self.relu(self.DC32(z)))
            return z

        # Common Conv1x1 layer to DCN16 & DCN8.
        x3 = self.bn1(self.relu(self.Conv1x1_x3(x3)))
        z = self.dbn2(self.relu(self.DC2(z)))

        z = z + x3

        # If DCN16, forward pass through DCN16 and DBN16 and return output.
        if self.variant == '16':
            z = self.dbn16(self.relu(self.DC16(z)))
            return z

        # If DCN8, continue through remaining layers to output.
        elif self.variant == '8':
            x2 = self.bn1(self.relu(self.Conv1x1_x2(x2)))
            z = self.dbn4(self.relu(self.DC4(z)))

            z = z + x2

            z = self.dbn8(self.relu(self.DC8(z)))

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

    def __init__(self, criterion: Optional[Any] = None, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, zero_init_residual: bool = False,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None, encoder: bool = False) -> None:
        super(ResNet18, self).__init__(criterion=criterion, input_shape=input_size, n_classes=n_classes)

        self.network = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=input_size[0], n_classes=n_classes,
                              zero_init_residual=zero_init_residual, groups=1, width_per_group=64,
                              replace_stride_with_dilation=replace_stride_with_dilation,
                              norm_layer=norm_layer, encoder=encoder)

    def forward(self, x: torch.FloatTensor) -> Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Performs a forward pass of the ResNet.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            If inited as an backbone, returns a tuple of outputs from each `layer' 1-4. Else, returns torch.Tensor
                of the likelihoods the network places on the input 'x' being of each class.
        """
        return self.network(x)


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

    def __init__(self, criterion: Optional[Any] = None, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, zero_init_residual: bool = False,
                 replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None, encoder: bool = False) -> None:
        super(ResNet34, self).__init__(criterion=criterion, input_shape=input_size, n_classes=n_classes)

        self.network = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=input_size[0], n_classes=n_classes,
                              zero_init_residual=zero_init_residual, groups=1, width_per_group=64,
                              replace_stride_with_dilation=replace_stride_with_dilation,
                              norm_layer=norm_layer, encoder=encoder)

    def forward(self, x: torch.FloatTensor) -> Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Performs a forward pass of the ResNet.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.Tensor): Input data to network.

        Returns:
            If inited as an backbone, returns a tuple of outputs from each `layer' 1-4. Else, returns torch.Tensor
                of the likelihoods the network places on the input 'x' being of each class.
        """
        return self.network(x)


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

    def __init__(self, criterion: Optional[Any] = None, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None, encoder: bool = False) -> None:
        super(ResNet50, self).__init__(criterion=criterion, input_shape=input_size, n_classes=n_classes)

        self.network = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=input_size[0], n_classes=n_classes,
                              zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
                              replace_stride_with_dilation=replace_stride_with_dilation,
                              norm_layer=norm_layer, encoder=encoder)

    def forward(self, x: torch.FloatTensor) -> Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Performs a forward pass of the ResNet.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.Tensor): Input data to network.

        Returns:
            If inited as an backbone_name, returns a tuple of outputs from each `layer' 1-4. Else, returns torch.Tensor
                of the likelihoods the network places on the input 'x' being of each class.
        """
        return self.network(x)


class _FCN(MinervaModel, ABC):
    """Base Fully Convolutional Network (FCN) class to be subclassed by FCN variants described in the FCN paper.

    Subclasses MinervaModel.

    Attributes:
        backbone (torch.nn.Module): Backbone of the FCN that takes the imagery input and
            extracts learned representations.
        decoder (torch.nn.Module): Decoder that takes the learned representations from the backbone encoder
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
        backbone_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int], List[int]] = (12, 256, 256), n_classes: int = 8,
                 backbone_name: str = 'ResNet18', decoder_name: str = 'DCN', decoder_variant: str = '32',
                 batch_size: int = 16, backbone_kwargs: Optional[Dict[str, Any]] = None) -> None:

        super(_FCN, self).__init__(criterion=criterion, input_shape=input_size, n_classes=n_classes)

        self.backbone = globals()[backbone_name](input_size=input_size, n_classes=n_classes, encoder=True,
                                                 **backbone_kwargs)
        self.backbone.determine_output_dim()

        if decoder_name == 'DCN':
            self.decoder = DCN(in_channel=self.backbone.output_shape[0], n_classes=n_classes, variant=decoder_variant)
        if decoder_name == 'Decoder':
            self.decoder = Decoder(batch_size=batch_size, image_size=input_size[1:], n_classes=n_classes)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """Performs a forward pass of the FCN by using the forward methods of the backbone and
        feeding its output into the forward for the decoder.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (torch.FloatTensor): Input data to network.

        Returns:
            z (torch.Tensor): segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.
        """
        z = self.backbone(x)
        z = self.decoder(z)

        return z


class FCNResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone but a decoder NOT defined in the original FCN paper.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        batch_size (int): Optional; Number of samples in each batch supplied to the network.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, batch_size: int = 16, **resnet_kwargs) -> None:
        super(FCNResNet18, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                          batch_size=batch_size, backbone_name='ResNet18', decoder_name='Decoder',
                                          backbone_kwargs=resnet_kwargs)


class FCNResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone but a decoder NOT defined in the original FCN paper.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        batch_size (int): Optional; Number of samples in each batch supplied to the network.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, batch_size: int = 16, **resnet_kwargs) -> None:
        super(FCNResNet34, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                          batch_size=batch_size, backbone_name='ResNet34', decoder_name='Decoder',
                                          backbone_kwargs=resnet_kwargs)


class FCNResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet50 backbone but a decoder NOT defined in the original FCN paper.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        batch_size (int): Optional; Number of samples in each batch supplied to the network.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, batch_size: int = 16, **resnet_kwargs) -> None:
        super(FCNResNet50, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                          batch_size=batch_size, backbone_name='ResNet50', decoder_name='Decoder',
                                          backbone_kwargs=resnet_kwargs)


class FCN32ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN32ResNet18, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                            backbone_name='ResNet18', decoder_variant='32',
                                            backbone_kwargs=resnet_kwargs)


class FCN32ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN32ResNet34, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                            backbone_name='ResNet34', decoder_variant='32',
                                            backbone_kwargs=resnet_kwargs)


class FCN32ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN32ResNet50, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                            backbone_name='ResNet50', decoder_variant='32',
                                            backbone_kwargs=resnet_kwargs)


class FCN16ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN16ResNet18, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                            backbone_name='ResNet18', decoder_variant='16',
                                            backbone_kwargs=resnet_kwargs)


class FCN16ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN16ResNet34, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                            backbone_name='ResNet34', decoder_variant='16',
                                            backbone_kwargs=resnet_kwargs)


class FCN8ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN8ResNet18, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                           backbone_name='ResNet18', decoder_variant='8',
                                           backbone_kwargs=resnet_kwargs)


class FCN8ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN8ResNet34, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                           backbone_name='ResNet34', decoder_variant='8',
                                           backbone_kwargs=resnet_kwargs)


class FCN8ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet50 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(self, criterion: Any, input_size: Union[Tuple[int, ...], List[int]] = (12, 256, 256),
                 n_classes: int = 8, **resnet_kwargs) -> None:
        super(FCN8ResNet50, self).__init__(criterion=criterion, input_size=input_size, n_classes=n_classes,
                                           backbone_name='ResNet50', decoder_variant='8',
                                           backbone_kwargs=resnet_kwargs)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_output_shape(model: torch.nn.Module, image_dim: Union[List[int], Tuple[int, ...]]) -> Any:
    """Gets the output shape of a model.

    Args:
        model (torch.nn.Module): Model for which the shape of the output needs to be found.
        image_dim (list[int] or tuple[int]): Expected shape of the input data to the model.

    Returns:
        The shape of the output data from the model.
    """
    output = model(torch.rand([1, *image_dim]))
    return output[0].data.shape[1:]


def bilinear_init(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
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
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight)
