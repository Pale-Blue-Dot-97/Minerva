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
"""Module containing core utility functions and abstract classes for models."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
from typing import (
    Any,
    Callable,
    Iterable,
    List,
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
from torch.nn.modules import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torchvision.models._api import WeightsEnum

from minerva.utils.utils import func_by_str

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"

__all__ = [
    "MinervaModel",
    "MinervaDataParallel",
    "MinervaBackbone",
    "get_torch_weights",
    "get_output_shape",
    "bilinear_init",
]

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
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
        ...

    @overload
    def step(
        self, x: Tensor, *, train: bool = False
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
        ...

    def step(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        train: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
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
            Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]]]: Tuple of the loss computed by the loss function
            and the model outputs.
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
        loss: Tensor = self.criterion(z, y)

        # Performs a backward pass if this is a training step.
        if train:
            loss.backward()
            self.optimiser.step()

        return loss, z


class MinervaBackbone(MinervaModel):
    """Abstract class to mark a model for use as a backbone."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def __init__(
        self,
        model: Module,
        Paralleliser: Union[Type[DataParallel], Type[DistributedDataParallel]],
        *args,
        **kwargs,
    ) -> None:
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


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_model(model_name: str) -> Callable[..., MinervaModel]:
    model: Callable[..., MinervaModel] = func_by_str("minerva.models", model_name)
    return model


def get_torch_weights(weights_name: str) -> WeightsEnum:
    """Loads pre-trained model weights from ``torchvision`` via Torch Hub API.

    Args:
        weights_name (str): Name of model weights. See ... for a list of possible pre-trained weights.

    Returns:
        WeightsEnum: API query for the specified weights. See note on use:

    Raises:
        OSError: If no internet connection, ``OSError`` 101 will be raised. Reverts to using local cache.

    Note:
        This function only returns a query for the API of the weights. To actually use them, you need to call
        ``get_state_dict(progress)`` where progress is a ``bool`` on whether to show a progress bar for the
        downloading of the weights (if not already in cache).
    """
    weights: WeightsEnum
    try:
        weights = torch.hub.load("pytorch/vision", "get_weight", name=weights_name)
    except OSError:
        th_dir = os.environ.get("TORCH_HUB", os.path.expanduser("~/.cache/torch/hub"))
        weights = torch.hub.load(
            f"{th_dir}/pytorch_vision_main",
            "get_weight",
            name=weights_name,
            source="local",
        )

    return weights


def get_output_shape(
    model: Module,
    image_dim: Union[Tuple[int, ...], List[int]],
    sample_pairs: bool = False,
) -> Union[int, Sequence[int]]:
    """Gets the output shape of a model.

    Args:
        model (Module): Model for which the shape of the output needs to be found.
        image_dim (list[int] or tuple[int, ...]): Expected shape of the input data to the model.

    Returns:
        The shape of the output data from the model.
    """
    _image_dim: Union[Tuple[int, ...], List[int], int] = image_dim
    try:
        if len(image_dim) == 1:
            _image_dim = image_dim[0]
    except TypeError:
        if not hasattr(image_dim, "__len__"):
            pass
        else:
            raise TypeError

    if not hasattr(_image_dim, "__len__"):
        random_input = torch.rand([4, _image_dim])
    elif sample_pairs:
        assert isinstance(_image_dim, Iterable)
        random_input = torch.rand([2, 4, *_image_dim])
    else:
        assert isinstance(_image_dim, Iterable)
        random_input = torch.rand([4, *_image_dim])

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
    weight: NDArray[Any, Any] = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype="float32"
    )
    weight[range(in_channels), range(out_channels), :, :] = filt

    weights = torch.from_numpy(weight)  # type: ignore[attr-defined]
    assert isinstance(weights, Tensor)
    return weights
