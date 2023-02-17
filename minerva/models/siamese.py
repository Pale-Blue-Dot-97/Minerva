# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker

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
"""Module containing Siamese models."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "MinervaSiamese",
    "SimCLR18",
    "SimCLR34",
    "SimCLR50",
    "SimSiam18",
    "SimSiam34",
    "SimSiam50",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.modules as nn
from torch import Tensor
from torch.nn.modules import Module

from .core import MinervaBackbone, MinervaModel, get_model


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaSiamese(MinervaBackbone):
    """Abstract class for Siamese models.

    Attributes:
        backbone (MinervaModel):
        proj_head (Module):
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone: MinervaModel
        self.proj_head: Module

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of the network by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (Tensor): Pair of batches of input data to the network.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple of:
                * Ouput feature vectors concated together.
                * Output feature vector A.
                * Output feature vector B.
                * Detached embedding, A, from the backbone.
                * Detached embedding, B, from the backbone.
        """
        return self.forward_pair(x)

    def forward_pair(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of the network by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Args:
            x (Tensor): Pair of batches of input data to the network.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple of:
                * Ouput feature vectors concated together.
                * Output feature vector A.
                * Output feature vector B.
                * Embedding, A, from the backbone.
                * Embedding, B, from the backbone.
        """
        g_a, f_a = self.forward_single(x[0])
        g_b, f_b = self.forward_single(x[1])

        g = torch.cat([g_a, g_b], dim=0)  # type: ignore[attr-defined]

        assert isinstance(g, Tensor)

        return g, g_a, g_b, f_a, f_b

    @abc.abstractmethod
    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass of a single head of the network by using the forward methods of the backbone
        and feeding its output into the projection heads.

        Args:
            x (Tensor): (Unpaired) Batch of input data to the network.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of the feature vector outputted from the projection head and the detached
            embedding vector from the backbone.
        """
        raise NotImplementedError  # pragma: no cover


class _SimCLR(MinervaSiamese):
    """Base SimCLR class to be subclassed by SimCLR variants.

    Subclasses :class:`MinervaSiamse`.

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

        super(_SimCLR, self).__init__(criterion=criterion, input_size=input_size)

        self.backbone: MinervaModel = get_model(backbone_name)(
            input_size=input_size, encoder=True, **backbone_kwargs  # type: ignore[arg-type]
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

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass of a single head of the network by using the forward methods of the backbone
        and feeding its output into the projection heads.

        Args:
            x (Tensor): (Unpaired) Batch of input data to the network.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of the feature vector outputted from the projection head and the detached
            embedding vector from the backbone.
        """
        f: Tensor = torch.flatten(self.backbone(x)[0], start_dim=1)
        g: Tensor = self.proj_head(f)

        return g, f

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If ``self.optimiser`` is None.

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

        assert self.criterion

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        # Forward pass.
        z, z_a, z_b, _, _ = self.forward(x)

        # Compute Loss.
        loss: Tensor = self.criterion(z_a, z_b)  # type: ignore[arg-type]

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


class _SimSiam(MinervaSiamese):
    """Base SimSiam class to be subclassed by SimSiam variants.

    Subclasses :class:`MinervaSiamese`.

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

        super(_SimSiam, self).__init__(criterion=criterion, input_size=input_size)

        self.backbone: MinervaModel = get_model(backbone_name)(
            input_size=input_size, encoder=True, **backbone_kwargs  # type: ignore[arg-type]
        )

        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)

        prev_dim = np.prod(backbone_out_shape)

        self.proj_head = nn.Sequential(  # type: ignore[arg-type]
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

        # Build a 2-layer predictor.
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, feature_dim),
        )  # output layer

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass of a single head of :class:`_SimSiam` by using the forward methods of the backbone
        and feeding its output into the projection heads.

        Args:
            x (Tensor): (Unpaired) Batch of input data to the network.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of the feature vector outputted from the projection head and the detached
            embedding vector from the backbone.
        """
        z: Tensor = self.proj_head(torch.flatten(self.backbone(x)[0], start_dim=1))  # type: ignore[attr-defined]

        p: Tensor = self.predictor(z)

        return p, z.detach()

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If ``self.optimiser`` is None.

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

        assert self.criterion

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        # Forward pass.
        p, p_a, p_b, z_a, z_b = self.forward(x)

        # Compute Loss.
        loss: Tensor = 0.5 * (self.criterion(z_a, p_b) + self.criterion(z_b, p_a))  # type: ignore[arg-type]

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
