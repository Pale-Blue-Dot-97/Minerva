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
"""Module containing Siamese models."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict, Optional, Tuple, Sequence
import numpy as np
import torch
from torch import Tensor
import torch.nn.modules as nn

from .core import MinervaModel, MinervaBackbone, get_model

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
        
        self.backbone: MinervaModel = get_model(backbone_name)(
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

        self.backbone: MinervaModel = get_model(backbone_name)(
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
