# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
# MIT License

# Copyright (c) 2023 Harry Baker

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
#
"""Module containing Siamese models."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "MinervaSiamese",
    "SimCLR",
    "SimCLR18",
    "SimCLR34",
    "SimCLR50",
    "SimSiam",
    "SimSiam18",
    "SimSiam34",
    "SimSiam50",
    "SimConv",
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
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module

from .core import MinervaBackbone, MinervaModel, MinervaWrapper, get_model
from .psp import PSPEncoder


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaSiamese(MinervaBackbone):
    """Abstract class for Siamese models.

    Attributes:
        backbone (MinervaModel): The backbone encoder for the Siamese model.
        proj_head (~torch.nn.Module): The projection head for re-projecting the outputs
            from the :attr:`~MinervaSiamese.backbone`.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone: MinervaModel
        self.proj_head: Module

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of the network by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Can be called directly as a method (e.g. ``model.forward()``) or when
        data is parsed to model (e.g. ``model()``).

        Args:
            x (~torch.Tensor): Pair of batches of input data to the network.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor]: Tuple of:
                * Ouput feature vectors concated together.
                * Output feature vector ``A``.
                * Output feature vector ``B``.
                * Detached embedding, ``A``, from the :attr:`~MinervaSiamese.backbone`.
                * Detached embedding, ``B``, from the :attr:`~MinervaSiamese.backbone`.
        """
        return self.forward_pair(x)

    def forward_pair(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Performs a forward pass of the network by using the forward methods of the backbone and
        feeding its output into the projection heads.

        Args:
            x (~torch.Tensor): Pair of batches of input data to the network.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor]: Tuple of:
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
            x (~torch.Tensor): Batch of unpaired input data to the network.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Tuple of the feature vector outputted from the projection head
            and the detached embedding vector from the backbone.
        """
        raise NotImplementedError  # pragma: no cover


class SimCLR(MinervaSiamese):
    """Base SimCLR class to be subclassed by SimCLR variants.

    Subclasses :class:`MinervaSiamese`.

    Attributes:
        backbone_name (str): Name of the :attr:`~SimCLR.backbone` within this module to use.
        backbone (~torch.nn.Module): Backbone of SimCLR that takes the imagery input and
            extracts learned representations.
        proj_head (~torch.nn.Module): Projection head that takes the learned representations from
            the :attr:`~SimCLR.backbone` encoder.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int, int, int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        backbone_kwargs (dict[str, ~typing.Any]): Optional; Keyword arguments for the :attr:`~SimCLR.backbone`
            packed up into a dict.
    """

    __metaclass__ = abc.ABCMeta
    backbone_name = "ResNet18"

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 128,
        scaler: Optional[GradScaler] = None,
        backbone_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(SimCLR, self).__init__(
            criterion=criterion, input_size=input_size, scaler=scaler
        )

        self.backbone: MinervaModel = get_model(self.backbone_name)(
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
        """Performs a forward pass of a single head of the network by using the forward methods of the
        :attr:`~SimCLR.backbone` and feeding its output into the :attr:`~SimCLR.proj_head`.

        Overwrites :meth:`MinervaSiamese.forward_single`

        Args:
            x (~torch.Tensor): Batch of unpaired input data to the network.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Tuple of the feature vector outputted from the
            :attr:`~SimCLR.proj_head` and the detached embedding vector from the :attr:`~SimCLR.backbone`.
        """
        f: Tensor = torch.flatten(self.backbone(x)[0], start_dim=1)
        g: Tensor = self.proj_head(f)

        return g, f

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`~models.core.MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If :attr:`~models.core.MinervaModel.optimiser` is ``None``.

        Args:
            x (~torch.Tensor): Batch of input data to network.
            train (bool): Sets whether this shall be a training step or not. ``True`` for training step which will then
                clear the :attr:`~models.core.MinervaModel.optimiser`, and perform a backward pass of the network then
                update the :attr:`~models.core.MinervaModel.optimiser`. If ``False`` for a validation or testing step,
                these actions are not taken.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Loss computed by the loss function and a :class:`~torch.Tensor`
            with both projection's logits.
        """

        if self.optimiser is None:
            raise NotImplementedError("Optimiser has not been set!")

        assert self.criterion

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        loss: Tensor

        mix_precision: bool = True if self.scaler else False
        device_type = "cpu" if x.device.type == "cpu" else "cuda"

        # CUDA does not support ``torch.bfloat16`` while CPU does not support ``torch.float16`` for autocasting.
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        # Will enable mixed precision (if a Scaler has been set).
        with torch.amp.autocast_mode.autocast(
            device_type=device_type, dtype=autocast_dtype, enabled=mix_precision
        ):
            # Forward pass.
            z, z_a, z_b, _, _ = self.forward(x)

            # Compute Loss.
            loss = self.criterion(z_a, z_b)  # type: ignore[arg-type]

        # Performs a backward pass if this is a training step.
        if train:
            # Scales the gradients if using mixed precision training.
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

            else:
                loss.backward()
                self.optimiser.step()

        return loss, z


class SimCLR18(SimCLR):
    """:class:`SimCLR` network using a :class:`~models.resnet.ResNet18` :attr:`~SimCLR.backbone`."""

    backbone_name = "ResNet18"


class SimCLR34(SimCLR):
    """:class:`SimCLR` network using a :class:`~models.resnet.ResNet32` :attr:`~SimCLR.backbone`."""

    backbone_name = "ResNet34"


class SimCLR50(SimCLR):
    """:class:`SimCLR` network using a :class:`~models.resnet.ResNet50` :attr:`~SimCLR.backbone`."""

    backbone_name = "ResNet50"


class SimSiam(MinervaSiamese):
    """Base SimSiam class to be subclassed by SimSiam variants.

    Subclasses :class:`MinervaSiamese`.

    Attributes:
        backbone_name (str): Name of the :attr:`~SimSiam.backbone` within this module to use.
        backbone (~torch.nn.Module): Backbone of SimSiam that takes the imagery input and
            extracts learned representations.
        proj_head (~torch.nn.Module): Projection head that takes the learned representations from the backbone encoder.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int, int, int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.

        backbone_kwargs (dict[str, ~typing.Any]): Optional; Keyword arguments for the :attr:`~SimSiam.backbone`
            packed up into a dict.
    """

    __metaclass__ = abc.ABCMeta
    backbone_name = "ResNet18"

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 128,
        pred_dim: int = 512,
        scaler: Optional[GradScaler] = None,
        backbone_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(SimSiam, self).__init__(
            criterion=criterion, input_size=input_size, scaler=scaler
        )

        self.backbone: MinervaModel = get_model(self.backbone_name)(
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
        """Performs a forward pass of a single head of :class:`SimSiam` by using the forward methods of the backbone
        and feeding its output into the :attr:`~SimSiam.proj_head`.

        Args:
            x (~torch.Tensor): Batch of unpaired input data to the network.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Tuple of the feature vector outputted from :attr:`~SimSiam.proj_head`
            and the detached embedding vector from the :attr:`~SimSiam.backbone`.
        """
        z: Tensor = self.proj_head(torch.flatten(self.backbone(x)[0], start_dim=1))  # type: ignore[attr-defined]

        p: Tensor = self.predictor(z)

        return p, z.detach()

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`~models.core.MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If :attr:`~models.core.MinervaModel.optimiser` is ``None``.

        Args:
            x (~torch.Tensor): Batch of input data to network.
            train (bool): Sets whether this shall be a training step or not. ``True`` for training step which will then
                clear the :attr:`~models.core.MinervaModel.optimiser`, and perform a backward pass of the network then
                update the :attr:`~models.core.MinervaModel.optimiser`. If ``False`` for a validation or testing step,
                these actions are not taken.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Loss computed by the loss function and a :class:`~torch.Tensor`
            with both projection's logits.
        """

        if self.optimiser is None:
            raise NotImplementedError("Optimiser has not been set!")

        assert self.criterion

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        loss: Tensor

        mix_precision: bool = True if self.scaler else False
        device_type = "cpu" if x.device.type == "cpu" else "cuda"

        # CUDA does not support ``torch.bfloat16`` while CPU does not support ``torch.float16`` for autocasting.
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        # Will enable mixed precision (if a Scaler has been set).
        with torch.amp.autocast_mode.autocast(
            device_type=device_type, dtype=autocast_dtype, enabled=mix_precision
        ):
            # Forward pass.
            p, p_a, p_b, z_a, z_b = self.forward(x)

            # Compute Loss.
            loss = 0.5 * (self.criterion(z_a, p_b) + self.criterion(z_b, p_a))  # type: ignore[arg-type]

        # Performs a backward pass if this is a training step.
        if train:
            # Scales the gradients if using mixed precision training.
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

            else:
                loss.backward()
                self.optimiser.step()

        return loss, p


class SimSiam18(SimSiam):
    """:class:`SimSiam` network using a :class:`~models.resnet.ResNet18` :attr:`~SimSiam.backbone`."""

    backbone_name = "ResNet18"


class SimSiam34(SimSiam):
    """:class:`SimSiam` network using a :class:`~models.resnet.ResNet34` :attr:`~SimSiam.backbone`."""

    backbone_name = "ResNet34"


class SimSiam50(SimSiam):
    """:class:`SimSiam` network using a :class:`~models.resnet.ResNet50` :attr:`~SimSiam.backbone`."""

    backbone_name = "ResNet50"


class SimConv(MinervaSiamese):
    """Base SimConv class.

    Subclasses :class:`MinervaSiamese`.

    Attributes:
        backbone_name (str): Name of the :attr:`~SimCLR.backbone` within this module to use.
        backbone (~torch.nn.Module): Backbone of SimCLR that takes the imagery input and
            extracts learned representations.
        proj_head (~torch.nn.Module): Projection head that takes the learned representations from
            the :attr:`~SimCLR.backbone` encoder.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int, int, int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        backbone_kwargs (dict[str, ~typing.Any]): Optional; Keyword arguments for the :attr:`~SimCLR.backbone`
            packed up into a dict.
    """

    __metaclass__ = abc.ABCMeta
    backbone_name = "resnet18"

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, int, int] = (4, 256, 256),
        feature_dim: int = 2048,
        scaler: Optional[GradScaler] = None,
        backbone_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(SimConv, self).__init__(
            criterion=criterion, input_size=input_size, scaler=scaler
        )

        # Set of required kwargs for the `PSPNet` adapted from `minerva` style kwargs.
        new_kwargs = {
            "encoder_name": self.backbone_name,
            "psp_out_channels": feature_dim,
            "in_channels": input_size[0],
            "encoder_weights": None,
        }

        # Update the supplied kwargs with the required, adapted kwargs for the `PSPNet`.
        if backbone_kwargs is not None:
            new_kwargs.update(backbone_kwargs)

        self.backbone = MinervaWrapper(
            PSPEncoder,
            input_size=input_size,
            criterion=None,
            n_classes=None,
            scaler=None,
            **new_kwargs,
        )

        self.proj_head = nn.Sequential(
            nn.Conv2d(feature_dim, 512, 3, 2, padding=1),  # 3x3 Conv
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass of a single head of the network by using the forward methods of the
        :attr:`~SimCLR.backbone` and feeding its output into the :attr:`~SimCLR.proj_head`.

        Overwrites :meth:`MinervaSiamese.forward_single`

        Args:
            x (~torch.Tensor): Batch of unpaired input data to the network.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Tuple of the feature vector outputted from the
            :attr:`~SimCLR.proj_head` and the detached embedding vector from the :attr:`~SimCLR.backbone`.
        """
        f: Tensor = self.backbone(x)
        g: Tensor = self.proj_head(f)

        return g, f

    def step(self, x: Tensor, *args, train: bool = False) -> Tuple[Tensor, Tensor]:
        """Overwrites :class:`~models.core.MinervaModel` to account for paired logits.

        Raises:
            NotImplementedError: If :attr:`~models.core.MinervaModel.optimiser` is ``None``.

        Args:
            x (~torch.Tensor): Batch of input data to network.
            train (bool): Sets whether this shall be a training step or not. ``True`` for training step which will then
                clear the :attr:`~models.core.MinervaModel.optimiser`, and perform a backward pass of the network then
                update the :attr:`~models.core.MinervaModel.optimiser`. If ``False`` for a validation or testing step,
                these actions are not taken.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor]: Loss computed by the loss function and a :class:`~torch.Tensor`
            with both projection's logits.
        """

        if self.optimiser is None:  # pragma: no cover
            raise NotImplementedError("Optimiser has not been set!")

        assert self.criterion

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        loss: Tensor

        mix_precision: bool = True if self.scaler else False
        device_type = "cpu" if x.device.type == "cpu" else "cuda"

        # CUDA does not support ``torch.bfloat16`` while CPU does not support ``torch.float16`` for autocasting.
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        # Will enable mixed precision (if a Scaler has been set).
        with torch.amp.autocast_mode.autocast(
            device_type=device_type, dtype=autocast_dtype, enabled=mix_precision
        ):
            # Forward pass.
            z, z_a, z_b, _, _ = self.forward(x)

            # Compute Loss.
            loss = self.criterion(z_a, z_b)  # type: ignore[arg-type]

        # Performs a backward pass if this is a training step.
        if train:
            # Scales the gradients if using mixed precision training.
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

            else:
                loss.backward()
                self.optimiser.step()

        return loss, z
