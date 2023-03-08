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
"""Module containing UNet models. Most code from https://github.com/milesial/Pytorch-UNet"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"

__all__ = [
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
    "UNet",
    "UNetR",
    "UNetR18",
    "UNetR34",
    "UNetR50",
    "UNetR101",
    "UNetR152",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn.modules as nn
from torch import Tensor
from torch.nn.modules import Module

from .core import MinervaModel, get_model


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class DoubleConv(Module):
    """Applies a double convolution to the input ``(convolution => [BN] => ReLU) * 2``

    Adapted from https://github.com/milesial/Pytorch-UNet for :mod:`minerva`.

    Attributes:
        double_conv (~torch.nn.Module): Double convolutions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int): Optional; Intermediate number of channels between convolutions.
            If ``None``, set to ``out_channels``.
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies the double convolutions to the input.

        Args:
            x (~torch.Tensor): Input tensor.

        Returns:
            ~torch.Tensor: Input passed through the double convolutions.
        """
        x = self.double_conv(x)
        assert isinstance(x, Tensor)
        return x


class Down(Module):
    """Downscaling with maxpool then double convolution.

    Adapted from https://github.com/milesial/Pytorch-UNet for :mod:`minerva`.

    Attributes:
        maxpool_conv (~torch.nn.Module): :class:`~torch.nn.Sequential` of :class:`~torch.nn.MaxPool2d`
            then :class:`DoubleConv`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies a maxpool then double convolution to the input.

        Args:
            x (~torch.Tensor): Input tensor.

        Returns:
            ~torch.Tensor: Input tensor passed through maxpooling then double convolutions.
        """
        x = self.maxpool_conv(x)
        assert isinstance(x, Tensor)
        return x


class Up(Module):
    """Upscaling then double convolution.

    Adapted from https://github.com/milesial/Pytorch-UNet for use in :mod:`minerva`.

    Attributes:
        up (~torch.nn.Module): Upsampling if ``bilinear==True``, else transpose convolutional layer.
        conv (DoubleConv): Double convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool): Optional;

    """

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super().__init__()

        self.up: Module

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Applies upscaling to ``x1``, concats ``x1`` with ``x2`` then applies :class:`DoubleConv` to the result.

        Args:
            x1 (~torch.Tensor): Input tensor 1 to be upscaled to match ``x2``.
            x2 (~torch.Tensor): Input tensor 2 to be concated with upscaled ``x1``
                and passed through :class:`DoubleConv`.

        Returns:
            ~torch.Tensor: Output tensor of the the upscaling and double convolutions.
        """

        x1 = self.up(x1)
        # input is CHW
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])

        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)  # type: ignore[attr-defined]

        x = self.conv(x)

        assert isinstance(x, Tensor)
        return x


class OutConv(Module):
    """``1x1`` convolution to change the number of channels down of the input.

    Adapted from https://github.com/milesial/Pytorch-UNet for :mod:`minerva`.

    Attributes:
        conv (~torch.nn.Module): ``1x1`` convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Passes the input tensor through the 1x1 convolutional layer.

        Args:
            x (~torch.Tensor): Input tensor.

        Returns:
            ~torch.Tensor: Input passed reduced from ``in_channels`` to ``out_channels``.
        """
        x = self.conv(x)
        assert isinstance(x, Tensor)
        return x


class UNet(MinervaModel):
    """UNet model. Good for segmentation problems.

    Adapted from https://github.com/milesial/Pytorch-UNet for use in :mod:`minerva`.

    Attributes:
        bilinear (bool):
        inc (DoubleConv): Double convolutional layers as input to the network to 64 channels.
        down1 (Down): Downscale then double convolution from 64 channels to 128.
        down2 (Down): Downscale then double convolution from 128 channels to 256.
        down3 (Down): Downscale then double convolution from 256 channels to 512.
        down4 (Down): Downscale then double convolution from 512 channels to 1024 and the latent space.
        up1 (Up): First upsample then concatenated input double de-convolutional layer.
        up2 (Up): Second upsample then concatenated input double de-convolutional layer.
        up3 (Up): Third upsample then concatenated input double de-convolutional layer.
        up4 (Up): Fourth upsample then concatenated input double de-convolutional layer.
        outc (OutConv): 1x1 output convolutional layer.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int, ...]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        bilinear (bool): Optional;
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        bilinear: bool = False,
    ) -> None:
        super(UNet, self).__init__(
            criterion=criterion, input_size=input_size, n_classes=n_classes
        )

        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(input_size[0], 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the :class:`UNet`.

        Adapted from https://github.com/milesial/Pytorch-UNet for :mod:`minerva`.

        Args:
            x (~torch.Tensor): Input tensor to the UNet.

        Returns:
            ~torch.Tensor: Output from the UNet.
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits: Tensor = self.outc(x)

        assert isinstance(logits, Tensor)
        return logits


class UNetR(MinervaModel):
    """UNet model which incorporates a :class:`~models.resnet.ResNet` as the encoder.

    Attributes:
        backbone_name (str): Name of the backbone class.
        backbone (~torch.nn.Module): Backbone of the FCN that takes the imagery input and
            extracts learned representations.
        up1 (Up): First upsample then concatenated input double de-convolutional layer.
        up2 (Up): Second upsample then concatenated input double de-convolutional layer.
        up3 (Up): Third upsample then concatenated input double de-convolutional layer.
        upsample1 (~torch.nn.Module): First upsample from output of ``up3``.
        upsample2 (~torch.nn.Module): Second upsample from output of ``up3`` to match input spatial size.
        outc (OutConv): 1x1 output convolutional layer.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int, ...]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        bilinear (bool): Optional;
        backbone_name (str): Optional; Name of the backbone within this module to use for the UNet.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        backbone_kwargs (dict[str, ~typing.Any]): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    backbone_name = "ResNet18"

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        bilinear: bool = False,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        backbone_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(UNetR, self).__init__(
            criterion=criterion, input_size=input_size, n_classes=n_classes
        )

        factor = 2 if bilinear else 1

        # Initialises the selected Minerva backbone.
        self.backbone: MinervaModel = get_model(self.backbone_name)(  # type: ignore[arg-type]
            input_size=input_size, n_classes=n_classes, encoder=True, **backbone_kwargs
        )

        # Flag for when to concatenate the output of the Conv1 layer of the resnet with the upsampling
        # output of the decoder. ResNet50s and larger use a `Bottleneck` type residual block which quadruples
        # the number of feature maps compared to the `Basic` blocks of `ResNet18` and `ResNet34`.
        # This in turn affects the sizes of the output from decoding layers which requires a reordering of operations.
        self.early_cat = (
            True if self.backbone_name in ("ResNet18", "ResNet34") else False
        )

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:
            self.backbone.load_state_dict(torch.load(backbone_weight_path))

            # Freezes the weights of backbone to avoid end-to-end training.
            if freeze_backbone:
                self.backbone.requires_grad_(False)

        # Determines the output shape of the backbone so the correct input shape is known
        # for the proceeding layers of the network.
        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)

        latent_channels = backbone_out_shape[0]

        # De-conv back up concating in skip connections from backbone.
        self.up1 = Up(latent_channels, latent_channels // 2 * factor, bilinear)
        self.up2 = Up(latent_channels // 2, latent_channels // 4 * factor, bilinear)
        self.up3 = Up(latent_channels // 4, latent_channels // 8 * factor, bilinear)

        self.upsample1 = nn.ConvTranspose2d(
            latent_channels // 8, latent_channels // 16, kernel_size=2, stride=2
        )
        self.upsample2 = nn.ConvTranspose2d(
            latent_channels // 16, latent_channels // 32, kernel_size=2, stride=2
        )

        self.outc = OutConv(latent_channels // 32, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the UNet using ``backbone``.

        Passes the input tensor to the ``backbone``. Then passes the output tensors from the
        resnet residual blocks to corresponding stages of the decoder, upsampling to create
        an output with the same spatial size as the input size.

        Args:
            x (~torch.Tensor): Input tensor to the :class:`UNetR`.

        Returns:
            ~torch.Tensor: Output from the :class:`UNetR`.
        """
        # Output tensors from the residual blocks of the resnet.
        x4, x3, x2, x1, x0 = self.backbone(x)

        # Concats and upsamples the outputs of the resnet.
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Add the upsampled and deconv tensor to the output of the input convolutional layer of the resnet.
        if self.early_cat:
            x = x + x0

        # Upsample this result to match the input spatial size.
        x = self.upsample1(x)
        x = self.upsample2(x)

        # Reduces the latent channels to the number of classes for the ouput tensor.
        logits: Tensor = self.outc(x)

        assert isinstance(logits, Tensor)
        return logits


class UNetR18(UNetR):
    """UNet with a :class:`~models.resnet.ResNet18` as the backbone."""

    backbone_name = "ResNet18"


class UNetR34(UNetR):
    """UNet with a :class:`~models.resnet.ResNet34` as the backbone."""

    backbone_name = "ResNet34"


class UNetR50(UNetR):
    """UNet with a :class:`~models.resnet.ResNet50` as the backbone."""

    backbone_name = "ResNet50"


class UNetR101(UNetR):
    """UNet with a :class:`~models.resnet.ResNet101` as the backbone."""

    backbone_name = "ResNet101"


class UNetR152(UNetR):
    """UNet with a :class:`~models.resnet.ResNet152` as the backbone."""

    backbone_name = "ResNet152"
