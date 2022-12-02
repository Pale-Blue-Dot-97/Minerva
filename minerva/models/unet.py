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
"""Module containing UNet models. Most code from https://github.com/milesial/Pytorch-UNet"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from abc import ABC
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
    Sequence,
)

import torch
import torch.nn.modules as nn
import torch.nn.functional as F
from torch import Tensor

from .core import MinervaModel, bilinear_init
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"

__all__ = []

# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
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
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super().__init__()

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
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(MinervaModel, ABC):
    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_name: str = "ResNet18",
        bilinear: bool = False,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(UNet, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        factor = 2 if bilinear else 1

        # Initialises the selected Minerva backbone.
        self.backbone: MinervaModel = globals()[backbone_name](
            input_size=input_size, n_classes=n_classes, encoder=True, **backbone_kwargs
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

        print(backbone_out_shape)

        self.up1 = Up(backbone_out_shape[0], 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x4, x3, x2, x1, x0 = self.backbone(x)

        print(f"{x0.size()=}")
        print(f"{x1.size()=}")
        print(f"{x2.size()=}")
        print(f"{x3.size()=}")
        print(f"{x4.size()=}")

        x = self.up1(x4, x3)
        print(f"{x.size()=}")

        x = self.up2(x, x2)
        print(f"{x.size()=}")

        x = self.up3(x, x1)
        print(f"{x.size()=}")

        x = self.up4(x, x0)
        print(f"{x.size()=}")

        logits = self.outc(x)

        return logits


class UNetR18(UNet):
    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(UNetR18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )
