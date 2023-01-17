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
"""Module containing Fully Convolutional Network (FCN) models."""

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
from torch import Tensor

from .core import MinervaModel, MinervaBackbone, bilinear_init, get_model

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"

__all__ = [
    "FCN8ResNet18",
    "FCN8ResNet34",
    "FCN8ResNet50",
    "FCN8ResNet101",
    "FCN8ResNet152",
    "FCN16ResNet18",
    "FCN16ResNet34",
    "FCN16ResNet50",
    "FCN32ResNet18",
    "FCN32ResNet34",
    "FCN32ResNet50",
]


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class _FCN(MinervaBackbone, ABC):
    """Base Fully Convolutional Network (FCN) class to be subclassed by FCN variants described in the FCN paper.

    Subclasses MinervaModel.

    Attributes:
        backbone (Module): Backbone of the FCN that takes the imagery input and
            extracts learned representations.
        decoder (Module): Decoder that takes the learned representations from the backbone encoder
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
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        backbone_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_name: str = "ResNet18",
        decoder_variant: Literal["32", "16", "8"] = "32",
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super(_FCN, self).__init__(
            criterion=criterion, input_shape=input_size, n_classes=n_classes
        )

        # Initialises the selected Minerva backbone.
        self.backbone: MinervaModel = get_model(backbone_name)(
            input_size=input_size, n_classes=n_classes, encoder=True, **backbone_kwargs  # type: ignore
        )

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:  # pragma: no cover
            self.backbone.load_state_dict(torch.load(backbone_weight_path))

            # Freezes the weights of backbone to avoid end-to-end training.
            if freeze_backbone:
                self.backbone.requires_grad_(False)

        # Determines the output shape of the backbone so the correct input shape is known
        # for the proceeding layers of the network.
        self.backbone.determine_output_dim()

        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)
        self.decoder = DCN(
            in_channel=backbone_out_shape[0],
            n_classes=n_classes,
            variant=decoder_variant,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the FCN by using the forward methods of the backbone and
        feeding its output into the forward for the decoder.

        Overwrites MinervaModel abstract method.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (Tensor): Input data to network.

        Returns:
            z (Tensor): segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.
        """
        z = self.backbone(x)
        z = self.decoder(z)

        assert isinstance(z, Tensor)
        return z


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

    Raises:
        NotImplementedError: Raised if ``variant`` does not match known types.
    """

    def __init__(
        self,
        in_channel: int = 512,
        n_classes: int = 21,
        variant: Literal["32", "16", "8"] = "32",
    ) -> None:

        super(DCN, self).__init__(n_classes=n_classes)
        self.variant: Literal["32", "16", "8"] = variant

        assert type(self.n_classes) is int

        # Common to all variants.
        self.relu = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.n_classes, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(self.n_classes)

        if variant == "32":
            self.DC32 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(64, 64),
                stride=(32, 32),
                dilation=1,
                padding=(16, 16),
            )
            self.DC32.weight.data = bilinear_init(self.n_classes, self.n_classes, 64)
            self.dbn32 = nn.BatchNorm2d(self.n_classes)

        if variant in ("16", "8"):
            self.Conv1x1_x3 = nn.Conv2d(
                int(in_channel / 2), self.n_classes, kernel_size=(1, 1)
            )
            self.DC2 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(4, 4),
                stride=(2, 2),
                dilation=1,
                padding=(1, 1),
            )
            self.DC2.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn2 = nn.BatchNorm2d(self.n_classes)

        if variant == "16":
            self.DC16 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(32, 32),
                stride=(16, 16),
                dilation=1,
                padding=(8, 8),
            )
            self.DC16.weight.data = bilinear_init(self.n_classes, self.n_classes, 32)
            self.dbn16 = nn.BatchNorm2d(self.n_classes)

        if variant == "8":
            self.Conv1x1_x2 = nn.Conv2d(
                int(in_channel / 4), self.n_classes, kernel_size=(1, 1)
            )

            self.DC4 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(4, 4),
                stride=(2, 2),
                dilation=1,
                padding=(1, 1),
            )
            self.DC4.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn4 = nn.BatchNorm2d(self.n_classes)

            self.DC8 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(16, 16),
                stride=(8, 8),
                dilation=1,
                padding=(4, 4),
            )
            self.DC8.weight.data = bilinear_init(self.n_classes, self.n_classes, 16)
            self.dbn8 = nn.BatchNorm2d(self.n_classes)

        if variant not in ("32", "16", "8"):
            raise NotImplementedError(
                f"Variant {self.variant} does not match known types"
            )

    def forward(self, x: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Performs a forward pass of the decoder. Depending on DCN variant, will take multiple inputs
        throughout pass from the encoder.

        Can be called directly as a method (e.g. model.forward()) or when data is parsed to model (e.g. model()).

        Args:
            x (tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): Input data to network.
                Should be from a backbone that supports output at multiple points e.g ResNet.

        Returns:
            Tensor segmentation mask with a channel for each class of the likelihoods the network places on
                each pixel input 'x' being of that class.

        Raises:
            NotImplementedError: Raised if ``variant`` does not match known types.
        """
        if self.variant not in ("32", "16", "8"):
            raise NotImplementedError(
                f"Variant {self.variant} does not match known types"
            )

        # Unpack outputs from the ResNet layers.
        x4, x3, x2, *_ = x

        # All DCNs have a common 1x1 Conv input block.
        z = self.bn1(self.relu(self.Conv1x1(x4)))

        # If DCN32, forward pass through DC32 and DBN32 and return output.
        if self.variant == "32":
            z = self.dbn32(self.relu(self.DC32(z)))
            assert isinstance(z, Tensor)
            return z

        # Common Conv1x1 layer to DCN16 & DCN8.
        x3 = self.bn1(self.relu(self.Conv1x1_x3(x3)))
        z = self.dbn2(self.relu(self.DC2(z)))

        z = z + x3

        # If DCN16, forward pass through DCN16 and DBN16 and return output.
        if self.variant == "16":
            z = self.dbn16(self.relu(self.DC16(z)))
            assert isinstance(z, Tensor)
            return z

        # If DCN8, continue through remaining layers to output.
        else:
            x2 = self.bn1(self.relu(self.Conv1x1_x2(x2)))
            z = self.dbn4(self.relu(self.DC4(z)))

            z = z + x2

            z = self.dbn8(self.relu(self.DC8(z)))

            assert isinstance(z, Tensor)
            return z


class FCN32ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN32ResNet18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            decoder_variant="32",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN32ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN32ResNet34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet34",
            decoder_variant="32",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN32ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN32 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN32ResNet50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet50",
            decoder_variant="32",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN16ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN16ResNet18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            decoder_variant="16",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN16ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN16ResNet34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet34",
            decoder_variant="16",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN16ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet50 backbone with a DCN16 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN16ResNet50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet50",
            decoder_variant="16",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet18(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet18 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet18, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet18",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet34(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet34 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet34, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet34",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet50(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet50 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet50, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet50",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet101(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet101 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet101, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet101",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )


class FCN8ResNet152(_FCN):
    """Fully Convolutional Network (FCN) using a ResNet152 backbone with a DCN8 decoder.

    Args:
        criterion: PyTorch loss function model will use.
        input_size (tuple[int] or list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        resnet_kwargs (dict): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **resnet_kwargs,
    ) -> None:

        super(FCN8ResNet152, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            backbone_name="ResNet152",
            decoder_variant="8",
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=resnet_kwargs,
        )
