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
"""Module containing neural network model classes."""

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
from torch import Tensor

from minerva.models import MinervaModel

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


class _FCN(MinervaModel, ABC):
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
