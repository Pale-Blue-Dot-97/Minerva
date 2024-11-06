# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
"""Module containing a DeepLabV3 adapted for use in :mod:`minerva`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["DynamicDeepLabV3Plus", "MinervaDeepLabV3Plus"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Optional

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
)
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module

from minerva.models import MinervaWrapper


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class DynamicDeepLabV3Plus(smp.DeepLabV3Plus):
    """Adaptation of the :class:`segmentation_models_pytorch.DeepLabV3Plus`.

    Designed to be flexible and dynamic for pre-training and downstream applications.

    Args:
        n_classes (int): Number of classes in input data.
        encoder_name (str): Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth (int): A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights (str): One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        psp_out_channels (int): A number of filters in Spatial Pyramid
        psp_use_batchnorm (bool): If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        psp_dropout (float): Spatial dropout rate in [0, 1) used in Spatial Pyramid
        in_channels (int): Optional; Defines the shape of the input data. Typically in order of
            number of channels, image width, image height but may vary dependant on model specs.
        activation (str | callable): An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling (int): Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params (dict): Dictionary with parameters of the auxiliary output (classification head).
            Auxiliary output is build on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1]
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,
        in_channels: Optional[int] = None,
        n_classes: int = 1,
        activation: Optional[str | Callable[..., Any]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict[str, Any]] = None,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        encoder: bool = True,
        segmentation_on: bool = True,
        classification_on: bool = False,
    ) -> None:
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )

        self.encoder_mode = encoder
        self.segmentation_on = segmentation_on
        self.classification_on = classification_on

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:  # pragma: no cover
            backbone = torch.load(
                backbone_weight_path, map_location=torch.device("cpu")
            )

            self.encoder = backbone.encoder
            self.decoder = backbone.decoder

        # Will freeze the weights of the backbone to avoid end-to-end training if `freeze_backbone==True`.
        self.freeze_backbone(freeze_backbone)

    def make_segmentation_head(
        self,
        n_classes: int,
        activation: Optional[str | Callable[..., Any]] = None,
        upsampling: int = 8,
    ) -> None:
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=n_classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        self.encoder_mode = True
        self.segmentation_on = True

    def make_classification_head(self, aux_params: dict[str, Any]) -> None:
        # Makes the classification head.
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], **aux_params
        )

        # Initialise classification head weights.
        smp.base.initialization.initialize_head(self.classification_head)

        # Ensure the forward pass goes through the entire model.
        self.encoder_mode = True
        self.segmentation_on = True
        self.classification_on = True

    def set_encoder_mode(self, encode: bool) -> None:
        self.encoder_mode = encode

    def set_segmentation_on(self, on: bool) -> None:
        self.segmentation_on = on

    def set_classification_on(self, on: bool) -> None:
        self.classification_on = on

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the encoder so that the weights do not change while the rest of the model trains.

        Args:
            freeze (bool): Whether to 'freeze' the encoder (Set :meth:`~torch.Tensor.requires_grad_` to `False`).
                Defaults to `True`.

        .. versionadded:: 0.28
        """
        self.encoder.requires_grad_(False if freeze else True)

    def get_backbone(self) -> Module:
        """Get the backbone encoder of the PSP.

        Returns:
            Module: Backbone encoder.

        .. versionadded:: 0.29
        """
        assert isinstance(self.encoder, Module)
        return self.encoder

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...]:
        f = self.encoder(x)

        if not self.encoder_mode:
            return f[-1]  # type:ignore[no-any-return]

        g = self.decoder(*f)

        if self.segmentation_on:
            masks = self.segmentation_head(g)

        else:
            return g  # type:ignore[no-any-return]

        if self.classification_on:
            labels = self.classification_head(f[-1])
            return masks, labels

        return masks  # type:ignore[no-any-return]


class MinervaDeepLabV3Plus(MinervaWrapper):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int, ...]] = None,
        n_classes: int = 1,
        scaler: Optional[GradScaler] = None,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,
        activation: Optional[str | Callable[..., Any]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict[str, Any]] = None,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        encoder: bool = False,
        segmentation_on: bool = True,
        classification_on: bool = False,
    ) -> None:
        assert input_size is not None
        super().__init__(
            DynamicDeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                encoder_depth=encoder_depth,
                in_channels=input_size[0],
                n_classes=n_classes,
                activation=activation,
                upsampling=upsampling,
                aux_params=aux_params,
                backbone_weight_path=backbone_weight_path,
                freeze_backbone=freeze_backbone,
                encoder=encoder,
                segmentation_on=segmentation_on,
                classification_on=classification_on,
            ),
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            scaler=scaler,
        )

        self.upsampling = upsampling

    def _remake_classifier(self) -> None:
        self.model.make_segmentation_head(
            self.n_classes, upsampling=self.upsampling, activation=torch.nn.PReLU
        )
        if self.model.classification_on:
            self.make_classification_head(
                {"classes": self.n_classes, "activation": torch.nn.PReLU}
            )
