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
"""Module containing a PSPNet adapted for use in :mod:`minerva`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["DynamicPSP", "MinervaPSP", "MinervaPSPUNet"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Optional

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import (
    ClassificationHead,
    Conv2dReLU,
    SegmentationHead,
)
from segmentation_models_pytorch.decoders.pspnet.decoder import PSPModule
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module

from minerva.models import MinervaWrapper

from .unet import OutConv, Up


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class DynamicPSP(smp.PSPNet):
    """Adaptation of the :class:`segmentation_models_pytorch.PSPNet`.

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
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
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
            psp_out_channels=psp_out_channels,
            psp_use_batchnorm=psp_use_batchnorm,
            psp_dropout=psp_dropout,
            in_channels=in_channels,
            classes=n_classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )

        self.encoder_mode = encoder
        self.segmentation_on = segmentation_on
        self.classification_on = classification_on

        self.psp_out_channels = psp_out_channels

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
            in_channels=self.psp_out_channels,
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


class MinervaPSP(MinervaWrapper):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int, ...]] = None,
        n_classes: int = 1,
        scaler: Optional[GradScaler] = None,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
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
            DynamicPSP(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                encoder_depth=encoder_depth,
                psp_out_channels=psp_out_channels,
                psp_use_batchnorm=psp_use_batchnorm,
                psp_dropout=psp_dropout,
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


class PSPDecoderBlock(Module):
    def __init__(
        self,
        encoder_channel,
        use_batchnorm: bool = True,
        out_channels: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channel,
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = Conv2dReLU(
            in_channels=encoder_channel * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x


class PSPUNetDecoder(Module):
    def __init__(
        self,
        encoder_channels: tuple[int, int, int, int, int, int],
        n_classes: int,
        use_batchnorm: bool = True,
        dropout: float = 0.2,
        bilinear: bool = False,
        small_encoder: bool = False,
    ):
        super().__init__()

        self.factor = 2 if bilinear else 1
        self.encoder_channels = encoder_channels
        self.small_encoder = small_encoder

        if self.small_encoder:
            self.psp_0 = PSPDecoderBlock(
                self.encoder_channels[0],
                use_batchnorm,
                self.encoder_channels[1] // 4,
                dropout=dropout,
            )
            self.psp_1 = PSPDecoderBlock(
                self.encoder_channels[1],
                use_batchnorm,
                self.encoder_channels[1] // 2,
                dropout=dropout,
            )
        else:
            self.psp_0 = PSPDecoderBlock(
                self.encoder_channels[0],
                use_batchnorm,
                self.encoder_channels[1] // 2,
                dropout=dropout,
            )
            self.psp_1 = PSPDecoderBlock(
                self.encoder_channels[1],
                use_batchnorm,
                self.encoder_channels[1],
                dropout=dropout,
            )
            self.psp_2_extra = PSPDecoderBlock(
                self.encoder_channels[2],
                use_batchnorm,
                self.encoder_channels[2] // 2,
                dropout=dropout,
            )

        self.psp_2 = PSPDecoderBlock(
            self.encoder_channels[2],
            use_batchnorm,
            self.encoder_channels[2],
            dropout=dropout,
        )
        self.psp_3 = PSPDecoderBlock(
            self.encoder_channels[3],
            use_batchnorm,
            self.encoder_channels[3],
            dropout=dropout,
        )
        self.psp_4 = PSPDecoderBlock(
            self.encoder_channels[4],
            use_batchnorm,
            self.encoder_channels[4],
            dropout=dropout,
        )
        self.psp_5 = PSPDecoderBlock(
            self.encoder_channels[5],
            use_batchnorm,
            self.encoder_channels[5],
            dropout=dropout,
        )

        # De-conv back up concating in skip connections from backbone.
        self.up1 = Up(
            self.encoder_channels[5], self.encoder_channels[4] * self.factor, bilinear
        )
        self.up2 = Up(
            self.encoder_channels[4], self.encoder_channels[3] * self.factor, bilinear
        )
        self.up3 = Up(
            self.encoder_channels[3], self.encoder_channels[2] * self.factor, bilinear
        )

        if self.small_encoder:
            self.up4 = Up(
                self.encoder_channels[2],
                self.encoder_channels[2] // 2 * self.factor,
                bilinear,
            )
            self.up5 = Up(
                self.encoder_channels[2] // 2,
                self.encoder_channels[2] // 4 * self.factor,
                bilinear,
            )
            self.outc = OutConv(self.encoder_channels[2] // 4 * self.factor, n_classes)
        else:
            self.up4 = Up(
                self.encoder_channels[2] // 2,
                self.encoder_channels[1] * self.factor,
                bilinear,
            )
            self.up5 = Up(
                self.encoder_channels[1],
                self.encoder_channels[1] // 2 * self.factor,
                bilinear,
            )
            self.outc = OutConv(self.encoder_channels[1] // 2 * self.factor, n_classes)

    def _remake_classifier(self, n_classes: int) -> None:
        if self.small_encoder:
            self.outc = OutConv(self.encoder_channels[2] // 4 * self.factor, n_classes)
        else:
            self.outc = OutConv(self.encoder_channels[1] // 2 * self.factor, n_classes)

    def forward(self, *x: Tensor) -> Tensor:
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
        x0, x1, x2, x3, x4, x5 = x

        # Concats and upsamples the outputs of the resnet.
        x = self.up1(self.psp_5(x5), self.psp_4(x4))
        x = self.up2(x, self.psp_3(x3))
        x = self.up3(x, self.psp_2(x2))

        if not self.small_encoder:
            x = self.psp_2_extra(x)

        x = self.up4(x, self.psp_1(x1))
        x = self.up5(x, self.psp_0(x0))

        # Reduces the latent channels to the number of classes for the ouput tensor.
        logits: Tensor = self.outc(x)

        assert isinstance(logits, Tensor)
        return logits


class MinervaPSPUNet(MinervaWrapper):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int, ...]] = None,
        n_classes: int = 1,
        scaler: Optional[GradScaler] = None,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
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
            DynamicPSP(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                encoder_depth=encoder_depth,
                psp_out_channels=psp_out_channels,
                psp_use_batchnorm=psp_use_batchnorm,
                psp_dropout=psp_dropout,
                in_channels=input_size[0],
                n_classes=n_classes,
                activation=activation,
                upsampling=upsampling,
                aux_params=aux_params,
                backbone_weight_path=backbone_weight_path,
                freeze_backbone=freeze_backbone,
                encoder=encoder,
                segmentation_on=False,
                classification_on=False,
            ),
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            scaler=scaler,
        )

        # Flag for when to concatenate the output of the Conv1 layer of the resnet with the upsampling
        # output of the decoder. ResNet50s and larger use a `Bottleneck` type residual block which quadruples
        # the number of feature maps compared to the `Basic` blocks of `ResNet18` and `ResNet34`.
        # This in turn affects the sizes of the output from decoding layers which requires a reordering of operations.
        small_encoder = (
            True
            if encoder_name in ("ResNet18", "ResNet34", "resnet18", "resnet34")
            else False
        )

        self.model.decoder = PSPUNetDecoder(
            self.encoder.out_channels,
            n_classes,
            use_batchnorm=True,
            dropout=psp_dropout,
            bilinear=False,
            small_encoder=small_encoder,
        )

    def _remake_classifier(self) -> None:
        self.model.decoder._remake_classifier(self.n_classes)
        self.model.segmentation_head = torch.nn.Identity()
        if self.model.classification_on:
            self.model.make_classification_head(
                {"classes": self.n_classes, "activation": torch.nn.PReLU}
            )
