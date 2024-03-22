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
"""Module containing PSPNetss adapted for use in :mod:`minerva`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "PSPEncoder",
    "DownstreamPSP",
]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Optional, Tuple, Union

import segmentation_models_pytorch as smp
import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module

from minerva.models import MinervaModel


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class PSPEncoder(smp.PSPNet):
    def forward(self, x: Tensor) -> Tensor:
        f = self.encoder(x)
        z = self.decoder(*f)
        assert isinstance(z, Tensor)
        return z


class DownstreamPSP(smp.PSPNet, MinervaModel):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        n_classes: Optional[int] = None,
        scaler: Optional[GradScaler] = None,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_depth: int = 5,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
        backbone_weight_path=None,
        freeze_backbone: bool = False,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            psp_out_channels=psp_out_channels,
            psp_use_batchnorm=psp_use_batchnorm,
            psp_dropout=psp_dropout,
            in_channels=input_size[0],
            classes=n_classes,
            activation=activation,
            upsampling=upsampling,
            aux_params=aux_params,
        )
        MinervaModel.__init__(
            self,
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            scaler=scaler,
        )

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:  # pragma: no cover
            backbone = torch.load(backbone_weight_path)
            self.encoder = backbone.encoder
            self.decoder = backbone.decoder

            # Freezes the weights of backbone to avoid end-to-end training.
            self.backbone.requires_grad_(False if freeze_backbone else True)
