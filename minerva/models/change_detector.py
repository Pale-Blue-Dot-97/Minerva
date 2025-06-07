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
"""Module for change detection models"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["ChangeSceneDetector", "ChangeSegmentationDetector"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from pathlib import Path
from typing import Any, Optional, Sequence, Callable

import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module
from segmentation_models_pytorch.base import SegmentationHead

from minerva.utils.utils import func_by_str

from .core import MinervaModel
from .unet import DynamicUNet


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class ChangeSceneDetector(MinervaModel):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int]] = None,
        n_classes: int = 1,
        scaler: Optional[GradScaler] = None,
        fc_dim: int = 512,
        intermediate_dim: int = 256,
        encoder_on: bool = False,
        filter_dim: int = 0,
        freeze_backbone: bool = False,
        backbone_weight_path: Optional[str | Path] = None,
        backbone_args: dict[str, Any] = {},
        clamp_outputs: bool = False,
    ) -> None:
        super().__init__(criterion, input_size, n_classes, scaler)

        _backbone = func_by_str(backbone_args.pop("module"), backbone_args.pop("name"))

        backbone: Module = _backbone(**backbone_args)

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:  # pragma: no cover
            backbone.load_state_dict(
                torch.load(backbone_weight_path, map_location=torch.device("cpu"))
            )

            # Freezes the weights of backbone to avoid end-to-end training.
            backbone.requires_grad_(False if freeze_backbone else True)

        # Extract the actual encoder network from the backbone.
        if hasattr(backbone, "encoder"):
            backbone = backbone.encoder

        self.backbone = backbone

        self.encoder_on = encoder_on
        self.filter_dim = filter_dim
        self.fc_dim = fc_dim
        self.intermediate_dim = intermediate_dim

        # Will clamp the outputs of the classification head to the range (0, 1).
        self.clamp_outputs = clamp_outputs

        self._make_classification_head()

    def _make_classification_head(self) -> None:
        assert self.n_classes is not None
        self.classification_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * self.fc_dim, self.intermediate_dim),
            torch.nn.BatchNorm1d(self.intermediate_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Linear(self.intermediate_dim, self.n_classes),
        )

    def _remake_classifier(self) -> None:
        self._make_classification_head()

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. ``model.forward``) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor: Likelihoods the network places on the
            input ``x`` being of each class.
        """
        x_0, x_1 = x[0], x[1]

        f_0 = self.backbone(x_0)
        f_1 = self.backbone(x_1)

        if self.encoder_on:
            f_0 = f_0[self.filter_dim]
            f_1 = f_1[self.filter_dim]

        f = torch.cat((f_0, f_1), 1)

        z: Tensor = self.classification_head(f)

        assert isinstance(z, Tensor)

        if self.clamp_outputs:
            return z.clamp(0, 1)
        else:
            return z


class ChangeSegmentationDetector(MinervaModel):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int]] = None,
        n_classes: int = 1,
        scaler: Optional[GradScaler] = None,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = None,
        encoder_weight_path: Optional[str | Path] = None,
        encoder_depth: int = 5,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        freeze_backbone: bool = False,
        activation: Optional[str | Callable[..., Any]] = None,
        backbone_weight_path: Optional[str | Path] = None,
        aux_params: Optional[dict[str, Any]] = None,
        clamp_outputs: bool = True,
    ) -> None:
        super().__init__(criterion, input_size, n_classes, scaler)

        assert input_size is not None

        self.backbone = DynamicUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            in_channels=input_size[0],
            n_classes=n_classes,
            decoder_channels=decoder_channels,
            activation=activation,
            aux_params=aux_params,
            encoder_weight_path=encoder_weight_path,
            backbone_weight_path=backbone_weight_path,
            freeze_backbone=freeze_backbone,
            encoder=True,
            segmentation_on=True,
            classification_on=False,
            )

        # Rebuild the segmentation head to handle both epochs of images and output the change detection map.
        self.backbone.segmentation_head = SegmentationHead(
            in_channels=2*decoder_channels[-1],
            out_channels=n_classes,
            activation=activation,
            kernel_size=3,
        )

        self.clamp_outputs = clamp_outputs

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the :class:`ResNet`.

        Can be called directly as a method (e.g. ``model.forward``) or when data is parsed
        to model (e.g. ``model()``).

        Args:
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor: Likelihoods the network places on the
            input ``x`` being of each class.
        """
        x_0, x_1 = x[0], x[1]

        # Embeddings
        f_0 = self.backbone.encoder(x_0)
        f_1 = self.backbone.encoder(x_1)

        g_0 = self.backbone.decoder(*f_0)
        g_1 = self.backbone.decoder(*f_1)

        g = torch.cat((g_0, g_1), 1)

        z = self.backbone.segmentation_head(g)

        assert isinstance(z, Tensor)

        z = z.moveaxis(-3, -1)  # Move the class dimension to the last position.

        if self.clamp_outputs:
            return z.clamp(0, 1)
        else:
            return z
