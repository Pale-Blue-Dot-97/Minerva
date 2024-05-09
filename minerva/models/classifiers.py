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
""""""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["FlexiSceneClassifier"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module

from minerva.utils.utils import func_by_str

from .core import FilterOutputs, MinervaBackbone


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class FlexiSceneClassifier(MinervaBackbone):
    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[Tuple[int]] = None,
        n_classes: int = 1,
        scaler: Optional[GradScaler] = None,
        fc_dim: int = 512,
        encoder_on: bool = False,
        backbone_args: Dict[str, Any] = {},
    ) -> None:
        super().__init__(criterion, input_size, n_classes, scaler)

        _backbone = func_by_str(backbone_args.pop("module"), backbone_args.pop("name"))

        self.backbone = _backbone(**backbone_args)

        self.encoder_on = encoder_on
        self.filter = FilterOutputs(-1)

        self.classification_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(fc_dim, n_classes),
        )

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
        f = self.backbone(x)

        if self.encoder_on:
            f = self.filter(f)

        return self.classification_head(f)
