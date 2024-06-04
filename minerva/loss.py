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
"""Library of specialised loss functions for :mod:`minerva`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["SegBarlowTwinsLoss", "AuxCELoss"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import importlib

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules import Module
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import BarlowTwinsLoss
except (OSError, NewConnectionError, MaxRetryError):  # pragma: no cover
    BarlowTwinsLoss = getattr(
        importlib.import_module("lightly.loss"), "BarlowTwinsLoss"
    )


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class SegBarlowTwinsLoss(BarlowTwinsLoss):
    """Adaptation of :class:`lightly.loss.BarlowTwinsLoss` for segmented Siamese network."""

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Computes the Barlow Twins Loss between projection A and B but accounts for the segmentation mask shapes.

        Args:
            z_a (~torch.Tensor): Projection A from the Segmentation Barlow Twins network.
            z_b (~torch.Tensor): Projection B from the Segmentation Barlow Twins network.

        Returns:
            Tensor: The loss computed between A and B.
        """

        ch = z_a.size()[1]

        # Reshapes the A and B representations from the convolutional projector from [B, C, H, W] to [B, H, W, C]
        z_a = z_a.permute(0, 2, 3, 1).reshape(-1, ch)
        z_b = z_b.permute(0, 2, 3, 1).reshape(-1, ch)

        # Then just use the standard ``BarlowTwinsLoss.forward`` with the reshaped representations.
        loss = super().forward(z_a, z_b)
        assert isinstance(loss, Tensor)
        return loss


class AuxCELoss(Module):
    """Cross Entropy loss for using auxilary inputs for use in PSPNets.

    Adapted from the PSPNet paper and for use in PyTorch.

    Source: https://github.com/xitongpu/PSPNet/blob/main/src/model/cell.py
    """

    def __init__(self, ignore_index=255, alpha: float = 0.4) -> None:
        super().__init__()
        self.loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.alpha = alpha

    def forward(self, net_out: Tensor, target: Tensor) -> Tensor:
        # If length of ``net_out==2``, assume it is a tuple of the output of
        # the segmentation head (predict) and the classification head (predict_aux).
        if len(net_out) == 2:
            predicted_masks, predicted_labels = net_out

            labels = torch.mode(torch.flatten(target, start_dim=1)).values

            # Calculate the CrossEntropyLoss of both outputs.
            CE_loss = self.loss(predicted_masks, target)
            CE_loss_aux = self.loss(predicted_labels, labels)

            # Weighted sum the losses together and return
            loss = CE_loss + (self.alpha * CE_loss_aux)
            assert isinstance(loss, Tensor)
            return loss

        # Else, assume this is just the loss from the segmentation head
        # so perform a standard CrossEntropyLoss op.
        loss = self.loss(net_out, target)
        assert isinstance(loss, Tensor)
        return loss
