# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

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
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["SegBarlowTwinsLoss"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import importlib

import torch
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import BarlowTwinsLoss
except (OSError, NewConnectionError, MaxRetryError):
    BarlowTwinsLoss = getattr(
        importlib.import_module("lightly.loss"), "BarlowTwinsLoss"
    )


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class SegBarlowTwinsLoss(BarlowTwinsLoss):
    """Adaptation of :class:`lightly.loss.BarlowTwinsLoss` for segmented Siamese network."""

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # Reshapes the A and B representations from the convolutional projector from [B, C, H, W] to [B * C, H * W].
        z_a = z_a.reshape(z_a.size()[0] * z_a.size()[1], z_a.size()[2] * z_a.size()[3])
        z_b = z_b.reshape(z_b.size()[0] * z_b.size()[1], z_b.size()[2] * z_b.size()[3])

        # Then just use the standard ``BarlowTwinsLoss.forward`` with the reshaped representations.
        return super().forward(z_a, z_b)
