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
r"""Tests for :mod:`minerva.models.unet`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from typing import Tuple

import pytest
import torch
from torch import Tensor

from minerva.models import (
    MinervaModel,
    UNet,
    UNetR18,
    UNetR34,
    UNetR50,
    UNetR101,
    UNetR152,
)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def unet_test(
    model: MinervaModel,
    x: Tensor,
    y: Tensor,
    batch_size: int,
    n_classes: int,
    input_size: Tuple[int, int, int],
) -> None:
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape == input_size[1:]

    loss, z = model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (batch_size, n_classes, *input_size[1:])


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
@pytest.mark.parametrize(
    "model_cls",
    (
        UNetR18,
        UNetR34,
        UNetR50,
        UNetR101,
        UNetR152,
    ),
)
def test_unetrs(
    model_cls: MinervaModel,
    x_entropy_loss,
    random_rgbi_batch: Tensor,
    random_mask_batch: Tensor,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
) -> None:
    model: MinervaModel = model_cls(x_entropy_loss, rgbi_input_size)

    unet_test(
        model,
        random_rgbi_batch,
        random_mask_batch,
        std_batch_size,
        std_n_classes,
        rgbi_input_size,
    )


def test_unet(
    x_entropy_loss,
    random_rgbi_batch: Tensor,
    random_mask_batch: Tensor,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
) -> None:
    model = UNet(x_entropy_loss, input_size=rgbi_input_size)
    unet_test(
        model,
        random_rgbi_batch,
        random_mask_batch,
        std_batch_size,
        std_n_classes,
        rgbi_input_size,
    )

    bilinear_model = UNet(x_entropy_loss, input_size=rgbi_input_size, bilinear=True)
    unet_test(
        bilinear_model,
        random_rgbi_batch,
        random_mask_batch,
        std_batch_size,
        std_n_classes,
        rgbi_input_size,
    )
