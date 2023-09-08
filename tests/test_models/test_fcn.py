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
r"""Tests for :mod:`minerva.models.fcn`.
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
    FCN8ResNet18,
    FCN8ResNet34,
    FCN8ResNet50,
    FCN8ResNet101,
    FCN8ResNet152,
    FCN16ResNet18,
    FCN16ResNet34,
    FCN16ResNet50,
    FCN32ResNet18,
    FCN32ResNet34,
    FCN32ResNet50,
    MinervaModel,
    ResNet18,
)
from minerva.models.fcn import DCN


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
@pytest.mark.parametrize(
    "model_cls",
    (
        FCN8ResNet18,
        FCN8ResNet34,
        FCN8ResNet50,
        FCN8ResNet101,
        FCN8ResNet152,
        FCN16ResNet18,
        FCN16ResNet34,
        FCN16ResNet50,
        FCN32ResNet18,
        FCN32ResNet34,
        FCN32ResNet50,
    ),
)
def test_fcn(
    model_cls: MinervaModel,
    x_entropy_loss,
    random_rgbi_batch: Tensor,
    random_mask_batch: Tensor,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
) -> None:
    model: MinervaModel = model_cls(x_entropy_loss, input_size=rgbi_input_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape == rgbi_input_size[1:]

    loss, z = model.step(random_rgbi_batch, random_mask_batch, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (std_batch_size, std_n_classes, *rgbi_input_size[1:])


def test_dcn(random_rgbi_batch: Tensor) -> None:
    with pytest.raises(
        NotImplementedError, match="Variant 42 does not match known types"
    ):
        _ = DCN(variant="42")  # type: ignore[arg-type]

    dcn = DCN(variant="32")
    resnet = ResNet18()
    with pytest.raises(
        NotImplementedError, match="Variant 42 does not match known types"
    ):
        dcn.variant = "42"  # type: ignore[assignment]
        _ = dcn.forward(resnet(random_rgbi_batch))
