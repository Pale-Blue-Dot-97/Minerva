# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
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
