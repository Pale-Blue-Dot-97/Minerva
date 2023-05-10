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
#                                                     GLOBALS
# =====================================================================================================================
input_size = (4, 64, 64)
batch_size = 2
n_classes = 8

x = torch.rand((batch_size, *input_size))
y = torch.randint(0, n_classes, (batch_size, *input_size[1:]))  # type: ignore[attr-defined]


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def unet_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
    optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

    test_model.set_optimiser(optimiser)

    test_model.determine_output_dim()
    assert test_model.output_shape == input_size[1:]

    loss, z = test_model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (batch_size, n_classes, *input_size[1:])


def test_unet(x_entropy_loss) -> None:
    model = UNet(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)

    bilinear_model = UNet(x_entropy_loss, input_size=input_size, bilinear=True)
    unet_test(bilinear_model, x, y)


def test_unetr18(x_entropy_loss) -> None:
    model = UNetR18(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)


def test_unetr34(x_entropy_loss) -> None:
    model = UNetR34(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)


def test_unetr50(x_entropy_loss) -> None:
    model = UNetR50(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)


def test_unetr101(x_entropy_loss) -> None:
    model = UNetR101(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)


def test_unetr152(x_entropy_loss) -> None:
    model = UNetR152(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)
