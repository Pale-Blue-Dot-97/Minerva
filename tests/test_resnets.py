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
r"""Tests for :mod:`minerva.models.resnet`.
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
from torch import LongTensor, Tensor
from torchvision.models.resnet import BasicBlock

from minerva.models import (
    MinervaModel,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from minerva.models.resnet import ResNet, _preload_weights


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def resnet_test(
    model: MinervaModel, x: Tensor, y: LongTensor, batch_size: int, n_classes: int
) -> None:
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    loss, z = model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (batch_size, n_classes)


@pytest.mark.parametrize(
    "model_cls",
    (
        ResNet18,
        ResNet34,
        ResNet50,
        ResNet101,
        ResNet152,
    ),
)
@pytest.mark.parametrize("zero_init", (True, False))
def test_resnets(
    model_cls: MinervaModel,
    zero_init: bool,
    x_entropy_loss,
    random_rgbi_batch: Tensor,
    random_scene_classification_batch: LongTensor,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
) -> None:
    model: MinervaModel = model_cls(
        x_entropy_loss, input_size=rgbi_input_size, zero_init_residual=zero_init
    )
    resnet_test(
        model,
        random_rgbi_batch,
        random_scene_classification_batch,
        std_batch_size,
        std_n_classes,
    )


def test_resnet() -> None:
    assert isinstance(ResNet(BasicBlock, [2, 2, 2, 2], groups=2), ResNet)

    with pytest.raises(ValueError):
        _ = ResNet18(replace_stride_with_dilation=(True, False))  # type: ignore[arg-type]


def test_replace_stride(
    x_entropy_loss,
    random_rgbi_batch: Tensor,
    random_scene_classification_batch: LongTensor,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
) -> None:
    for model in (
        ResNet50(x_entropy_loss, input_size=rgbi_input_size),
        ResNet50(
            x_entropy_loss,
            input_size=rgbi_input_size,
            replace_stride_with_dilation=(True, True, False),
            zero_init_residual=True,
        ),
    ):
        resnet_test(
            model,
            random_rgbi_batch,
            random_scene_classification_batch,
            std_batch_size,
            std_n_classes,
        )


def test_resnet_encoder(
    x_entropy_loss,
    random_rgbi_batch: Tensor,
    rgbi_input_size: Tuple[int, int, int],
) -> None:
    encoder = ResNet18(x_entropy_loss, input_size=rgbi_input_size, encoder=True)
    optimiser = torch.optim.SGD(encoder.parameters(), lr=1.0e-3)

    encoder.set_optimiser(optimiser)

    encoder.determine_output_dim()
    print(encoder.output_shape)
    assert encoder.output_shape == (512, 2, 2)

    assert len(encoder(random_rgbi_batch)) == 5


def test_preload_weights() -> None:
    resnet = ResNet(BasicBlock, [2, 2, 2, 2])
    new_resnet = _preload_weights(resnet, None, (4, 32, 32), encoder_on=False)

    assert resnet == new_resnet
