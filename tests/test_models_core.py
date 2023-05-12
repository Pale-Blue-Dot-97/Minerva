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
r"""Tests for :mod:`minerva.models.core`.
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
import importlib

import internet_sabotage
import numpy as np
import pytest
import torch
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import NTXentLoss
except (OSError, NewConnectionError, MaxRetryError):
    NTXentLoss = getattr(importlib.import_module("lightly.loss"), "NTXentLoss")
try:
    from lightly.models import ResNetGenerator
except (OSError, NewConnectionError, MaxRetryError):
    NTXentLoss = getattr(importlib.import_module("lightly.loss"), "NTXentLoss")
from torch import LongTensor, Tensor
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from minerva.models import (
    MinervaWrapper,
    SimCLR18,
    bilinear_init,
    get_output_shape,
    get_torch_weights,
)
from minerva.models.__depreciated import MLP


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_minerva_model(x_entropy_loss) -> None:
    x = torch.rand(16, (288))
    y = torch.LongTensor(np.random.randint(0, 8, size=16))

    with pytest.raises(NotImplementedError, match="Criterion has not been set!"):
        model_fail = MLP()
        optimiser = torch.optim.SGD(model_fail.parameters(), lr=1.0e-3)

        model_fail.set_optimiser(optimiser)
        _ = model_fail.step(x, y, train=True)

    model = MLP(x_entropy_loss)

    with pytest.raises(NotImplementedError, match="Optimiser has not been set!"):
        _ = model.step(x, y, train=True)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    for mode in ("train", "val", "test"):
        if mode == "train":
            loss, z = model.step(x, y, train=True)
        else:
            loss, z = model.step(x, y, train=False)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (16, 8)


def test_minerva_backbone() -> None:
    loss_func = NTXentLoss(0.3)
    input_size = (4, 64, 64)

    model = SimCLR18(loss_func, input_size=input_size)

    assert isinstance(model.get_backbone(), Module)


def test_minerva_wrapper(x_entropy_loss) -> None:
    input_size = (3, 32, 32)
    n_classes = 8
    model = MinervaWrapper(
        ResNetGenerator,
        x_entropy_loss,
        input_size=input_size,
        n_classes=n_classes,
        num_classes=n_classes,
        name="resnet-9",
    )

    assert isinstance(repr(model), str)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    x = torch.rand(6, *input_size)
    y = LongTensor(np.random.randint(0, 8, size=6))

    for mode in ("train", "val", "test"):
        if mode == "train":
            loss, z = model.step(x, y, train=True)
        else:
            loss, z = model.step(x, y, train=False)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (6, 8)


def test_get_torch_weights() -> None:
    try:
        weights1 = get_torch_weights("ResNet18_Weights.IMAGENET1K_V1")

        try:
            assert isinstance(weights1, WeightsEnum)

            with internet_sabotage.no_connection():
                weights2 = get_torch_weights("ResNet18_Weights.IMAGENET1K_V1")

                assert isinstance(weights2, WeightsEnum)

                weights3 = get_torch_weights("ResNet50_Weights.IMAGENET1K_V1")

                assert weights3 is None

        except (AssertionError, ImportError) as err:
            print(err)
    except ImportError as err:
        print(err)


def test_get_output_shape(exp_mlp) -> None:
    output_shape = get_output_shape(exp_mlp, 64)

    assert output_shape == 8


def test_bilinear_init() -> None:
    weights = bilinear_init(12, 12, 5)
    assert isinstance(weights, Tensor)
