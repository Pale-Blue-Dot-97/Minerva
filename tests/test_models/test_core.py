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
from typing import Tuple

import internet_sabotage
import numpy as np
import pytest
import torch
from pytest_lazyfixture import lazy_fixture
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
from torchvision.models.resnet import resnet18

from minerva.models import (
    MinervaBackbone,
    MinervaSiamese,
    MinervaWrapper,
    SimCLR18,
    bilinear_init,
    extract_wrapped_model,
    get_output_shape,
    get_torch_weights,
    is_minerva_model,
    is_minerva_subtype,
)
from minerva.models.__depreciated import MLP


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_minerva_model(x_entropy_loss, std_n_classes: int, std_n_batches: int) -> None:
    x = torch.rand(std_n_batches, (288))
    y = torch.LongTensor(np.random.randint(0, std_n_classes, size=std_n_batches))

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
    assert isinstance(model.output_shape, tuple)
    assert model.output_shape[0] is model.n_classes

    for train in (True, False):
        loss, z = model.step(x, y, train=train)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (std_n_batches, std_n_classes)


def test_minerva_backbone(rgbi_input_size: Tuple[int, int, int]) -> None:
    loss_func = NTXentLoss(0.3)

    model = SimCLR18(loss_func, input_size=rgbi_input_size)

    assert isinstance(model.get_backbone(), Module)


def test_minerva_wrapper(
    x_entropy_loss,
    small_patch_size: Tuple[int, int],
    std_n_classes: int,
    std_n_batches: int,
) -> None:
    input_size = (3, *small_patch_size)
    model = MinervaWrapper(
        ResNetGenerator,
        x_entropy_loss,
        input_size=input_size,
        n_classes=std_n_classes,
        num_classes=std_n_classes,
        name="resnet-9",
    )

    assert isinstance(repr(model), str)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert isinstance(model.output_shape, tuple)
    assert model.output_shape[0] is model.n_classes

    x = torch.rand(std_n_batches, *input_size)
    y = LongTensor(np.random.randint(0, std_n_classes, size=std_n_batches))

    for train in (True, False):
        loss, z = model.step(x, y, train=train)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (std_n_batches, std_n_classes)


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


def test_get_output_shape(exp_mlp, std_n_classes: int) -> None:
    output_shape = get_output_shape(exp_mlp, 64)

    assert output_shape == (std_n_classes,)


def test_bilinear_init() -> None:
    weights = bilinear_init(12, 12, 5)
    assert isinstance(weights, Tensor)


@pytest.mark.parametrize(
    ("model", "answer"),
    [
        (lazy_fixture("exp_fcn"), True),
        (lazy_fixture("exp_cnn"), True),
        (resnet18(), False),
    ],
)
@pytest.mark.parametrize("compile_model", (True, False))
def test_is_minerva_model(model: Module, compile_model: bool, answer: bool) -> None:
    if compile_model:
        model = torch.compile(model)

    assert is_minerva_model(model) == answer


@pytest.mark.parametrize(
    ("model", "subtype", "answer"),
    [
        (lazy_fixture("exp_fcn"), MinervaBackbone, True),
        (lazy_fixture("exp_cnn"), MinervaSiamese, False),
        (lazy_fixture("exp_simconv"), MinervaSiamese, True),
        (resnet18(), MinervaBackbone, False),
    ],
)
@pytest.mark.parametrize("compile_model", (True, False))
def test_is_minerva_subtype(
    model: Module, subtype: type, compile_model: bool, answer: bool
) -> None:
    if compile_model:
        model = torch.compile(model)

    assert is_minerva_subtype(model, subtype) == answer
