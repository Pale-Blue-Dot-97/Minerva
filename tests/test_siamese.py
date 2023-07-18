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
r"""Tests for :mod:`minerva.models.siamese`.
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

import pytest
import torch
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import NegativeCosineSimilarity, NTXentLoss
except (OSError, NewConnectionError, MaxRetryError):
    NegativeCosineSimilarity = getattr(
        importlib.import_module("lightly.loss"), "NegativeCosineSimilarity"
    )
    NTXentLoss = getattr(importlib.import_module("lightly.loss"), "NTXentLoss")

from minerva.loss import SegBarlowTwinsLoss
from minerva.models import (
    MinervaSiamese,
    SimCLR18,
    SimCLR34,
    SimCLR50,
    SimConv,
    SimSiam18,
    SimSiam34,
    SimSiam50,
)


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_simclr() -> None:
    loss_func = NTXentLoss(0.3)

    input_size = (4, 32, 32)

    x = torch.rand((3, *input_size))

    x = torch.stack([x, x])

    for model in (
        SimCLR18(loss_func, input_size=input_size),
        SimCLR34(loss_func, input_size=input_size),
        SimCLR50(loss_func, input_size=input_size),
    ):
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim(sample_pairs=True)
        assert model.output_shape == (128,)

        loss, z = model.step(x, train=True)

        assert type(loss.item()) is float
        assert z.size() == (6, 128)

    model = SimCLR18(loss_func, input_size=input_size)

    with pytest.raises(NotImplementedError, match="Optimiser has not been set!"):
        _ = model.step(x, train=True)


def test_simsiam() -> None:
    loss_func = NegativeCosineSimilarity()

    input_size = (4, 32, 32)

    x = torch.rand((3, *input_size))

    x = torch.stack([x, x])

    for model in (
        SimSiam18(loss_func, input_size=input_size),
        SimSiam34(loss_func, input_size=input_size),
        SimSiam50(loss_func, input_size=input_size),
    ):
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim(sample_pairs=True)
        assert model.output_shape == (128,)

        loss, z = model.step(x, train=True)

        assert type(loss.item()) is float
        assert z.size() == (6, 128)

    model = SimSiam18(loss_func, input_size=input_size)

    with pytest.raises(NotImplementedError, match="Optimiser has not been set!"):
        _ = model.step(x, train=True)


def test_simconv() -> None:
    loss_func = SegBarlowTwinsLoss()

    input_size = (4, 32, 32)

    x = torch.rand((3, *input_size))

    x = torch.stack([x, x])

    model: MinervaSiamese = SimConv(loss_func, input_size=input_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim(sample_pairs=True)
    assert model.output_shape == (128, input_size[1], input_size[2])

    loss, z = model.step(x, train=True)

    assert type(loss.item()) is float
    assert z.size() == (6, 128, input_size[1], input_size[2])
