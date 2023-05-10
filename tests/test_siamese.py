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

from minerva.models import SimCLR18, SimCLR34, SimCLR50, SimSiam18, SimSiam34, SimSiam50


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
