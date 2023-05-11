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
r"""Tests for :mod:`minerva.optimsiers`.
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
import pytest
import torch
import torch.nn.modules as nn

from minerva.models.__depreciated import CNN
from minerva.optimisers import LARS


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_lars() -> None:
    model = CNN(nn.CrossEntropyLoss(), input_size=(3, 224, 224))

    with pytest.raises(ValueError, match="Invalid learning rate: -0.1"):
        _ = LARS(model.parameters(), lr=-0.1)

    with pytest.raises(ValueError, match="Invalid momentum value: -0.1"):
        _ = LARS(model.parameters(), lr=1.0, momentum=-0.1)

    with pytest.raises(ValueError, match="Invalid weight_decay value: -0.01"):
        _ = LARS(model.parameters(), lr=0.1, weight_decay=-0.01)

    with pytest.raises(ValueError, match="Invalid LARS coefficient value: -0.02"):
        _ = LARS(model.parameters(), lr=1.0, eta=-0.02)

    model.set_optimiser(LARS(model.parameters(), lr=1.0e-3))

    x = torch.rand(60, 6, 3, 224, 224)
    y = torch.randint(0, 8, size=(60, 6))  # type: ignore[attr-defined]

    for mode in (True, False):
        for i in range(60):
            loss, z = model.step(x[i], y[i], train=mode)
