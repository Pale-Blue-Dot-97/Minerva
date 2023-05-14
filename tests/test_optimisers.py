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
