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
r"""Tests for :mod:`minerva.models._depreciated`.
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
import numpy as np
import torch
from torch import Tensor

from minerva.models.__depreciated import CNN, MLP


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_mlp(x_entropy_loss) -> None:
    model = MLP(x_entropy_loss, hidden_sizes=128)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert isinstance(model.output_shape, tuple)
    assert model.output_shape[0] is model.n_classes

    x = torch.rand(16, (288))

    z = model(x)

    assert isinstance(z, Tensor)
    assert z.size() == (16, 8)


def test_cnn(x_entropy_loss) -> None:
    input_size = (4, 64, 64)
    model = CNN(x_entropy_loss, input_size=input_size)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert isinstance(model.output_shape, tuple)
    assert model.output_shape[0] is model.n_classes

    x = torch.rand(6, *input_size)
    y = torch.LongTensor(np.random.randint(0, 8, size=6))

    loss, z = model.step(x, y, train=True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (6, 8)
