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
r"""Tests for :mod:`minerva.datasets.collators`.
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
from collections import defaultdict
from typing import Any, Dict, List, Union

import torch
from numpy.testing import assert_array_equal
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox

from minerva import datasets as mdt


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_get_collator() -> None:
    collator_params_1 = {"module": "torchgeo.datasets.utils", "name": "stack_samples"}
    collator_params_2 = {"name": "stack_sample_pairs"}

    assert callable(mdt.get_collator(collator_params_1))
    assert callable(mdt.get_collator(collator_params_2))


def test_stack_sample_pairs() -> None:
    image_1 = torch.rand(size=(3, 52, 52))
    mask_1 = torch.randint(0, 8, (52, 52))  # type: ignore[attr-defined]
    bbox_1 = [BoundingBox(0, 1, 0, 1, 0, 1)]

    image_2 = torch.rand(size=(3, 52, 52))
    mask_2 = torch.randint(0, 8, (52, 52))  # type: ignore[attr-defined]
    bbox_2 = [BoundingBox(0, 1, 0, 1, 0, 1)]

    sample_1: Dict[str, Union[Tensor, List[Any]]] = {
        "image": image_1,
        "mask": mask_1,
        "bbox": bbox_1,
    }

    sample_2: Dict[str, Union[Tensor, List[Any]]] = {
        "image": image_2,
        "mask": mask_2,
        "bbox": bbox_2,
    }

    samples = []

    for _ in range(6):
        samples.append((sample_1, sample_2))

    stacked_samples_1, stacked_samples_2 = mdt.stack_sample_pairs(samples)

    assert isinstance(stacked_samples_1, defaultdict)
    assert isinstance(stacked_samples_2, defaultdict)

    for key in ("image", "mask", "bbox"):
        for i in range(6):
            assert_array_equal(stacked_samples_1[key][i], sample_1[key])
            assert_array_equal(stacked_samples_2[key][i], sample_2[key])
