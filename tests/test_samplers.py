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
r"""Tests for :mod:`minerva.samplers`.
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
from pathlib import Path
from typing import Any, Dict

import pytest
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import BoundingBox

from minerva.datasets import PairedDataset, TstImgDataset, stack_sample_pairs
from minerva.samplers import (
    RandomPairBatchGeoSampler,
    RandomPairGeoSampler,
    get_greater_bbox,
)


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_randompairgeosampler(img_root: Path) -> None:
    dataset = PairedDataset(TstImgDataset, img_root, res=1.0)

    sampler = RandomPairGeoSampler(dataset, size=32, length=32, max_r=52)
    loader: DataLoader[Dict[str, Any]] = DataLoader(
        dataset, batch_size=8, sampler=sampler, collate_fn=stack_sample_pairs
    )

    batch = next(iter(loader))

    assert isinstance(batch[0], defaultdict)
    assert isinstance(batch[1], defaultdict)
    assert len(batch[0]["image"]) == 8
    assert len(batch[1]["image"]) == 8


def test_randompairbatchgeosampler(img_root: Path) -> None:
    dataset = PairedDataset(TstImgDataset, img_root, res=1.0)

    sampler = RandomPairBatchGeoSampler(
        dataset, size=32, length=32, batch_size=8, max_r=52, tiles_per_batch=1
    )
    loader: DataLoader[Dict[str, Any]] = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=stack_sample_pairs
    )

    assert isinstance(loader, DataLoader)

    batch = next(iter(loader))

    assert isinstance(batch[0], defaultdict)
    assert isinstance(batch[1], defaultdict)
    assert len(batch[0]["image"]) == 8
    assert len(batch[1]["image"]) == 8

    with pytest.raises(
        ValueError, match="tiles_per_batch=2 is not a multiple of batch_size=7"
    ):
        _ = RandomPairBatchGeoSampler(
            dataset, size=32, length=32, batch_size=7, max_r=52, tiles_per_batch=2
        )


def test_get_greater_bbox(simple_bbox: BoundingBox) -> None:
    new_bbox = get_greater_bbox(simple_bbox, 1.0, 1.0)
    assert new_bbox == BoundingBox(-1.0, 2.0, -1.0, 2.0, 0.0, 1.0)
