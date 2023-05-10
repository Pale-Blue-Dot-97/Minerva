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

data_root = Path("tests", "tmp")
img_root = str(data_root / "data" / "test_images")
lc_root = str(data_root / "data" / "test_lc")


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_randompairgeosampler() -> None:
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


def test_randompairbatchgeosampler() -> None:
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


def test_get_greater_bbox(simple_bbox) -> None:
    new_bbox = get_greater_bbox(simple_bbox, 1.0, 1.0)
    assert new_bbox == BoundingBox(-1.0, 2.0, -1.0, 2.0, 0.0, 1.0)
