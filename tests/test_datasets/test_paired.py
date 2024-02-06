# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
r"""Tests for :mod:`minerva.datasets.paired`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

from pathlib import Path

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from typing import Tuple

import matplotlib.pyplot as plt
import pytest
from rasterio.crs import CRS
from torch.utils.data import RandomSampler
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers.utils import get_random_bounding_box

from minerva.datasets import (
    NonGeoSSL4EOS12Sentinel2,
    PairedConcatDataset,
    PairedGeoDataset,
    PairedNonGeoDataset,
    PairedUnionDataset,
)
from minerva.datasets.__testing import TstImgDataset
from minerva.transforms import MinervaCompose, Normalise


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_paired_geodatasets(img_root: Path) -> None:
    dataset1 = PairedGeoDataset(TstImgDataset, str(img_root))
    dataset2 = TstImgDataset(str(img_root))

    with pytest.raises(
        ValueError,
        match=f"Intersecting a dataset of {type(dataset2)} and a PairedGeoDataset is not supported!",
    ):
        _ = dataset1 & dataset2  # type: ignore[operator]

    dataset3 = PairedGeoDataset(dataset2)

    non_dataset = 42
    with pytest.raises(
        ValueError,
        match=f"``dataset`` is of unsupported type {type(non_dataset)} not GeoDataset",
    ):
        _ = PairedGeoDataset(42)  # type: ignore[call-overload]

    bounds = BoundingBox(411248.0, 412484.0, 4058102.0, 4059399.0, 0, 1e12)
    query_1 = get_random_bounding_box(bounds, (32, 32), 10.0)
    query_2 = get_random_bounding_box(bounds, (32, 32), 10.0)

    for dataset in (dataset1, dataset3):
        sample_1, sample_2 = dataset[(query_1, query_2)]

        assert isinstance(sample_1, dict)
        assert isinstance(sample_2, dict)

        assert isinstance(dataset.crs, CRS)
        assert isinstance(getattr(dataset, "crs"), CRS)
        assert isinstance(dataset.dataset, TstImgDataset)
        assert isinstance(dataset.__getattr__("dataset"), TstImgDataset)

        with pytest.raises(AttributeError):
            dataset.roi

        assert isinstance(dataset.__repr__(), str)

        assert isinstance(
            dataset.plot_random_sample((32, 32), 1.0, suptitle="test"), plt.Figure  # type: ignore[attr-defined]
        )


def test_paired_union_datasets(img_root: Path) -> None:
    def dataset_test(_dataset) -> None:
        query_1 = get_random_bounding_box(bounds, (32, 32), 10.0)
        query_2 = get_random_bounding_box(bounds, (32, 32), 10.0)
        sample_1, sample_2 = _dataset[(query_1, query_2)]

        assert isinstance(sample_1, dict)
        assert isinstance(sample_2, dict)

    bounds = BoundingBox(411248.0, 412484.0, 4058102.0, 4059399.0, 0, 1e12)

    dataset1 = TstImgDataset(str(img_root))
    dataset2 = TstImgDataset(str(img_root))
    dataset3 = PairedGeoDataset(TstImgDataset, str(img_root))
    dataset4 = PairedGeoDataset(TstImgDataset, str(img_root))

    union_dataset1 = PairedGeoDataset(dataset1 | dataset2)
    union_dataset2 = dataset3 | dataset4
    union_dataset3 = union_dataset1 | dataset3
    union_dataset4 = union_dataset1 | dataset2  # type: ignore[operator]
    union_dataset5 = dataset3 | dataset2  # type: ignore[operator]

    for dataset in (
        union_dataset1,
        union_dataset2,
        union_dataset3,
        union_dataset4,
        union_dataset5,
    ):
        assert isinstance(dataset, PairedUnionDataset)
        dataset_test(dataset)


def test_paired_nongeodatasets(data_root: Path) -> None:
    path = str(data_root / "SSL4EO-S12")
    dataset = NonGeoSSL4EOS12Sentinel2(
        path,
        transforms=MinervaCompose(Normalise(4095), key="image"),
        bands=["B2", "B3", "B4", "B8"],
    )
    paired_dataset = PairedNonGeoDataset(dataset, size=32, max_r=32)

    assert isinstance(paired_dataset, PairedNonGeoDataset)

    sample_1, sample_2 = paired_dataset[0]

    assert isinstance(sample_1, dict)
    assert isinstance(sample_2, dict)

    assert isinstance(paired_dataset.dataset, NonGeoSSL4EOS12Sentinel2)
    assert isinstance(paired_dataset.__getattr__("dataset"), NonGeoSSL4EOS12Sentinel2)

    assert isinstance(paired_dataset.__repr__(), str)

    assert isinstance(paired_dataset.plot_random_sample(suptitle="test"), plt.Figure)


def test_paired_concat_datasets(
    data_root: Path, small_patch_size: Tuple[int, int]
) -> None:
    def dataset_test(_dataset) -> None:
        for sub_dataset in _dataset.datasets:
            assert isinstance(sub_dataset, (PairedNonGeoDataset, PairedConcatDataset))
        sampler = RandomSampler(_dataset)
        sample_1, sample_2 = _dataset[next(iter(sampler))]

        assert isinstance(sample_1, dict)
        assert isinstance(sample_2, dict)

    root = str(data_root / "SSL4EO-S12")
    dataset1 = NonGeoSSL4EOS12Sentinel2(root)
    dataset2 = NonGeoSSL4EOS12Sentinel2(root)
    dataset3 = PairedNonGeoDataset(NonGeoSSL4EOS12Sentinel2, small_patch_size, 32, root)
    dataset4 = PairedNonGeoDataset(NonGeoSSL4EOS12Sentinel2, small_patch_size, 64, root)

    concat_dataset1 = PairedNonGeoDataset(dataset1 | dataset2, small_patch_size, 16)
    concat_dataset2 = dataset3 | dataset4
    concat_dataset3 = concat_dataset1 | dataset3

    with pytest.raises(ValueError):
        _ = concat_dataset1 | dataset2

    with pytest.raises(ValueError):
        _ = dataset3 | dataset2

    for dataset in (
        concat_dataset1,
        concat_dataset2,
        concat_dataset3,
    ):
        assert isinstance(dataset, PairedConcatDataset)
        dataset_test(dataset)
