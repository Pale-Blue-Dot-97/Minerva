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
r"""Tests for :mod:`minerva.datasets`.
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
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, UnionDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers.utils import get_random_bounding_box

from minerva import datasets as mdt
from minerva.datasets import (
    PairedDataset,
    PairedUnionDataset,
    TstImgDataset,
    TstMaskDataset,
)
from minerva.utils.utils import CONFIG


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_make_bounding_box() -> None:
    assert mdt.make_bounding_box() is None
    assert mdt.make_bounding_box(False) is None

    bbox = (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
    assert mdt.make_bounding_box(bbox) == BoundingBox(*bbox)

    with pytest.raises(
        ValueError,
        match="``roi`` must be a sequence of floats or ``False``, not ``True``",
    ):
        _ = mdt.make_bounding_box(True)


def test_tinydataset(img_root: Path, lc_root: Path) -> None:
    """Source of TIFF: https://github.com/mommermi/geotiff_sample"""

    imagery = TstImgDataset(str(img_root))
    labels = TstMaskDataset(str(lc_root))

    dataset = imagery & labels
    assert isinstance(dataset, IntersectionDataset)


def test_paired_datasets(img_root: Path) -> None:
    dataset1 = PairedDataset(TstImgDataset, img_root)
    dataset2 = TstImgDataset(str(img_root))

    with pytest.raises(
        ValueError,
        match=f"Intersecting a dataset of {type(dataset2)} and a PairedDataset is not supported!",
    ):
        _ = dataset1 & dataset2  # type: ignore[operator]

    dataset3 = PairedDataset(dataset2)

    non_dataset = 42
    with pytest.raises(
        ValueError,
        match=f"``dataset`` is of unsupported type {type(non_dataset)} not GeoDataset",
    ):
        _ = PairedDataset(42)  # type: ignore[call-overload]

    bounds = BoundingBox(411248.0, 412484.0, 4058102.0, 4059399.0, 0, 1e12)
    query_1 = get_random_bounding_box(bounds, (32, 32), 10.0)
    query_2 = get_random_bounding_box(bounds, (32, 32), 10.0)

    for dataset in (dataset1, dataset3):
        sample_1, sample_2 = dataset[(query_1, query_2)]

        assert type(sample_1) == dict
        assert type(sample_2) == dict

        assert type(dataset.crs) == CRS
        assert type(getattr(dataset, "crs")) == CRS
        assert type(dataset.dataset) == TstImgDataset
        assert type(dataset.__getattr__("dataset")) == TstImgDataset

        with pytest.raises(AttributeError):
            dataset.roi

        assert type(dataset.__repr__()) == str

        assert isinstance(
            dataset.plot_random_sample((32, 32), 1.0, suptitle="test"), plt.Figure  # type: ignore[attr-defined]
        )


def test_paired_union_datasets(img_root: Path) -> None:
    def dataset_test(_dataset) -> None:
        query_1 = get_random_bounding_box(bounds, (32, 32), 10.0)
        query_2 = get_random_bounding_box(bounds, (32, 32), 10.0)
        sample_1, sample_2 = _dataset[(query_1, query_2)]

        assert type(sample_1) == dict
        assert type(sample_2) == dict

    bounds = BoundingBox(411248.0, 412484.0, 4058102.0, 4059399.0, 0, 1e12)

    dataset1 = TstImgDataset(str(img_root))
    dataset2 = TstImgDataset(str(img_root))
    dataset3 = PairedDataset(TstImgDataset, img_root)
    dataset4 = PairedDataset(TstImgDataset, img_root)

    union_dataset1 = PairedDataset(dataset1 | dataset2)
    union_dataset2 = dataset3 | dataset4
    union_dataset3 = union_dataset1 | dataset3
    union_dataset4 = union_dataset1 | dataset2
    union_dataset5 = dataset3 | dataset2

    for dataset in (
        union_dataset1,
        union_dataset2,
        union_dataset3,
        union_dataset4,
        union_dataset5,
    ):
        assert isinstance(dataset, PairedUnionDataset)
        dataset_test(dataset)


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

    assert type(stacked_samples_1) == defaultdict
    assert type(stacked_samples_2) == defaultdict

    for key in ("image", "mask", "bbox"):
        for i in range(6):
            assert_array_equal(stacked_samples_1[key][i], sample_1[key])
            assert_array_equal(stacked_samples_2[key][i], sample_2[key])


def test_intersect_datasets(img_root: Path, lc_root: Path) -> None:
    imagery = PairedDataset(TstImgDataset, img_root)
    labels = PairedDataset(TstMaskDataset, lc_root)

    assert isinstance(mdt.intersect_datasets([imagery, labels]), IntersectionDataset)


def test_make_dataset(exp_dataset_params: Dict[str, Any]) -> None:
    data_dir = ["tests", "tmp", "data"]

    dataset_1, subdatasets_1 = mdt.make_dataset(data_dir, exp_dataset_params)

    assert isinstance(dataset_1, type(subdatasets_1[0]))
    assert isinstance(dataset_1, TstImgDataset)

    dataset_2, subdatasets_2 = mdt.make_dataset(
        data_dir,
        exp_dataset_params,
        sample_pairs=True,
    )

    assert isinstance(dataset_2, type(subdatasets_2[0]))
    assert isinstance(dataset_2, PairedDataset)

    exp_dataset_params["mask"] = {
        "module": "minerva.datasets",
        "name": "TstMaskDataset",
        "root": "test_lc",
        "params": {"res": 10.0},
    }

    dataset_params2 = {
        "image": {
            "image_1": exp_dataset_params["image"],
            "image_2": exp_dataset_params["image"],
        },
        "mask": exp_dataset_params["mask"],
    }

    dataset_3, subdatasets_3 = mdt.make_dataset(data_dir, dataset_params2)
    assert isinstance(dataset_3, IntersectionDataset)
    assert isinstance(subdatasets_3[0], UnionDataset)

    dataset_4, subdatasets_4 = mdt.make_dataset(
        data_dir,
        dataset_params2,
        sample_pairs=True,
    )
    assert isinstance(dataset_4, IntersectionDataset)
    assert isinstance(subdatasets_4[0], UnionDataset)


@pytest.mark.parametrize(
    ["sampler_params", "kwargs"],
    [
        (
            {
                "module": "torchgeo.samplers",
                "name": "RandomBatchGeoSampler",
                "roi": False,
                "params": {
                    "size": 224,
                    "length": 4096,
                },
            },
            {},
        ),
        (
            {
                "module": "minerva.samplers",
                "name": "RandomPairGeoSampler",
                "roi": False,
                "params": {
                    "size": 224,
                    "length": 4096,
                },
            },
            {"sample_pairs": True},
        ),
        (
            {
                "module": "torchgeo.samplers",
                "name": "RandomBatchGeoSampler",
                "roi": False,
                "params": {
                    "size": 224,
                    "length": 4096,
                },
            },
            {"world_size": 2},
        ),
    ],
)
def test_construct_dataloader(
    exp_dataset_params: Dict[str, Any],
    sampler_params: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> None:
    data_dir = ["tests", "tmp", "data"]

    batch_size = 256

    dataloader_params = {"num_workers": 2, "pin_memory": True}

    dataloader = mdt.construct_dataloader(
        data_dir,
        exp_dataset_params,
        sampler_params,
        dataloader_params,
        batch_size,
        sample_pairs=kwargs.get("sample_pairs", False),
        world_size=kwargs.get("world_size", 1),
    )

    assert isinstance(dataloader, DataLoader)


def test_get_transform() -> None:
    name = "RandomResizedCrop"
    params = {"module": "torchvision.transforms", "size": 128}
    transform = mdt.get_transform(name, params)

    assert callable(transform)

    with pytest.raises(TypeError):
        _ = mdt.get_transform("DataFrame", {"module": "pandas"})


@pytest.mark.parametrize(
    ["params", "key"],
    [
        (
            {
                "CenterCrop": {"module": "torchvision.transforms", "size": 128},
                "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7},
            },
            None,
        ),
        (
            {
                "RandomApply": {
                    "CenterCrop": {"module": "torchvision.transforms", "size": 128},
                    "p": 0.3,
                },
                "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7},
            },
            None,
        ),
        (
            {
                "MinervaCompose": {
                    "CenterCrop": {"module": "torchvision.transforms", "size": 128},
                    "RandomHorizontalFlip": {
                        "module": "torchvision.transforms",
                        "p": 0.7,
                    },
                },
                "RandomApply": {
                    "CenterCrop": {"module": "torchvision.transforms", "size": 128},
                    "p": 0.3,
                },
                "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7},
            },
            "image",
        ),
    ],
)
def test_make_transformations(params: Dict[str, Any], key: str) -> None:
    if params:
        transforms = mdt.make_transformations(params, key)
        assert callable(transforms)
    else:
        assert mdt.make_transformations(False) is None


def test_make_loaders() -> None:
    old_params = CONFIG.copy()

    mask_transforms = {"RandomHorizontalFlip": {"module": "torchvision.transforms"}}
    transform_params = {
        "train": {
            "image": False,
            "mask": mask_transforms,
        },
        "val": {
            "image": False,
            "mask": mask_transforms,
        },
        "test": {
            "image": False,
            "mask": False,
        },
    }

    old_params["transform_params"] = transform_params

    loaders, n_batches, class_dist, params = mdt.make_loaders(**old_params)

    for mode in ("train", "val", "test"):
        assert isinstance(loaders[mode], DataLoader)
        assert type(n_batches[mode]) is int
        assert type(class_dist) is list
        assert type(params) == dict


def test_get_manifest_path() -> None:
    assert mdt.get_manifest_path() == str(
        Path("tests", "tmp", "cache", "Chesapeake7_Manifest.csv")
    )


def test_get_manifest() -> None:
    manifest_path = Path("tests", "tmp", "cache", "Chesapeake7_Manifest.csv")

    if manifest_path.exists():
        manifest_path.unlink()

    assert isinstance(mdt.get_manifest(manifest_path), pd.DataFrame)
    assert isinstance(mdt.get_manifest(manifest_path), pd.DataFrame)

    new_path = Path("tests", "tmp", "empty", "Chesapeake7_Manifest.csv")
    if new_path.exists():
        print("exists")
        new_path.unlink()

    if new_path.parent.exists():
        new_path.parent.rmdir()

    assert isinstance(mdt.get_manifest(new_path), pd.DataFrame)

    if new_path.exists():
        new_path.unlink()

    if new_path.parent.exists():
        new_path.parent.rmdir()

    if manifest_path.exists():
        manifest_path.unlink()
