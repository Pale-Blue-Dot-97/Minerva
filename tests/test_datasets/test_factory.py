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
r"""Tests for :mod:`minerva.datasets.factory`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, UnionDataset

from minerva import datasets as mdt
from minerva.datasets import PairedGeoDataset
from minerva.datasets.__testing import TstImgDataset
from minerva.utils.utils import make_hash


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_make_dataset(exp_dataset_params: dict[str, Any], data_root: Path) -> None:
    dataset_params2 = {
        "image": {
            "subdatasets": {
                "image_1": exp_dataset_params["image"],
                "image_2": exp_dataset_params["image"],
            },
        },
        "mask": exp_dataset_params["mask"],
    }

    dataset_3, subdatasets_3 = mdt.make_dataset(data_root, dataset_params2, cache=False)
    assert isinstance(dataset_3, IntersectionDataset)
    assert isinstance(subdatasets_3[0], UnionDataset)

    dataset_4, subdatasets_4 = mdt.make_dataset(
        data_root,
        dataset_params2,
        sample_pairs=True,
        cache=False,
    )
    assert isinstance(dataset_4, IntersectionDataset)
    assert isinstance(subdatasets_4[0], UnionDataset)

    dataset_params3 = dataset_params2
    dataset_params3["image"]["subdatasets"]["image_1"]["transforms"] = {
        "AutoNorm": {"length": 12}
    }
    dataset_params3["image"]["subdatasets"]["image_2"]["transforms"] = {
        "AutoNorm": {"length": 12}
    }
    dataset_params3["image"]["transforms"] = {"AutoNorm": {"length": 12}}

    dataset_5, subdatasets_5 = mdt.make_dataset(data_root, dataset_params3, cache=False)

    assert isinstance(dataset_5, IntersectionDataset)
    assert isinstance(subdatasets_5[0], UnionDataset)

    del exp_dataset_params["mask"]

    dataset_1, subdatasets_1 = mdt.make_dataset(
        data_root, exp_dataset_params, cache=False
    )

    assert isinstance(dataset_1, type(subdatasets_1[0]))
    assert isinstance(dataset_1, TstImgDataset)

    dataset_2, subdatasets_2 = mdt.make_dataset(
        data_root,
        exp_dataset_params,
        sample_pairs=True,
        cache=False,
    )

    assert isinstance(dataset_2, type(subdatasets_2[0]))
    assert isinstance(dataset_2, PairedGeoDataset)


@pytest.mark.parametrize("sample_pairs", (False, True))
def test_caching_datasets(
    exp_dataset_params: dict[str, Any],
    data_root: Path,
    cache_dir: Path,
    sample_pairs: bool,
) -> None:
    # Just want to test with an imagery dataset.
    del exp_dataset_params["mask"]

    # Make the path to the cached dataset.
    cached_dataset_path = cache_dir / (make_hash(exp_dataset_params["image"]) + ".obj")

    # Ensure that any previous caches are deleted.
    cached_dataset_path.unlink(missing_ok=True)

    # This first call will make the dataset from scratch then cache it.
    dataset_1, subdatasets_1 = mdt.make_dataset(
        data_root,
        exp_dataset_params,
        sample_pairs=sample_pairs,
        cache=True,
        cache_dir=cache_dir,
    )

    # The cached dataset should now exist.
    assert cached_dataset_path.exists()

    # Second call to make dataset with the same args should now load that cached dataset.
    dataset_2, subdatasets_2 = mdt.make_dataset(
        data_root,
        exp_dataset_params,
        sample_pairs=sample_pairs,
        cache=True,
        cache_dir=cache_dir,
    )

    # Datasets from calls 1 should be the same as those from 2.
    assert isinstance(dataset_1, type(dataset_2))
    for i, _ in enumerate(subdatasets_1):
        assert isinstance(subdatasets_1[i], type(subdatasets_2[i]))

    # Clean-up.
    cached_dataset_path.unlink()


@pytest.mark.parametrize(
    ["sampler_params", "kwargs"],
    [
        (
            {
                "_target_": "torchgeo.samplers.RandomBatchGeoSampler",
                "roi": False,
                "size": 224,
                "length": 4096,
            },
            {},
        ),
        (
            {
                "_target_": "minerva.samplers.RandomPairGeoSampler",
                "roi": False,
                "size": 224,
                "length": 4096,
            },
            {"sample_pairs": True},
        ),
        (
            {
                "_target_": "torchgeo.samplers.RandomBatchGeoSampler",
                "roi": False,
                "size": 224,
                "length": 4096,
            },
            {"world_size": 2},
        ),
    ],
)
def test_construct_dataloader(
    exp_dataset_params: dict[str, Any],
    data_root: Path,
    sampler_params: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    batch_size = 256

    dataloader_params = {"num_workers": 2, "pin_memory": True}

    dataloader = mdt.construct_dataloader(
        data_root,
        exp_dataset_params,
        sampler_params,
        dataloader_params,
        batch_size,
        sample_pairs=kwargs.get("sample_pairs", False),
        world_size=kwargs.get("world_size", 1),
        cache=False,
    )

    assert isinstance(dataloader, DataLoader)


def test_make_loaders(default_config: DictConfig) -> None:
    old_params = OmegaConf.to_object(deepcopy(default_config))

    loader, n_batches, class_dist, params = mdt.make_loaders(  # type: ignore[arg-type]
        **old_params,
        task_name="fit-val",
    )

    assert isinstance(loader, DataLoader)
    assert isinstance(n_batches, int)
    assert isinstance(class_dist, list)
    assert isinstance(params, dict)

    old_params_2 = OmegaConf.to_object(deepcopy(default_config))
    assert isinstance(old_params_2, dict)
    dataset_params = old_params_2["tasks"]["fit-val"]["dataset_params"].copy()
    old_params_2["tasks"]["fit-val"]["dataset_params"] = {
        "val-1": dataset_params,
        "val-2": dataset_params,
    }

    loaders, n_batches, class_dist, params = mdt.make_loaders(  # type: ignore[arg-type]
        **old_params_2,
        task_name="fit-val",
    )

    assert isinstance(loaders, dict)
    assert isinstance(loaders["val-1"], DataLoader)
    assert isinstance(n_batches, dict)
    assert isinstance(n_batches["val-2"], int)
    assert isinstance(class_dist, list)
    assert isinstance(params, dict)


def test_get_manifest(
    data_root: Path,
    exp_dataset_params: dict[str, Any],
    exp_sampler_params: dict[str, Any],
) -> None:
    manifest_path = Path("tests", "tmp", "cache", "Chesapeake7_Manifest.csv")

    if manifest_path.exists():
        manifest_path.unlink()

    # Make a manifest.
    assert isinstance(
        mdt.get_manifest(
            manifest_path,
            data_dir=data_root,
            dataset_params=exp_dataset_params,
            sampler_params=exp_sampler_params,
            loader_params={"num_workers": 0},
        ),
        pd.DataFrame,
    )

    # Now try fetching it from cache.
    assert isinstance(mdt.get_manifest(manifest_path), pd.DataFrame)

    new_path = Path("tests", "tmp", "empty", "Chesapeake7_Manifest.csv")
    if new_path.exists():
        new_path.unlink()

    if new_path.parent.exists():
        new_path.parent.rmdir()

    assert isinstance(
        mdt.get_manifest(
            manifest_path,
            data_dir=data_root,
            dataset_params=exp_dataset_params,
            sampler_params=exp_sampler_params,
            loader_params={"num_workers": 0},
        ),
        pd.DataFrame,
    )

    if new_path.exists():
        new_path.unlink()

    if new_path.parent.exists():
        new_path.parent.rmdir()

    if manifest_path.exists():
        manifest_path.unlink()
