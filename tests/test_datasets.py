import os
from collections import defaultdict

from numpy.testing import assert_array_equal
import pytest
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers.utils import get_random_bounding_box
from rasterio.crs import CRS

from minerva import datasets as mdt
from minerva.utils.utils import config

data_root = os.path.join("tests", "tmp")
img_root = os.path.join(data_root, "data", "test_images")
lc_root = os.path.join(data_root, "data", "test_lc")

bounds = BoundingBox(590520.0, 600530.0, 5780620.0, 5790630.0, 0, 1e12)


def test_make_bounding_box() -> None:
    assert mdt.make_bounding_box() is None
    assert mdt.make_bounding_box(False) is None

    bbox = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert mdt.make_bounding_box(bbox) == BoundingBox(*bbox)


def test_tinydataset() -> None:
    """Source of TIFF: https://github.com/mommermi/geotiff_sample"""

    imagery = mdt.TestImgDataset(img_root)
    labels = mdt.TestMaskDataset(lc_root)

    dataset = imagery & labels
    assert isinstance(dataset, IntersectionDataset)


def test_paired_datasets() -> None:
    dataset = mdt.PairedDataset(mdt.TestImgDataset, img_root)

    query_1 = get_random_bounding_box(bounds, (32, 32), 10.0)
    query_2 = get_random_bounding_box(bounds, (32, 32), 10.0)

    sample_1, sample_2 = dataset[(query_1, query_2)]

    assert type(sample_1) == dict
    assert type(sample_2) == dict

    assert type(dataset.crs) == CRS
    assert type(dataset.dataset) == mdt.TestImgDataset

    with pytest.raises(AttributeError):
        dataset.roi

    assert type(dataset.__repr__()) == str


def test_get_collator() -> None:
    collator_params_1 = {"module": "torchgeo.datasets.utils", "name": "stack_samples"}
    collator_params_2 = {"name": "stack_sample_pairs"}

    assert callable(mdt.get_collator(collator_params_1))
    assert callable(mdt.get_collator(collator_params_2))


def test_stack_sample_pairs() -> None:
    image_1 = torch.rand(size=(3, 52, 52))
    mask_1 = torch.randint(0, 8, (52, 52))
    bbox_1 = [BoundingBox(0, 1, 0, 1, 0, 1)]

    image_2 = torch.rand(size=(3, 52, 52))
    mask_2 = torch.randint(0, 8, (52, 52))
    bbox_2 = [BoundingBox(0, 1, 0, 1, 0, 1)]

    sample_1 = {
        "image": image_1,
        "mask": mask_1,
        "bbox": bbox_1,
    }

    sample_2 = {
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
            assert assert_array_equal(stacked_samples_1[key][i], sample_1[key]) is None
            assert assert_array_equal(stacked_samples_2[key][i], sample_2[key]) is None


def test_intersect_datasets() -> None:
    pass


def test_make_dataset() -> None:
    data_dir = ["tests", "tmp", "data"]

    dataset_params = {
        "image": {
            "module": "minerva.datasets",
            "name": "TestImgDataset",
            "root": "test_images",
            "params": {"res": 10.0},
        }
    }

    transform_params = {
        "image": {"Normalise": {"module": "minerva.transforms", "norm_value": 255}}
    }

    dataset_1, subdatasets_1 = mdt.make_dataset(data_dir, dataset_params)

    assert type(dataset_1) == type(subdatasets_1[0])
    assert isinstance(dataset_1, mdt.TestImgDataset)

    dataset_2, subdatasets_2 = mdt.make_dataset(
        data_dir, dataset_params, transform_params, sample_pairs=True
    )

    assert type(dataset_2) == type(subdatasets_2[0])
    assert isinstance(dataset_2, mdt.PairedDataset)


def test_construct_dataloader() -> None:
    data_dir = ["tests", "tmp", "data"]

    dataset_params = {
        "image": {
            "module": "minerva.datasets",
            "name": "TestImgDataset",
            "root": "test_images",
            "params": {"res": 10.0},
        }
    }

    sampler_params_1 = {
        "module": "torchgeo.samplers",
        "name": "RandomBatchGeoSampler",
        "roi": False,
        "params": {
            "size": 224,
            "length": 4096,
            "batch_size": 16,
        },
    }

    sampler_params_2 = {
        "module": "minerva.samplers",
        "name": "RandomPairGeoSampler",
        "roi": False,
        "params": {
            "size": 224,
            "length": 4096,
        },
    }

    transform_params = {
        "image": {"Normalise": {"module": "minerva.transforms", "norm_value": 255}}
    }

    dataloader_params = {"batch_size": 256, "num_workers": 10, "pin_memory": True}

    dataloader_1 = mdt.construct_dataloader(
        data_dir, dataset_params, sampler_params_1, dataloader_params
    )
    dataloader_2 = mdt.construct_dataloader(
        data_dir,
        dataset_params,
        sampler_params_2,
        dataloader_params,
        transform_params=transform_params,
        sample_pairs=True,
    )
    dataloader_3 = mdt.construct_dataloader(
        data_dir, dataset_params, sampler_params_1, dataloader_params, world_size=2
    )

    assert isinstance(dataloader_1, DataLoader)
    assert isinstance(dataloader_2, DataLoader)
    assert isinstance(dataloader_3, DataLoader)


def test_get_transform() -> None:
    name = "RandomResizedCrop"
    params = {"module": "torchvision.transforms", "size": 128}
    transform = mdt.get_transform(name, params)

    assert callable(transform)

    with pytest.raises(TypeError):
        _ = mdt.get_transform("DataFrame", {"module": "pandas"})


def test_make_transformations() -> None:
    transform_params_1 = {
        "CenterCrop": {"module": "torchvision.transforms", "size": 128},
        "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7},
    }

    transform_params_2 = {
        "RandomApply": {
            "CenterCrop": {"module": "torchvision.transforms", "size": 128},
            "p": 0.3,
        },
        "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7},
    }

    transforms_1 = mdt.make_transformations(transform_params_1)
    assert callable(transforms_1)

    transforms_2 = mdt.make_transformations(transform_params_2)
    assert callable(transforms_2)

    assert mdt.make_transformations(False) is None


def test_make_loaders() -> None:
    old_params = config.copy()

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
    assert mdt.get_manifest_path() == os.path.join(
        "tests", "tmp", "cache", "Chesapeake7_Manifest.csv"
    )


"""
def test_get_manifest() -> None:
    pass


def test_make_manifest() -> None:
    pass


def test_load_all_samples() -> None:
    pass
"""
