import os
from collections import defaultdict

from numpy.testing import assert_array_equal
import pytest
import torch
from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers.utils import get_random_bounding_box
from rasterio.crs import CRS

from minerva import datasets as mdt

data_root = os.path.join("tests", "tmp")
img_root = os.path.join(data_root, "data", "test_images")
lc_root = os.path.join(data_root, "data", "test_lc")

bounds = BoundingBox(590520.0, 600530.0, 5780620.0, 5790630.0, 0, 1e12)


class TestImgDataset(RasterDataset):
    filename_glob = "*_img.tif"


class TestMaskDataset(RasterDataset):
    filename_glob = "*_lc.tif"


def test_make_bounding_box() -> None:
    assert mdt.make_bounding_box() is None
    assert mdt.make_bounding_box(False) is None

    bbox = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert mdt.make_bounding_box(bbox) == BoundingBox(*bbox)


def test_tinydataset() -> None:
    """Source of TIFF: https://github.com/mommermi/geotiff_sample"""

    imagery = TestImgDataset(img_root)
    labels = TestMaskDataset(lc_root)

    dataset = imagery & labels
    assert isinstance(dataset, IntersectionDataset)


def test_paired_datasets() -> None:
    dataset = mdt.PairedDataset(TestImgDataset, img_root)

    query_1 = get_random_bounding_box(bounds, (32, 32), 10.0)
    query_2 = get_random_bounding_box(bounds, (32, 32), 10.0)

    sample_1, sample_2 = dataset[(query_1, query_2)]

    assert type(sample_1) == dict
    assert type(sample_2) == dict

    assert type(dataset.crs) == CRS
    assert type(dataset.dataset) == TestImgDataset

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
            "module": "tests.test_datasets",
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
    assert isinstance(dataset_1, TestImgDataset)

    dataset_2, subdatasets_2 = mdt.make_dataset(
        data_dir, dataset_params, transform_params, sample_pairs=True
    )

    assert type(dataset_2) == type(subdatasets_2[0])
    assert isinstance(dataset_2, mdt.PairedDataset)
