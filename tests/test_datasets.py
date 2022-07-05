import os

import pytest
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
