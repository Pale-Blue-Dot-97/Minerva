from collections import defaultdict
import os

from torch.utils.data import DataLoader

from minerva.samplers import RandomPairBatchGeoSampler, RandomPairGeoSampler
from minerva.datasets import TestImgDataset, PairedDataset, stack_sample_pairs


data_root = os.path.join("tests", "tmp")
img_root = os.path.join(data_root, "data", "test_images")
lc_root = os.path.join(data_root, "data", "test_lc")


def test_randompairgeosampler() -> None:
    dataset = PairedDataset(TestImgDataset, img_root, res=1.0)

    sampler = RandomPairGeoSampler(dataset, size=32, length=32)
    loader = DataLoader(
        dataset, batch_size=8, sampler=sampler, collate_fn=stack_sample_pairs
    )

    batch = next(iter(loader))

    assert isinstance(batch[0], defaultdict)
    assert isinstance(batch[1], defaultdict)
    assert len(batch[0]["image"]) == 8
    assert len(batch[1]["image"]) == 8


"""
def test_randompairbatchgeosampler() -> None:
    dataset = PairedDataset(TestImgDataset, img_root, res=1.0)

    sampler = RandomPairBatchGeoSampler(dataset, size=32, length=32, batch_size=8)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler, collate_fn=stack_sample_pairs)

    batch = next(iter(loader))

    assert isinstance(batch[0], defaultdict)
    assert isinstance(batch[1], defaultdict)
    assert len(batch[0]["image"]) == 8
    assert len(batch[1]["image"]) == 8
"""
