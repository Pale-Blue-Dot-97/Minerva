import pytest
from collections import defaultdict
from typing import Any, Dict
from pathlib import Path

from torch.utils.data import DataLoader

from minerva.samplers import RandomPairBatchGeoSampler, RandomPairGeoSampler
from minerva.datasets import TstImgDataset, PairedDataset, stack_sample_pairs
from minerva.utils.utils import set_seeds


data_root = Path("tests", "tmp")
img_root = str(data_root / "data" / "test_images")
lc_root = str(data_root / "data" / "test_lc")

set_seeds(42)


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
