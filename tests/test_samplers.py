import os
import tempfile

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchgeo.datasets import ChesapeakeMD

from minerva.samplers import RandomPairBatchGeoSampler, RandomPairGeoSampler
from minerva.utils.utils import pair_collate


def test_randompairgeosampler() -> None:
    pass


"""
    data_root = os.path.join("..", "cache") #tempfile.gettempdir()
    dc_root = os.path.abspath(os.path.join(data_root, "MD")) + "/"

    dataset = ChesapeakeMD(dc_root, download=True, checksum=True)
    sampler = RandomPairGeoSampler(dataset, size=224, length=64)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, collate_fn=pair_collate(default_collate))

    batch = next(loader)

    assert len(batch[0]) == 16
    assert len(batch[1]) == 16
"""
