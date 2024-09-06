# -*- coding: utf-8 -*-
#    Copyright 2024 Harry Baker

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
r"""Collection of classes and functions for use with the SSL4EO-S12 dataset.

Mostly adapted from:
https://github.com/zhu-xlab/SSL4EO-S12/tree/main/src/benchmark/pretrain_ssl/datasets/SSL4EO

"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = ["Harry Baker", "Yi Wang", "Adam J. Stewart"]
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "Apache 2.0 License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["GeoSSL4EOS12Sentinel2", "NonGeoSSL4EOS12Sentinel2", "MinervaSSL4EO"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import cv2
import lmdb
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchgeo.datasets import Sentinel2
from torchvision.datasets import VisionDataset
from torchvision.transforms import Normalize
from tqdm import tqdm

from minerva.transforms import SeasonTransform

from .multispectral import MultiSpectralDataset
from .utils import MinervaNonGeoDataset


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class GeoSSL4EOS12Sentinel2(Sentinel2):
    """Adapted version of :class:~`torchgeo.datasets.Sentinel2` that works with the SSL4EO-S12 data format.

    Attributes:
        filename_glob (str): Adapted pattern from :class:`~torchgeo.datasets.Sentinel2` that looks just for band ID.
        filename_regex (str): Adapted regex from :class:`~torchgeo.datasets.Sentinel2` that looks just for band IDs
            with either ``B0x`` (like standard Sentinel2) or ``Bx`` (like SSL4EO-S12) format.
        all_bands (list[str]): Sentinel2 bands with the leading 0 ommitted.
        rgb_bands (list[str]): RGB Sentinel2 bands with the leading 0 omitted.
    """

    filename_glob = "{}.*"
    filename_regex = r"""(?P<band>B[^[0-1]?[0-9]|B[^[1]?[0-9][\dA])\..*$"""
    date_format = ""
    all_bands: tuple[str, ...] = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    )
    rgb_bands: tuple[str, str, str] = ("B4", "B3", "B2")


class NonGeoSSL4EOS12Sentinel2(MultiSpectralDataset):
    all_bands: tuple[str, ...] = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    )
    rgb_bands: tuple[str, str, str] = ("B4", "B3", "B2")


class MinervaSSL4EO(VisionDataset, MinervaNonGeoDataset):
    """Adapation of the :class:`SSL4EO` dataset for RGBI imagery and improved integration into :mod:`torchgeo`.

    Source: https://github.com/zhu-xlab/SSL4EO-S12/tree/main/src/benchmark/pretrain_ssl/datasets/SSL4EO
    """

    ALL_BANDS_S2_L2A: tuple[str, ...] = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    )
    ALL_BANDS_S2_L1C: tuple[str, ...] = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    )
    RGB_BANDS: tuple[str, str, str] = ("B4", "B3", "B2")
    ALL_BANDS_S1_GRD: tuple[str, str] = ("VV", "VH")

    # Band statistics: mean & std
    # Calculated from 50k data
    S1_MEAN = {"VV": -12.54847273, "VH": -20.19237134}
    S1_STD = {"VV": 5.25697717, "VH": 5.91150917}

    S2A_MEAN = {
        "B1": 752.40087073,
        "B2": 884.29673756,
        "B3": 1144.16202635,
        "B4": 1297.47289228,
        "B5": 1624.90992062,
        "B6": 2194.6423161,
        "B7": 2422.21248945,
        "B8": 2517.76053101,
        "B8A": 2581.64687018,
        "B9": 2645.51888987,
        "B11": 2368.51236873,
        "B12": 1805.06846033,
    }

    S2A_STD = {
        "B1": 1108.02887453,
        "B2": 1155.15170768,
        "B3": 1183.6292542,
        "B4": 1368.11351514,
        "B5": 1370.265037,
        "B6": 1355.55390699,
        "B7": 1416.51487101,
        "B8": 1474.78900051,
        "B8A": 1439.3086061,
        "B9": 1582.28010962,
        "B11": 1455.52084939,
        "B12": 1343.48379601,
    }

    S2C_MEAN = {
        "B1": 1605.57504906,
        "B2": 1390.78157673,
        "B3": 1314.8729939,
        "B4": 1363.52445545,
        "B5": 1549.44374991,
        "B6": 2091.74883118,
        "B7": 2371.7172463,
        "B8": 2299.90463006,
        "B8A": 2560.29504086,
        "B9": 830.06605044,
        "B10": 22.10351321,
        "B11": 2177.07172323,
        "B12": 1524.06546312,
    }

    S2C_STD = {
        "B1": 786.78685367,
        "B2": 850.34818441,
        "B3": 875.06484736,
        "B4": 1138.84957046,
        "B5": 1122.17775652,
        "B6": 1161.59187054,
        "B7": 1274.39184232,
        "B8": 1248.42891965,
        "B8A": 1345.52684884,
        "B9": 577.31607053,
        "B10": 51.15431158,
        "B11": 1336.09932639,
        "B12": 1136.53823676,
    }

    def __init__(
        self,
        root: str,
        lmdb_file: Optional[str] = None,
        normalize: bool = False,
        mode: str = "s2a",
        bands: Optional[tuple[str, ...]] = None,
        dtype: str = "uint8",
        is_slurm_job=False,
        transforms=None,
        season_transform=None,
    ) -> None:

        super().__init__(root, transform=transforms, target_transform=None)

        self.normalize = normalize
        self.mode = mode
        self.bands = bands
        self.dtype = dtype
        self.lmdb_file = lmdb_file
        self.is_slurm_job = is_slurm_job

        if self.lmdb_file:
            if not self.is_slurm_job:
                self._init_db()
            else:
                # Workaround to have length from the start since we don't have LMDB at initialization time
                self.env = None

            self.length = 250000

        else:
            self.ids = os.listdir(os.path.join(self.root, self.mode))
            self.length = len(self.ids)

        self.season_transform: Optional[SeasonTransform]
        if season_transform is not None:
            self.season_transform = SeasonTransform(season_transform)
        else:
            self.season_transform = None

    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_file,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        assert self.env is not None
        with self.env.begin(write=False) as txn:  # type: ignore[unreachable]
            self.length = txn.stat()["entries"]

    def __getitem__(self, index: int) -> Dict[str, Union[Tuple[Tensor, ...], Tensor]]:
        if self.lmdb_file:
            if self.is_slurm_job:
                # Delay loading LMDB data until after initialization
                if self.env is None:
                    self._init_db()

            assert self.env is not None
            with self.env.begin(write=False) as txn:  # type: ignore[unreachable]
                data = txn.get(str(index).encode())

            # S1
            if self.mode == "s1":
                s1_bytes, s1_shape = pickle.loads(data)
                if self.dtype == "uint8":
                    image = np.frombuffer(s1_bytes, dtype=np.uint8).reshape(s1_shape)
                else:
                    image = np.frombuffer(s1_bytes, dtype=np.float32).reshape(s1_shape)  # type: ignore[assignment]

            # S2A
            elif self.mode == "s2a":
                s2a_bytes, s2a_shape = pickle.loads(data)

                if self.dtype == "uint8":
                    image = np.frombuffer(s2a_bytes, dtype=np.uint8).reshape(s2a_shape)
                else:
                    image = np.frombuffer(s2a_bytes, dtype=np.int16).reshape(s2a_shape)  # type: ignore[assignment]
                    image = (image / 10000.0).astype(np.float32)

            # S2C
            elif self.mode == "s2c":
                s2c_bytes, s2c_shape = pickle.loads(data)
                if self.dtype == "uint8":
                    image = np.frombuffer(s2c_bytes, dtype=np.uint8).reshape(s2c_shape)
                else:
                    image = np.frombuffer(s2c_bytes, dtype=np.int16).reshape(s2c_shape)  # type: ignore[assignment]
                    image = (image / 10000.0).astype(np.float32)

            else:
                raise ValueError(
                    f"Invalid value for mode {self.mode}! Must be `s1`, `s2a` or `s2c`"
                )

            # Convert to tensor from ndarray.
            image = torch.from_numpy(image)  # type: ignore[assignment]

            if self.season_transform is not None:
                image = self.season_transform(image)

            # Apply transforms.
            if self.transform is not None:
                image = self.transform(image)
            return {"image": image}  # type: ignore[dict-item]

        else:
            if self.mode == "s1":
                img_4s = self.get_array(
                    self.ids[index], "s1"
                )  # [4,2,264,264] float32 or uint8.
            elif self.mode == "s2a":
                img_4s = self.get_array(
                    self.ids[index], "s2a", self.bands
                )  # [4,12,264,264] int16 or uint8.
            elif self.mode == "s2c":
                img_4s = self.get_array(
                    self.ids[index], "s2c", self.bands
                )  # [4,13,264,264] int16 or uint8.
            else:
                raise ValueError(
                    f"Invalid value for mode {self.mode}! Must be `s1`, `s2a` or `s2c`"
                )

            # Convert to tensor from ndarray.
            img_4s = torch.from_numpy(img_4s)

            if self.season_transform is None:
                # Apply transforms.
                if self.transform is not None:
                    img_4s = self.transform(img_4s)

                return {"image": img_4s}

            elif self.season_transform.season == "random":
                img = self.season_transform(img_4s)

                # Apply transforms.
                if self.transform is not None:
                    img = self.transform(img)

                return {"image": img}

            elif self.season_transform.season == "pair":
                # Randomly pick 2 seasons from the possible 4.
                img1, img2 = self.season_transform(img_4s)

                # Note: Additional transforms should be applied via PairedNonGeoDataset.
                return {"image": torch.stack((img1, img2))}

            return {"image": img_4s}

    def get_array(
        self, patch_id: str, mode: str, bands: Optional[tuple[str, ...]] = None
    ):
        data_root_patch = os.path.join(self.root, mode, patch_id)
        patch_seasons = os.listdir(data_root_patch)
        seasons = []

        if mode == "s1":
            bands = self.ALL_BANDS_S1_GRD
            mean = self.S1_MEAN
            std = self.S1_STD
        elif mode == "s2a":
            bands = self.ALL_BANDS_S2_L2A if not bands else bands
            mean = self.S2A_MEAN
            std = self.S2A_STD
        elif mode == "s2c":
            bands = self.ALL_BANDS_S2_L1C if not bands else bands
            mean = self.S2C_MEAN
            std = self.S2C_STD

        assert bands is not None
        normalise = Normalize(
            mean=[mean[band] for band in bands], std=[std[band] for band in bands]
        )

        for patch_id_season in patch_seasons:
            chs = []
            for band in bands:
                patch_path = os.path.join(
                    data_root_patch, patch_id_season, f"{band}.tif"
                )
                with rasterio.open(patch_path) as dataset:
                    ch = dataset.read(1)
                    ch = cv2.resize(
                        ch, dsize=(264, 264), interpolation=cv2.INTER_LINEAR_EXACT
                    )  # [264,264]

                chs.append(ch)

            img = np.stack(chs, axis=0)  # [C,264,264]
            if self.normalize or (self.dtype == "uint8" and mode == "s1"):
                img = normalise(img)

            seasons.append(img)

        img_4s = np.stack(seasons, axis=0)  # [4,C,264,264]

        if self.normalize:
            return img_4s
        elif self.dtype == "uint8":
            if mode == "s1":
                return img_4s
            else:
                return (img_4s / 10000.0 * 255.0).astype("uint8")
        else:
            if mode == "s1":
                return img_4s.astype("float32")
            else:
                return img_4s.astype("int16")

    def __len__(self) -> int:
        return self.length


class Subset(Dataset):  # type: ignore[type-arg]

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):  # type: ignore[type-arg]
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        assert self.batch_sampler is not None
        assert isinstance(self.batch_sampler, BatchSampler)
        return len(self.batch_sampler.sampler)  # type: ignore[arg-type]

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)


def make_lmdb(dataset, lmdb_file, num_workers: int = 6) -> None:
    loader = InfiniteDataLoader(
        dataset, num_workers=num_workers, collate_fn=lambda x: x[0]
    )
    env = lmdb.open(lmdb_file, map_size=1099511627776)
    txn = env.begin(write=True)

    for index, sample in tqdm(
        enumerate(loader), total=len(dataset), desc="Creating LMDB"
    ):
        images = sample["image"]

        sample = np.array(images)
        obj = (sample.tobytes(), sample.shape)

        txn.put(str(index).encode(), pickle.dumps(obj))

        if index % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()
