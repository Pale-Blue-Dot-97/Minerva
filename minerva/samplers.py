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
"""Module containing custom samplers for :mod:`torchgeo` datasets.

:class:`DistributedSamplerWrapper` is sourced from :mod:`catalyst`
https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/dataset.py#L6
"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "RandomPairGeoSampler",
    "RandomPairBatchGeoSampler",
    "DistributedSamplerWrapper",
    "get_greater_bbox",
    "get_pair_bboxes",
    "get_sampler",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import random
import re
from operator import itemgetter
from typing import Any, Iterator, Optional, Sequence

import hydra
import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler
from torchgeo.datasets import GeoDataset, NonGeoDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import BatchGeoSampler, RandomGeoSampler, Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box

from minerva.datasets.utils import make_bounding_box
from minerva.utils import utils


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class RandomPairGeoSampler(RandomGeoSampler):
    """Samples geo-close pairs of elements from a region of interest randomly.

    An extension to :class:`~torchgeo.samplers.RandomGeoSampler` that supports paired sampling (i.e for GeoCLR).

    .. note::
        The ``size`` argument can either be:

        * a single :class:`float` - in which case the same value is used for the height and
            width dimension
        * a :class:`tuple` of two floats - in which case, the first :class:`float` is used for the
            height dimension, and the second :class:`float` for the width dimension

    Args:
        dataset (~torchgeo.datasets.GeoDataset): Dataset to index from.
        size (tuple[float, float] | float): Dimensions of each :term:`patch` in units of CRS.
        length (int): number of random samples to draw per epoch.
        roi (~torchgeo.datasets.utils.BoundingBox): Optional; Region of interest to sample from
            (``minx``, ``maxx``, ``miny``, ``maxy``, ``mint``, ``maxt``). (defaults to the bounds of ``dataset.index``).
        units (~torchgeo.samplers.Units): Optional; Defines whether ``size`` is in pixel or CRS units.
        max_r (float): Optional; Maximum geo-spatial distance (from centre to centre)
            to sample matching sample from.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        length: int,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        max_r: float = 256.0,
    ) -> None:
        super().__init__(dataset, size, length, roi, units)
        self.max_r = max_r

    def __iter__(self) -> Iterator[tuple[BoundingBox, BoundingBox]]:  # type: ignore[override]
        """Return a pair of :class:`~torchgeo.datasets.utils.BoundingBox` indices of a dataset
        that are geospatially close.

        Returns:
            tuple[~torchgeo.datasets.utils.BoundingBox, ~torchgeo.datasets.utils.BoundingBox]: Tuple of
            bounding boxes to index a dataset.
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            bbox_a, bbox_b = get_pair_bboxes(bounds, self.size, self.res, self.max_r)

            yield bbox_a, bbox_b


class RandomPairBatchGeoSampler(BatchGeoSampler):
    """Samples batches of pairs of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`patches` as possible.

    An extension to :class:`~torchgeo.samplers.RandomBatchGeoSampler` that supports
    paired sampling (i.e. for GeoCLR) and ability to samples from multiple tiles per batch
    to increase variance of batch.

    .. note::
        The ``size`` argument can either be:

        * a single :class:`float` - in which case the same value is used for the height and
            width dimension
        * a :class:`tuple` of two floats - in which case, the first :class:`float` is used for the
            height dimension, and the second *float* for the width dimension

    Args:
        dataset (~torchgeo.datasets.GeoDataset): Dataset to index from.
        size (tuple[float, float] | float): Dimensions of each :term:`patch` in units of CRS.
        batch_size (int): Number of samples per batch.
        length (int): Number of samples per epoch.
        roi (~torchgeo.datasets.utils.BoundingBox): Optional; Region of interest to sample from
            (``minx``, ``maxx``, ``miny``, ``maxy``, ``mint``, ``maxt``). (defaults to the bounds of ``dataset.index``)
        max_r (float): Optional; Maximum geo-spatial distance (from centre to centre)
            to sample matching sample from.
        tiles_per_batch (int): Optional; Number of tiles to sample from per batch.
            Must be a multiple of ``batch_size``.

    Raises:
        ValueError: If ``tiles_per_batch`` is not a multiple of ``batch_size``.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        batch_size: int,
        length: int,
        roi: Optional[BoundingBox] = None,
        max_r: float = 256.0,
        tiles_per_batch: int = 4,
    ) -> None:
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.batch_size = batch_size
        self.length = length
        self.max_r = max_r
        self.hits = list(self.index.intersection(tuple(self.roi), objects=True))

        self.tiles_per_batch = tiles_per_batch

        if self.batch_size % tiles_per_batch == 0:
            self.sam_per_tile = self.batch_size // tiles_per_batch
        else:
            raise ValueError(f"{tiles_per_batch=} is not a multiple of {batch_size=}")

    def __iter__(self) -> Iterator[list[tuple[BoundingBox, BoundingBox]]]:  # type: ignore[override]
        """Return the indices of a dataset.

        Returns:
            Batch of paired :class:`~torchgeo.datasets.utils.BoundingBox` to index a dataset.
        """
        for _ in range(len(self)):
            batch = []
            for _ in range(self.tiles_per_batch):
                # Choose a random tile
                hit = random.choice(self.hits)
                bounds = BoundingBox(*hit.bounds)  # type: ignore

                # Choose random indices within that tile
                for _ in range(self.sam_per_tile):
                    bbox_a, bbox_b = get_pair_bboxes(
                        bounds, self.size, self.res, self.max_r
                    )
                    batch.append((bbox_a, bbox_b))

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            int: Number of batches in an epoch
        """
        return self.length // self.batch_size


def get_greater_bbox(
    bbox: BoundingBox,
    r: float,
    size: float | int | Sequence[float],
) -> BoundingBox:
    """Return a bounding box at ``r`` distance around the first box.

    Args:
        bbox (~torchgeo.datasets.utils.BoundingBox): Bounding box of the original sample.
        r (float): Distance in pixels to extend the original bounding box by
            to get a new greater bounds to sample from.
        size (float | ~typing.Sequence[float]): The (``x``, ``y``) size of the :term:`patch` that ``bbox``
            represents in pixels. Will only use size[0] if a :class:`~typing.Sequence`.

    Returns:
        ~torchgeo.datasets.utils.BoundingBox: Greater bounds around original bounding box to sample from.
    """
    x: float
    if isinstance(size, Sequence):
        assert isinstance(size, Sequence)
        x = float(size[0])
    else:
        assert isinstance(size, (float, int))
        x = float(size)

    # Calculates the geospatial distance to add to the existing bounding box to get
    # the box to sample the other side of the pair from.
    r_in_crs = r * abs(bbox.maxx - bbox.minx) / float(x)

    return BoundingBox(
        bbox.minx - r_in_crs,
        bbox.maxx + r_in_crs,
        bbox.miny - r_in_crs,
        bbox.maxy + r_in_crs,
        bbox.mint,
        bbox.maxt,
    )


def get_pair_bboxes(
    bounds: BoundingBox,
    size: tuple[float, float] | float,
    res: float,
    max_r: float,
) -> tuple[BoundingBox, BoundingBox]:
    """Samples a pair of bounding boxes geo-spatially close to each other.

    Args:
        bounds (~torchgeo.datasets.utils.BoundingBox): Maximum bounds of the :term:`tile` to sample pair from.
        size (tuple[float, float] | float): Size of each :term:`patch`.
        res (float): Resolution to sample :term:`patch` at.
        max_r (float): Padding around original :term:`patch` to sample new :term:`patch` from.

    Returns:
        tuple[~torchgeo.datasets.utils.BoundingBox, ~torchgeo.datasets.utils.BoundingBox]: Pair of bounding boxes
        to sample pair of patches from dataset.
    """
    # Choose a random index within that tile.
    bbox_a = get_random_bounding_box(bounds, size, res)

    max_bounds = get_greater_bbox(bbox_a, max_r, size)

    # Check that the new bbox cannot exceed the bounds of the tile.
    max_bounds = utils.check_within_bounds(max_bounds, bounds)

    # Randomly sample another box at a max distance of max_r from box_a.
    bbox_b = get_random_bounding_box(max_bounds, size, res)

    return bbox_a, bbox_b


class DistributedSamplerWrapper(DistributedSampler):  # type: ignore[type-arg]
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.

    .. note::
        Sourced from :mod:`catalyst` https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
    """

    def __init__(
        self,
        sampler: Sampler[Any],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))  # type: ignore[arg-type]


class DatasetFromSampler(Dataset):  # type: ignore[type-arg]
    """Dataset to create indexes from `sampler`.

    Args:
        sampler (~torch.utils.data.Sampler): PyTorch sampler

    .. note::
        Sourced from :mod:`catalyst` https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/dataset.py#L6
    """

    def __init__(self, sampler: Sampler[Any]):
        """Initialisation for :class:`DatasetFromSampler`."""
        self.sampler = sampler
        self.sampler_list: Optional[list[Sampler[Any]]] = None

    def __getitem__(self, index: int) -> Any:
        """Gets element of the dataset.

        Args:
            index (int): Index of the element in the dataset.

        Returns:
            ~typing.Any: Single element by index.
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset
        """
        return len(self.sampler)  # type: ignore[arg-type]


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_sampler(
    params: dict[str, Any],
    dataset: GeoDataset | NonGeoDataset,
    batch_size: Optional[int] = None,
) -> Sampler[Any]:
    """Use :meth:`hydra.utils.instantiate` to get the sampler using config parameters.

    Args:
        params (dict[str, ~typing.Any]): Sampler parameters. Must include the ``_target_`` key pointing to
            the sampler class.
        dataset (~torchgeo.datasets.GeoDataset, ~torchgeo.datasets.NonGeoDataset]): Dataset to sample.
        batch_size (int): Optional; Batch size to sample if using a batch sampler.
            Use if you need to overwrite the config parameters due to distributed computing as the batch size
            needs to modified to split the batch across devices. Defaults to ``None``.

    Returns:
        ~torch.utils.data.Sampler: Sampler requested by config parameters.
    """

    batch_sampler = True if re.search(r"Batch", params["_target_"]) else False
    if batch_sampler and batch_size is not None:
        params["batch_size"] = batch_size

    if "roi" in params:
        sampler = hydra.utils.instantiate(
            params, dataset=dataset, roi=make_bounding_box(params["roi"])
        )
    else:
        if "torchgeo" in params["_target_"]:
            sampler = hydra.utils.instantiate(params, dataset=dataset)
        else:
            sampler = hydra.utils.instantiate(params, data_source=dataset)

    assert isinstance(sampler, Sampler)
    return sampler
