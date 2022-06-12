# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Module containing custom samplers for `torch` datasets."""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Tuple, Optional, Union, Iterator, List
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import GeoSampler, BatchGeoSampler
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box
from minerva.utils import utils
import random


# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class RandomPairGeoSampler(GeoSampler):
    """Samples geo-close pairs of elements from a region of interest randomly."""

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: int,
        roi: Optional[BoundingBox] = None,
        max_r: Optional[float] = 256,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset (GeoDataset): Dataset to index from.
            size (Tuple[float, float] | float): dimensions of each :term:`patch` in units of CRS.
            length (int): number of random samples to draw per epoch.
            roi (BoundingBox): Optional; region of interest to sample from (minx, maxx, miny, maxy, mint, maxt).
                (defaults to the bounds of ``dataset.index``).
            max_r (float):
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.length = length
        self.max_r = max_r
        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx > self.size[1]
                and bounds.maxy - bounds.miny > self.size[0]
            ):
                self.hits.append(hit)

        # Define the distance to add to an existing bounding box to get
        # the box to sample the other side of the pair from.
        self.r = (self.max_r - self.size[0], self.max_r - self.size[1])

    def __iter__(self) -> Iterator[Tuple[BoundingBox, BoundingBox]]:
        """Return a pair of BoundingBox indices of a dataset that are geospatially close.

        Returns:
            Tuple[BoundingBox, BoundingBox]: Tuple of bounding boxes to index a dataset.
        """
        for _ in range(len(self)):
            # Choose a random tile.
            hit = random.choice(self.hits)
            bounds = BoundingBox(*hit.bounds)

            bbox_a, bbox_b = get_pair_bboxes(bounds, self.size, self.res, self.r)

            yield bbox_a, bbox_b

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class RandomPairBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random `patches` as possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[BoundingBox] = None,
        max_r: Optional[float] = 256.0,
        tiles_per_batch: Optional[int] = 4,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch` in units of CRS
            batch_size: number of samples per batch
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.batch_size = batch_size
        self.length = length
        self.max_r = max_r
        self.hits = list(self.index.intersection(tuple(self.roi), objects=True))

        self.tiles_per_batch = tiles_per_batch

        if self.batch_size % tiles_per_batch == 0:
            self.sam_per_tile = int(self.batch_size / tiles_per_batch)
        else:
            raise ValueError(
                "Value given for `tiles_per_batch` is not a multiple of batch_size"
            )

        # Define the distance to add to an existing bounding box to get
        # the box to sample the other side of the pair from.
        self.r = (self.max_r - self.size[0], self.max_r - self.size[1])

    def __iter__(self) -> Iterator[List[Tuple[BoundingBox, BoundingBox]]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            batch = []
            for _ in range(self.tiles_per_batch):
                # Choose a random tile
                hit = random.choice(self.hits)
                bounds = BoundingBox(*hit.bounds)

                # Choose random indices within that tile
                for _ in range(self.sam_per_tile):
                    bbox_a, bbox_b = get_pair_bboxes(
                        bounds, self.size, self.res, self.r
                    )
                    batch.append((bbox_a, bbox_b))

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size


def get_greater_bbox(bbox: BoundingBox, r: Tuple[float, float]) -> BoundingBox:
    """Return a bounding box at max_r distance around the first box."""
    return BoundingBox(
        bbox.minx - r[0],
        bbox.maxx + r[0],
        bbox.miny - r[1],
        bbox.maxy + r[1],
        bbox.mint,
        bbox.maxt,
    )


def get_pair_bboxes(
    bounds: BoundingBox,
    size: Union[Tuple[float, float], float],
    res: float,
    r: Tuple[float, float],
) -> Tuple[BoundingBox, BoundingBox]:
    # Choose a random index within that tile.
    bbox_a = get_random_bounding_box(bounds, size, res)

    max_bounds = get_greater_bbox(bbox_a, r)

    # Check that the new bbox cannot exceed the bounds of the tile.
    max_bounds = utils.check_within_bounds(max_bounds, bounds)

    # Randomly sample another box at a max distance of max_r from box_a.
    bbox_b = get_random_bounding_box(max_bounds, size, res)

    return bbox_a, bbox_b
