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
"""Module containing custom samplers for :mod:`torchgeo` datasets."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import random
from typing import Iterator, List, Optional, Tuple, Union

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import BatchGeoSampler, GeoSampler
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box

from minerva.utils import utils

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
    """Samples geo-close pairs of elements from a region of interest randomly.

    An extension to :class:`RandomGeoSampler` that supports paired sampling (i.e for GeoCLR).
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: int,
        roi: Optional[BoundingBox] = None,
        max_r: float = 256.0,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset (GeoDataset): Dataset to index from.
            size (Tuple[float, float] | float): Dimensions of each :term:`patch` in units of CRS.
            length (int): number of random samples to draw per epoch.
            roi (BoundingBox): Optional; Region of interest to sample from (``minx``, ``maxx``, ``miny``, ``maxy``,
                ``mint``, ``maxt``). (defaults to the bounds of ``dataset.index``).
            max_r (float): Optional; Maximum geo-spatial distance (from centre to centre)
                to sample matching sample from.
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.length = length
        self.max_r = max_r
        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)  # type: ignore
            if (
                bounds.maxx - bounds.minx > self.size[1]
                and bounds.maxy - bounds.miny > self.size[0]
            ):
                self.hits.append(hit)

    def __iter__(self) -> Iterator[Tuple[BoundingBox, BoundingBox]]:  # type: ignore[override]
        """Return a pair of BoundingBox indices of a dataset that are geospatially close.

        Returns:
            Tuple[BoundingBox, BoundingBox]: Tuple of bounding boxes to index a dataset.
        """
        for _ in range(len(self)):
            # Choose a random tile.
            hit = random.choice(self.hits)
            bounds = BoundingBox(*hit.bounds)

            bbox_a, bbox_b = get_pair_bboxes(bounds, self.size, self.res, self.max_r)

            yield bbox_a, bbox_b

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            int: Length of the epoch.
        """
        return self.length


class RandomPairBatchGeoSampler(BatchGeoSampler):
    """Samples batches of pairs of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`patches` as possible.

    An extension to :class:`RandomBatchGeoSampler` that supports paired sampling (i.e. for GeoCLR)
    and ability to samples from multiple tiles per batch to increase variance of batch.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[BoundingBox] = None,
        max_r: float = 256.0,
        tiles_per_batch: int = 4,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset (GeoDataset): Dataset to index from.
            size (Union[Tuple[float, float], float]): Dimensions of each :term:`patch` in units of CRS.
            batch_size (int): Number of samples per batch.
            length (int): Number of samples per epoch.
            roi (BoundingBox): Optional; Region of interest to sample from (``minx``, ``maxx``, ``miny``, ``maxy``,
                ``mint``, ``maxt``). (defaults to the bounds of ``dataset.index``)
            max_r (float): Optional; Maximum geo-spatial distance (from centre to centre)
                to sample matching sample from.
            tiles_per_batch (int): Optional; Number of tiles to sample from per batch.
                Must be a multiple of ``batch_size``.

        Raises:
            ValueError: If ``tiles_per_batch`` is not a multiple of ``batch_size``.
        """
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
            raise ValueError(
                "Value given for `tiles_per_batch` is not a multiple of batch_size"
            )

    def __iter__(self) -> Iterator[List[Tuple[BoundingBox, BoundingBox]]]:  # type: ignore[override]
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
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
    bbox: BoundingBox, r: float, size: Union[float, Tuple[float, float]]
) -> BoundingBox:
    """Return a bounding box at ``max_r`` distance around the first box.

    Args:
        bbox (BoundingBox): Bounding box of the original sample.
        r (int): Distance in pixels to extend the original bounding box by
            to get a new greater bounds to sample from.
        size (float | Tuple[float, float]): The (``x``, ``y``) size of the :term:`patch` that ``bbox``
            represents in pixels.

    Returns:
        BoundingBox: Greater bounds around original bounding box to sample from.
    """
    x: float
    if type(size) == tuple:
        assert isinstance(size, tuple)
        x = size[0]
    else:
        assert isinstance(size, float)
        x = size

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
    size: Union[Tuple[float, float], float],
    res: float,
    max_r: float,
) -> Tuple[BoundingBox, BoundingBox]:
    """Samples a pair of bounding boxes geo-spatially close to each other.

    Args:
        bounds (BoundingBox): Maximum bounds of the :term:`tile` to sample pair from.
        size (Union[Tuple[float, float], float]): Size of each :term:`patch`.
        res (float): Resolution to sample :term:`patch` at.
        r (Tuple[float, float]): ``x`` and ``y`` padding around original :term:`patch`
            to sample new :term:`patch` from.

    Returns:
        Tuple[BoundingBox, BoundingBox]: Pair of bounding boxes to sample pair of patches from dataset.
    """
    # Choose a random index within that tile.
    bbox_a = get_random_bounding_box(bounds, size, res)

    max_bounds = get_greater_bbox(bbox_a, max_r, size)

    # Check that the new bbox cannot exceed the bounds of the tile.
    max_bounds = utils.check_within_bounds(max_bounds, bounds)

    # Randomly sample another box at a max distance of max_r from box_a.
    bbox_b = get_random_bounding_box(max_bounds, size, res)

    return bbox_a, bbox_b
