"""Module containing custom samplers for `torch` datasets.

    Copyright (C) 2022 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

TODO:
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Tuple, Optional, Union, Iterator
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers import GeoSampler
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box
from minerva.utils import utils
import random


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

            # Choose a random index within that tile.
            box_a = get_random_bounding_box(bounds, self.size, self.res)

            # Define a bounding box at max_r distance around the first box.
            max_bounds = BoundingBox(
                box_a.minx - self.r[0],
                box_a.maxx + self.r[0],
                box_a.miny - self.r[1],
                box_a.maxy + self.r[1],
                box_a.mint,
                box_a.maxt,
            )

            # Check that the new bbox cannot exceed the bounds of the tile.
            max_bounds = utils.check_within_bounds(max_bounds, bounds)

            # Randomly sample another box at a max distance of max_r from box_a.
            box_b = get_random_bounding_box(max_bounds, self.size, self.res)

            yield box_a, box_b

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length
