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
r"""Utility functions for datasets in :mod:`minerva`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "MinervaNonGeoDataset",
    "load_all_samples",
    "make_bounding_box",
    "intersect_datasets",
    "unionise_datasets",
    "get_random_sample",
    "masks_or_labels",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    NonGeoDataset,
    UnionDataset,
)
from torchgeo.datasets.utils import BoundingBox
from torchgeo.samplers.utils import get_random_bounding_box
from tqdm import tqdm

from minerva.utils import utils


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaNonGeoDataset(NonGeoDataset):
    def __or__(
        self,
        other: Union["MinervaNonGeoDataset", "MinervaConcatDataset"],
    ) -> "MinervaConcatDataset":
        """Take the union of two :class:`MinervaNonGeoDataset`.

        Args:
            other (MinervaNonGeoDataset | ~torch.utils.data.ConcatDataset): Another dataset.

        Returns:
            ~torch.utils.data.ConcatDataset[MinervaNonGeoDataset]: A single dataset.

        .. versionadded:: 0.28
        """
        return MinervaConcatDataset([self, other])


class MinervaConcatDataset(ConcatDataset):  # type: ignore[type-arg]
    def __or__(
        self,
        other: Union[MinervaNonGeoDataset, "MinervaConcatDataset"],
    ) -> "MinervaConcatDataset":
        """Take the union of two :class:`MinervaNonGeoDataset`.

        Args:
            other (MinervaNonGeoDataset | MinervaConcatDataset): Another dataset.

        Returns:
            MinervaConcatDataset: A single dataset.

        .. versionadded:: 0.28
        """
        return MinervaConcatDataset([self, other])

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return getattr(self, name)
        elif name in self.datasets[0].__dict__:
            return getattr(self.datasets[0], name)  # pragma: no cover
        elif name in self.datasets[1].__dict__:
            return getattr(self.datasets[1], name)  # pragma: no cover
        else:
            raise AttributeError


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def intersect_datasets(datasets: Sequence[GeoDataset]) -> IntersectionDataset:
    r"""
    Intersects a list of :class:`~torchgeo.datasets.GeoDataset` together to return a single dataset object.

    Args:
        datasets (list[~torchgeo.datasets.GeoDataset]): List of datasets to intersect together.
            Should have some geospatial overlap.

    Returns:
        ~torchgeo.datasets.IntersectionDataset: Final dataset object representing an intersection
        of all the parsed datasets.
    """
    master_dataset: GeoDataset | IntersectionDataset = datasets[0]

    for i in range(len(datasets) - 1):
        master_dataset = master_dataset & datasets[i + 1]

    assert isinstance(master_dataset, IntersectionDataset)
    return master_dataset


def unionise_datasets(
    datasets: Sequence[GeoDataset],
    transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> UnionDataset:
    """Unionises a list of :class:`~torchgeo.datasets.GeoDataset` together to return a single dataset object.

    Args:
        datasets (list[~torchgeo.datasets.GeoDataset]): List of datasets to unionise together.
        transforms (): Optional; Function that will transform any sample yielded from the union.

    .. note::
        The transforms of ``transforms`` will be applied to the sample after any transforms applied to it by
        the constituent dataset of the union the dataset came from. Therefore, ``transforms`` needs to compatible
        with all possible samples of the union.

    Returns:
        ~torchgeo.datasets.UnionDataset: Final dataset object representing an union of all the parsed datasets.
    """
    master_dataset: GeoDataset | UnionDataset = datasets[0]

    for i in range(len(datasets) - 1):
        master_dataset = master_dataset | datasets[i + 1]

    master_dataset.transforms = transforms
    assert isinstance(master_dataset, UnionDataset)
    return master_dataset


def concatenate_datasets(
    datasets: Sequence[NonGeoDataset],
    transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> MinervaConcatDataset:
    """Unionises a list of :class:`~torchgeo.datasets.GeoDataset` together to return a single dataset object.

    Args:
        datasets (list[~minerva.datasets.MinervaNonGeoDataset]): List of datasets to concatenate together.
        transforms (): Optional; Function that will transform any sample yielded from the union.

    .. note::
        The transforms of ``transforms`` will be applied to the sample after any transforms applied to it by
        the constituent dataset of the union the dataset came from. Therefore, ``transforms`` needs to compatible
        with all possible samples of the union.

    Returns:
        ~minerva.datasets.MinervaConcatDataset: Final dataset object representing an union of all the parsed datasets.
    """
    assert isinstance(datasets[0], MinervaNonGeoDataset)
    master_dataset: MinervaNonGeoDataset | MinervaConcatDataset = datasets[0]

    for i in range(len(datasets) - 1):
        master_dataset = master_dataset | datasets[i + 1]  # type: ignore[operator]

    if hasattr(master_dataset, "transforms"):
        master_dataset.transforms = transforms  # type: ignore[union-attr]
    assert isinstance(master_dataset, MinervaConcatDataset)
    return master_dataset


def make_bounding_box(roi: Sequence[float] | bool = False) -> Optional[BoundingBox]:
    """Construct a :class:`~torchgeo.datasets.utils.BoundingBox` object from the corners of the box.
    ``False`` for no :class:`~torchgeo.datasets.utils.BoundingBox`.

    Args:
        roi (~typing.Sequence[float] | bool): Either a :class:`tuple` or array of values defining
            the corners of a bounding box or False to designate no BoundingBox is defined.

    Returns:
        ~torchgeo.datasets.utils.BoundingBox | None: Bounding box made from parsed values
        or ``None`` if ``False`` was given.
    """
    if roi is False:
        return None
    elif roi is True:
        raise ValueError(
            "``roi`` must be a sequence of floats or ``False``, not ``True``"
        )
    else:
        return BoundingBox(*roi)


def load_all_samples(
    dataloader: DataLoader[Iterable[Any]],
    target_key: Literal["mask", "label"] = "mask",
) -> NDArray[Any]:
    """Loads all sample masks from parsed :class:`~torch.utils.data.DataLoader` and computes the modes of their classes.

    Args:
        dataloader (~torch.utils.data.DataLoader): DataLoader containing samples. Must be using a dataset with
            ``__len__`` attribute and a sampler that returns a dict with a ``"mask"`` or ``"label"`` key.
        target_key (~typing.Literal["mask", "label"]): Optional; Key for the targets in the dataset.
            Either ``"mask"`` or ``"label"``.

    Returns:
        ~numpy.ndarray: 2D array of the class modes within every sample defined by the parsed
        :class:`~torch.utils.data.DataLoader`.
    """
    sample_modes: list[list[tuple[int, int]]] = []
    for sample in tqdm(dataloader):
        modes = utils.find_modes(sample[target_key])
        sample_modes.append(modes)

    return np.array(sample_modes, dtype=object)


def get_random_sample(
    dataset: GeoDataset, size: tuple[int, int] | int, res: float
) -> dict[str, Any]:
    """Gets a random sample from the provided dataset of size ``size`` and at ``res`` resolution.

    Args:
        dataset (~torchgeo.datasets.GeoDataset): Dataset to sample from.
        size (tuple[int, int] | int): Size of the patch to sample.
        res (float): Resolution of the patch.

    Returns:
        dict[str, ~typing.Any]: Random sample from the dataset.
    """
    return dataset[get_random_bounding_box(dataset.bounds, size, res)]


def load_dataset_from_cache(
    cached_dataset_path: Path,
) -> NonGeoDataset | GeoDataset:
    """Load a pickled dataset object in from a cache.

    Args:
        cached_dataset_path (~pathlib.Path): Path to the cached dataset object

    Returns:
        ~torchgeo.datasets.GeoDataset: Dataset object loaded from cache
    """

    with open(cached_dataset_path, "rb") as fp:
        dataset = pickle.load(fp)

    assert isinstance(dataset, (NonGeoDataset, GeoDataset))
    return dataset


def cache_dataset(
    dataset: GeoDataset | NonGeoDataset, cached_dataset_path: Path
) -> None:
    """Pickle and cache a dataset object.

    Args:
        dataset (~torchgeo.datasets.GeoDataset | ~torchgeo.datasets.NonGeoDataset): Dataset object to cache.
        cached_dataset_path (~pathlib.Path): Path to save dataset to.
    """
    # Create missing directories in the path if they don't exist.
    cached_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cached_dataset_path, "xb") as fp:
        pickle.dump(dataset, fp)


def masks_or_labels(dataset_params: dict[str, Any]) -> str:
    for key in dataset_params.keys():
        if key not in (
            "sampler",
            "transforms",
            "image",
            "imagery_config",
            "data_config",
        ):
            if key == "mask" or key == "label":
                return key
            else:
                raise ValueError("Invalid keys in dataset params!")

    raise ValueError("No mask or label data was specified in the dataset parameters")
