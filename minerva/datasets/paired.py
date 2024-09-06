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
r"""Datasets to handle paired sampling for use in Siamese learning."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = ["Harry Baker", "Jonathon Hare"]
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "PairedGeoDataset",
    "PairedUnionDataset",
    "PairedNonGeoDataset",
    "PairedConcatDataset",
    "SamplePair",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import random
from inspect import signature
from typing import Any, Callable, Optional, Sequence, overload

import hydra
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    NonGeoDataset,
    RasterDataset,
    UnionDataset,
)
from torchgeo.datasets.utils import BoundingBox, concat_samples, merge_samples
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as ft

from minerva.utils import utils

from .utils import MinervaConcatDataset, get_random_sample


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class PairedGeoDataset(RasterDataset):
    """Custom dataset to act as a wrapper to other datasets to handle paired sampling.

    Attributes:
        dataset (~torchgeo.datasets.RasterDataset): Wrapped dataset to sampled from.

    Args:
        dataset (~typing.Callable[..., ~torchgeo.datasets.GeoDataset]): Constructor for a
            :class:`~torchgeo.datasets.RasterDataset` to be wrapped for paired sampling.
    """

    def __new__(  # type: ignore[misc]
        cls,
        dataset: Callable[..., GeoDataset] | GeoDataset,
        *args,
        **kwargs,
    ) -> "PairedGeoDataset" | "PairedUnionDataset":
        if isinstance(dataset, UnionDataset):
            return PairedUnionDataset(
                dataset.datasets[0], dataset.datasets[1], *args, **kwargs
            )
        else:
            return super(PairedGeoDataset, cls).__new__(cls)

    def __getnewargs__(self):
        return self.dataset, self._args, self._kwargs

    @overload
    def __init__(
        self, dataset: Callable[..., GeoDataset], *args, **kwargs
    ) -> None: ...  # pragma: no cover

    @overload
    def __init__(
        self, dataset: GeoDataset, *args, **kwargs
    ) -> None: ...  # pragma: no cover

    @overload
    def __init__(self, dataset: str, *args, **kwargs) -> None: ...  # pragma: no cover

    def __init__(
        self,
        dataset: Callable[..., GeoDataset] | GeoDataset | str,
        *args,
        **kwargs,
    ) -> None:
        # Needed for pickling/ unpickling.
        self._args = args
        self._kwargs = kwargs

        if isinstance(dataset, str):
            dataset = hydra.utils.get_method(dataset)

        if isinstance(dataset, GeoDataset):
            self.dataset = dataset
            self._res = dataset.res
            self._crs = dataset.crs

        elif "paths" in signature(dataset).parameters:
            super_sig = signature(RasterDataset.__init__).parameters.values()
            super_kwargs = {
                key.name: kwargs[key.name] for key in super_sig if key.name in kwargs
            }

            # Make sure PairedGeoDataset has access to the `all_bands` attribute of the dataset.
            # Needed for a subset of the bands to be selected if so desired.
            if hasattr(dataset, "all_bands"):
                self.all_bands = dataset.all_bands

            super().__init__(*args, **super_kwargs)
            self.dataset = dataset(*args, **kwargs)

        else:
            raise ValueError(
                f"``dataset`` is of unsupported type {type(dataset)} not GeoDataset"
            )

    def __getitem__(  # type: ignore[override]
        self, queries: tuple[BoundingBox, BoundingBox]
    ) -> tuple[dict[str, Any], ...]:
        return self.dataset.__getitem__(queries[0]), self.dataset.__getitem__(
            queries[1]
        )

    def __and__(self, other: "PairedGeoDataset") -> IntersectionDataset:  # type: ignore[override]
        """Take the intersection of two :class:`PairedGeoDataset`.

        Args:
            other (PairedGeoDataset): Another dataset.

        Returns:
            IntersectionDataset: A single dataset.

        Raises:
            ValueError: If other is not a :class:`PairedGeoDataset`

        .. versionadded:: 0.24
        """
        if not isinstance(other, PairedGeoDataset):
            raise ValueError(
                f"Intersecting a dataset of {type(other)} and a PairedGeoDataset is not supported!"
            )

        return IntersectionDataset(
            self, other, collate_fn=utils.pair_collate(concat_samples)
        )

    def __or__(self, other: GeoDataset) -> "PairedUnionDataset":  # type: ignore[override]
        """Take the union of two :class:~`torchgeo.datasets.GeoDataset`.

        Args:
            other (~torchgeo.datasets.GeoDataset): Another dataset.

        Returns:
            PairedUnionDataset: A single dataset.

        .. versionadded:: 0.24
        """
        return PairedUnionDataset(self, other)

    def __getattr__(self, name: str) -> Any:
        """
        Called only if the attribute 'name' is not found by usual means.
        Checks if 'name' exists in the dataset attribute.
        """
        # Instead of calling __getattr__ directly, access the dataset attribute directly
        dataset = super().__getattribute__("dataset")
        if hasattr(dataset, name):
            return getattr(dataset, name)
        # If not found in dataset, raise an AttributeError
        raise AttributeError(f"{name} cannot be found in self or dataset")

    def __getattribute__(self, name: str) -> Any:
        """
        Overrides default attribute access method to prevent recursion.
        Checks 'self' first and uses __getattr__ for fallback.
        """
        try:
            # First, try to get the attribute from the current instance
            return super().__getattribute__(name)
        except AttributeError:
            # If not found in self, __getattr__ will check in the dataset
            dataset = super().__getattribute__("dataset")
            if hasattr(dataset, name):
                return getattr(dataset, name)
            # Raise AttributeError if not found in dataset either
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __repr__(self) -> Any:
        return self.dataset.__repr__()

    @staticmethod
    def plot(
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plots a sample from the dataset.

        Adapted from :meth:`torchgeo.datasets.NAIP.plot`.

        Args:
            sample (dict[str, ~typing.Any]): Sample to plot.
            show_titles (bool): Optional; Add title to the figure. Defaults to True.
            suptitle (str): Optional; Super title to add to figure. Defaults to None.

        Returns:
            ~matplotlib.figure.Figure: :mod:`matplotlib` Figure object with plot of the random patch imagery.
        """

        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        # Setup the figure.
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        # Plot the image.
        ax.imshow(image)

        # Turn the axis off.
        ax.axis("off")

        # Add title to figure.
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def plot_random_sample(
        self,
        size: tuple[int, int] | int,
        res: float,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plots a random sample the dataset at a given size and resolution.

        Adapted from :meth:`torchgeo.datasets.NAIP.plot`.

        Args:
            size (tuple[int, int] | int): Size of the patch to plot.
            res (float): Resolution of the patch.
            show_titles (bool): Optional; Add title to the figure. Defaults to ``True``.
            suptitle (str): Optional; Super title to add to figure. Defaults to ``None``.

        Returns:
            ~matplotlib.figure.Figure: :mod:`matplotlib` Figure object with plot of the random patch imagery.
        """

        # Get a random sample from the dataset at the given size and resolution.
        sample = get_random_sample(self.dataset, size, res)
        return self.plot(sample, show_titles, suptitle)


class PairedUnionDataset(UnionDataset):
    """Adapted form of :class:`~torchgeo.datasets.UnionDataset` to handle paired samples.

    ..warning::

        Do not use with :class:`PairedGeoDataset` as this will essentially account for paired sampling twice
        and cause a :class:`TypeError`.
    """

    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[
            [Sequence[dict[str, Any]]], dict[str, Any]
        ] = merge_samples,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:

        # Extract the actual dataset out of the paired dataset (otherwise we'll be pairing the datasets twice!)
        if isinstance(dataset1, PairedGeoDataset):
            dataset1 = dataset1.dataset
        if isinstance(dataset2, PairedGeoDataset):
            dataset2 = dataset2.dataset

        super().__init__(dataset1, dataset2, collate_fn, transforms)

        new_datasets = []
        for _dataset in self.datasets:
            if isinstance(_dataset, PairedGeoDataset):
                new_datasets.append(_dataset.dataset)
            elif isinstance(_dataset, PairedUnionDataset):
                new_datasets.append(_dataset.datasets[0] | _dataset.datasets[1])
            else:
                new_datasets.append(_dataset)

        self.datasets = new_datasets

    def __getitem__(  # type: ignore[override]
        self, query: tuple[BoundingBox, BoundingBox]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Retrieve image and metadata indexed by query.

        Uses :meth:`torchgeo.datasets.UnionDataset.__getitem__` to send each query of the pair off to get a
        sample for each and returns as a tuple.

        Args:
            query (tuple[~torchgeo.datasets.utils.BoundingBox, ~torchgeo.datasets.utils.BoundingBox]): Coordinates
                to index in the form (minx, maxx, miny, maxy, mint, maxt).

        Returns:
            tuple[dict[str, ~typing.Any], dict[str, ~typing.Any]]: Sample of data/labels and metadata at that index.
        """
        return super().__getitem__(query[0]), super().__getitem__(query[1])

    def __or__(self, other: "PairedGeoDataset") -> "PairedUnionDataset":  # type: ignore[override]
        """Take the union of a PairedUnionDataset and a :class:`PairedGeoDataset`.

        Args:
            other (PairedGeoDataset): Another dataset.

        Returns:
            PairedUnionDataset: A single dataset.

        .. versionadded:: 0.24
        """
        return PairedUnionDataset(self, other)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return getattr(self, name)
        elif name in self.datasets[0].__dict__:
            return getattr(self.datasets[0], name)  # pragma: no cover
        elif name in self.datasets[1].__dict__:
            return getattr(self.datasets[1], name)  # pragma: no cover
        else:
            raise AttributeError


class PairedNonGeoDataset(NonGeoDataset):
    """Custom dataset to act as a wrapper to other datasets to handle paired sampling.

    Attributes:
        dataset (~torchgeo.datasets.NonGeoDataset): Wrapped dataset to sampled from.
        size (int): Size of the each of the samples in the pair to sample.
        max_r (int): Distance between the centres of each of the samples in the pair.

    Args:
        dataset (~typing.Callable[..., ~torchgeo.datasets.NonGeoDataset]): Constructor for a
            :class:`~torchgeo.datasets.NonGeoDataset` to be wrapped for paired sampling.
        size (tuple(int, int) | int): Size of the each of the samples in the pair to sample.
        max_r (int): Distance between the centres of each of the samples in the pair.

    .. note::
        If :param:`size` is a :class:`tuple`, the first entry will be used.

    .. versionadded:: 0.28
    """

    def __getnewargs__(self):
        return (
            self.dataset,
            self.size,
            self.max_r,
            self.season,
            self._args,
            self._kwargs,
        )

    @overload
    def __init__(
        self,
        dataset: Callable[..., NonGeoDataset],
        size: tuple[int, int] | int,
        max_r: int,
        season: bool = False,
        *args,
        **kwargs,
    ) -> None: ...  # pragma: no cover

    @overload
    def __init__(
        self,
        dataset: NonGeoDataset,
        size: tuple[int, int] | int,
        max_r: int,
        season: bool = False,
        *args,
        **kwargs,
    ) -> None: ...  # pragma: no cover

    @overload
    def __init__(
        self,
        dataset: str,
        size: tuple[int, int] | int,
        max_r: int,
        season: bool = False,
        *args,
        **kwargs,
    ) -> None: ...  # pragma: no cover

    def __init__(
        self,
        dataset: Callable[..., NonGeoDataset] | NonGeoDataset | str,
        size: tuple[int, int] | int,
        max_r: int,
        season: bool = False,
        *args,
        **kwargs,
    ) -> None:
        # Needed for pickling/ unpickling.
        self._args = args
        self._kwargs = kwargs

        if isinstance(size, Sequence):
            size = size[0]

        self.size = size
        self.max_r = max_r
        self.season = season

        if isinstance(dataset, PairedNonGeoDataset):
            raise ValueError("Cannot pair an already paired dataset!")

        if isinstance(dataset, str):
            dataset = hydra.utils.get_method(dataset)

        if isinstance(dataset, NonGeoDataset):
            self.dataset = dataset

        elif "root" in signature(dataset).parameters:
            # Make sure PairedNonGeoDataset has access to the `all_bands` attribute of the dataset.
            # Needed for a subset of the bands to be selected if so desired.
            if hasattr(dataset, "all_bands"):
                self.all_bands = dataset.all_bands

            self.dataset = dataset(*args, **kwargs)

        else:
            raise ValueError(
                f"``dataset`` is of unsupported type {type(dataset)} not NonGeoDataset"
            )

        # Move the transforms to this wrapper from the dataset
        # Need to do this for the paired sampling to work correctly.
        if hasattr(self.dataset, "transform"):
            self.transforms = self.dataset.transform
            self.dataset.transform = None

        self.make_geo_pair = SamplePair(self.size, self.max_r, season=season)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], ...]:  # type: ignore[override]
        patch = self.dataset[index]
        image_a, image_b = self.make_geo_pair(patch["image"])

        if self.transforms:
            return self.transforms({"image": image_a}), self.transforms(
                {"image": image_b}
            )
        else:
            return {"image": image_a}, {"image": image_b}

    def __or__(self, other: "PairedNonGeoDataset") -> "PairedConcatDataset":  # type: ignore[override]
        """Take the union of two :class:`PairedNonGeoDataset`.

        Args:
            other (PairedNonGeoDataset): Another dataset.

        Returns:
            PairedConcatDataset: A single dataset.

        .. versionadded:: 0.28
        """
        return PairedConcatDataset(self, other)

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __repr__(self) -> Any:
        return self.dataset.__repr__()

    def __getattr__(self, name: str) -> Any:
        """
        Called only if the attribute 'name' is not found by usual means.
        Checks if 'name' exists in the dataset attribute.
        """
        # Instead of calling __getattr__ directly, access the dataset attribute directly
        dataset = super().__getattribute__("dataset")
        if hasattr(dataset, name):
            return getattr(dataset, name)
        # If not found in dataset, raise an AttributeError
        raise AttributeError(f"{name} cannot be found in self or dataset")

    def __getattribute__(self, name: str) -> Any:
        """
        Overrides default attribute access method to prevent recursion.
        Checks 'self' first and uses __getattr__ for fallback.
        """
        try:
            # First, try to get the attribute from the current instance
            return super().__getattribute__(name)
        except AttributeError:
            # If not found in self, __getattr__ will check in the dataset
            dataset = super().__getattribute__("dataset")
            if hasattr(dataset, name):
                return getattr(dataset, name)
            # Raise AttributeError if not found in dataset either
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    @staticmethod
    def plot(
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plots a sample from the dataset.

        Adapted from :meth:`torchgeo.datasets.NAIP.plot`.

        Args:
            sample (dict[str, ~typing.Any]): Sample to plot.
            show_titles (bool): Optional; Add title to the figure. Defaults to True.
            suptitle (str): Optional; Super title to add to figure. Defaults to None.

        Returns:
            ~matplotlib.figure.Figure: :mod:`matplotlib` Figure object with plot of the random patch imagery.
        """

        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        # Setup the figure.
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        # Plot the image.
        ax.imshow(image)

        # Turn the axis off.
        ax.axis("off")

        # Add title to figure.
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def plot_random_sample(
        self,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plots a random sample the dataset at a given size and resolution.

        Adapted from :meth:`torchgeo.datasets.NAIP.plot`.

        Args:
            show_titles (bool): Optional; Add title to the figure. Defaults to ``True``.
            suptitle (str): Optional; Super title to add to figure. Defaults to ``None``.

        Returns:
            ~matplotlib.figure.Figure: :mod:`matplotlib` Figure object with plot of the random patch imagery.
        """

        # Get a random sample from the dataset.
        sample = self.dataset[random.randint(0, len(self.dataset))]
        return self.plot(sample, show_titles, suptitle)


class PairedConcatDataset(MinervaConcatDataset):  # type: ignore[type-arg]
    """Adapted form of :class:`~torch.utils.data.ConcatDataset` to handle paired samples."""

    def __init__(
        self,
        dataset1: NonGeoDataset | "PairedConcatDataset",
        dataset2: NonGeoDataset | "PairedConcatDataset",
        size: Optional[int] = None,
        max_r: Optional[int] = None,
    ) -> None:
        _datasets = [dataset1, dataset2]
        datasets = []
        for dataset in _datasets:
            if isinstance(dataset, GeoDataset):  # type: ignore[unreachable]
                raise TypeError("Cannot concatenate geo and non-geo datasets!")
            elif isinstance(dataset, PairedConcatDataset):
                datasets.extend(dataset.datasets)
            elif not isinstance(dataset, PairedNonGeoDataset):
                try:
                    assert size is not None
                    assert max_r is not None
                except AssertionError:
                    raise ValueError(
                        "Dataset is not paired. Cannot imply ``size`` and ``max_r``."
                        + "\nIf dataset is unpaired, you must specify ``size`` and ``max_r``."
                    )
                datasets.append(PairedNonGeoDataset(dataset, size, max_r))

            elif isinstance(dataset, PairedNonGeoDataset):
                datasets.append(dataset)
            else:
                raise TypeError(f"Unknown type {type(dataset)} in concatenation.")

        super().__init__(datasets)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Retrieve image and metadata indexed by query.

        Uses :meth:`torch.utils.data.ConcatDataset.__getitem__` to get the pair of samples from
        the concatenated :class:`~datasets.paired.PairedNonGeoDataset`

        Args:
            index (int): Index of the the concatenated datasets to sample.

        Returns:
            tuple[dict[str, ~typing.Any], dict[str, ~typing.Any]]: Sample of data/labels and metadata at that index.
        """
        sample_a, sample_b = super().__getitem__(index)
        assert isinstance(sample_a, dict)
        assert isinstance(sample_b, dict)
        return sample_a, sample_b

    def __or__(self, other: "PairedNonGeoDataset") -> "PairedConcatDataset":  # type: ignore[override]
        """Take the union of a PairedUnionDataset and a :class:`PairedGeoDataset`.

        Args:
            other (PairedNonGeoDataset): Another dataset.

        Returns:
            PairedConcatDataset: A single dataset.

        .. versionadded:: 0.28
        """
        return PairedConcatDataset(self, other)


class SamplePair:
    """Creates a pair of samples from an initial patch with a set distance of each other.

    Attributes:
        size (int): Size of the each of the samples to cut out from the intial patch.
        max_r (int): Distance between the centres of each of the samples in the pair.

    Args:
        size (int): Optional; Size of the each of the samples to cut out from the intial patch.
            Default 64.
        max_r (int): Optional; Distance between the centres of each of the samples in the pair.
            Default 20.

    .. versionadded:: 0.28
    """

    def __init__(self, size: int = 64, max_r: int = 20, season: bool = False) -> None:
        self.max_r = max_r
        self.size = size
        self.season = season

        # Calculate the global max width between samples in a pair, accounting for the max_r.
        self.max_width = int(1.4 * (self.size + self.max_r))

        # Transform to cut samples out at the desired output size.
        self.random_crop = RandomCrop(self.size)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.season:
            max_width, h, w = self._find_max_width(x[0])

            i, j = self._get_random_crop_params(x[0], max_width)

            # Transform to randomly cut out an area to sample the pair of samples from
            # that will ensure that the distance between the centres of the samples is
            # no more than ``max_r`` pixels apart.
            greater_x0 = ft.crop(x[0], i, j, h, w)
            greater_x1 = ft.crop(x[1], i, j, h, w)

            # Now cut out 2 random samples from within that sampling area and return.
            return self.random_crop(greater_x0), self.random_crop(greater_x1)

        else:
            max_width, _, _ = self._find_max_width(x)

            # Transform to randomly cut out an area to sample the pair of samples from
            # that will ensure that the distance between the centres of the samples is
            # no more than ``max_r`` pixels apart.
            crop_to_sampling_area = RandomCrop(max_width)
            sampling_area = crop_to_sampling_area(x)

            # Now cut out 2 random samples from within that sampling area and return.
            return self.random_crop(sampling_area), self.random_crop(sampling_area)

    def _find_max_width(self, x: Tensor) -> tuple[int, int, int]:
        max_width = self.max_width

        w = x.shape[-1]
        h = x.shape[-2]

        # Checks that the ``max_width`` will not exceed the size of this inital patch.
        # If so, set to the maxium width/ height of ``x``.
        if max_width > w:
            max_width = w  # pragma: no cover
        if max_width > h:
            max_width = h  # pragma: no cover

        return max_width, h, w

    @staticmethod
    def _get_random_crop_params(img: Tensor, max_width: int) -> tuple[int, int]:
        i = torch.randint(0, img.shape[-1] - max_width + 1, size=(1,)).item()
        j = torch.randint(0, img.shape[-2] - max_width + 1, size=(1,)).item()

        assert isinstance(i, int)
        assert isinstance(j, int)
        return i, j
