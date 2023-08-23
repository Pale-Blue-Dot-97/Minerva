# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

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
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "PairedDataset",
    "PairedUnionDataset",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, overload

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
    UnionDataset,
)
from torchgeo.datasets.utils import BoundingBox, concat_samples, merge_samples

from minerva.utils import utils

from .utils import get_random_sample


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class PairedDataset(RasterDataset):
    """Custom dataset to act as a wrapper to other datasets to handle paired sampling.

    Attributes:
        dataset (~torchgeo.datasets.RasterDataset): Wrapped dataset to sampled from.

    Args:
        dataset_cls (~typing.Callable[..., ~torchgeo.datasets.GeoDataset]): Constructor for a
            :class:`~torchgeo.datasets.RasterDataset` to be wrapped for paired sampling.
    """

    def __new__(  # type: ignore[misc]
        cls,
        dataset: Union[Callable[..., GeoDataset], GeoDataset],
        *args,
        **kwargs,
    ) -> Union["PairedDataset", "PairedUnionDataset"]:
        if isinstance(dataset, UnionDataset):
            return PairedUnionDataset(
                dataset.datasets[0], dataset.datasets[1], *args, **kwargs
            )
        else:
            return super(PairedDataset, cls).__new__(cls)

    @overload
    def __init__(self, dataset: Callable[..., GeoDataset], *args, **kwargs) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(self, dataset: GeoDataset, *args, **kwargs) -> None:
        ...  # pragma: no cover

    def __init__(
        self,
        dataset: Union[Callable[..., GeoDataset], GeoDataset],
        *args,
        **kwargs,
    ) -> None:
        if isinstance(dataset, GeoDataset):
            self.dataset = dataset
            self._res = dataset.res
            self._crs = dataset.crs

        elif callable(dataset):
            super_sig = inspect.signature(RasterDataset.__init__).parameters.values()
            super_kwargs = {
                key.name: kwargs[key.name] for key in super_sig if key.name in kwargs
            }

            if issubclass(dataset, RasterDataset):  # type: ignore[arg-type]
                # This is very sketchy but an unavoidable workaround due to TorchGeo's behaviour.
                RasterDataset.filename_glob = dataset.filename_glob  # type: ignore[attr-defined]
                RasterDataset.filename_regex = dataset.filename_regex  # type: ignore[attr-defined]

            super().__init__(*args, **super_kwargs)
            self.dataset = dataset(*args, **kwargs)

        else:
            raise ValueError(
                f"``dataset`` is of unsupported type {type(dataset)} not GeoDataset"
            )

    def __getitem__(  # type: ignore[override]
        self, queries: Tuple[BoundingBox, BoundingBox]
    ) -> Tuple[Dict[str, Any], ...]:
        return self.dataset.__getitem__(queries[0]), self.dataset.__getitem__(
            queries[1]
        )

    def __and__(self, other: "PairedDataset") -> IntersectionDataset:  # type: ignore[override]
        """Take the intersection of two :class:`PairedDataset`.

        Args:
            other (PairedDataset): Another dataset.

        Returns:
            IntersectionDataset: A single dataset.

        Raises:
            ValueError: If other is not a :class:`PairedDataset`

        .. versionadded:: 0.24
        """
        if not isinstance(other, PairedDataset):
            raise ValueError(
                f"Intersecting a dataset of {type(other)} and a PairedDataset is not supported!"
            )

        return IntersectionDataset(
            self, other, collate_fn=utils.pair_collate(concat_samples)
        )

    def __or__(self, other: "PairedDataset") -> "PairedUnionDataset":  # type: ignore[override]
        """Take the union of two :class:`PairedDataset`.

        Args:
            other (PairedDataset): Another dataset.

        Returns:
            PairedUnionDataset: A single dataset.

        .. versionadded:: 0.24
        """
        return PairedUnionDataset(self, other)

    def __getattr__(self, item):
        if item in self.dataset.__dict__:
            return getattr(self.dataset, item)  # pragma: no cover
        elif item in self.__dict__:
            return getattr(self, item)
        else:
            raise AttributeError

    def __repr__(self) -> Any:
        return self.dataset.__repr__()

    @staticmethod
    def plot(
        sample: Dict[str, Any],
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
        size: Union[Tuple[int, int], int],
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

        Do not use with :class:`PairedDataset` as this will essentially account for paired sampling twice
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
        super().__init__(dataset1, dataset2, collate_fn, transforms)

        new_datasets = []
        for _dataset in self.datasets:
            if isinstance(_dataset, PairedDataset):
                new_datasets.append(_dataset.dataset)
            elif isinstance(_dataset, PairedUnionDataset):
                new_datasets.append(_dataset.datasets[0] | _dataset.datasets[1])
            else:
                new_datasets.append(_dataset)

        self.datasets = new_datasets

    def __getitem__(  # type: ignore[override]
        self, query: Tuple[BoundingBox, BoundingBox]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    def __or__(self, other: "PairedDataset") -> "PairedUnionDataset":  # type: ignore[override]
        """Take the union of a PairedUnionDataset and a :class:`PairedDataset`.

        Args:
            other (PairedDataset): Another dataset.

        Returns:
            PairedUnionDataset: A single dataset.

        .. versionadded:: 0.24
        """
        return PairedUnionDataset(self, other)
