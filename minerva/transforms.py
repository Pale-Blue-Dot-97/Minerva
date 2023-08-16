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
"""Module containing custom transforms to be used with :mod:`torchvision.transforms`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "ClassTransform",
    "PairCreate",
    "Normalise",
    "DetachedColorJitter",
    "SingleLabel",
    "ToRGB",
    "MinervaCompose",
    "SwapKeys",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import functools
import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import rasterio
import torch
from torch import LongTensor, Tensor
from torchgeo.datasets import BoundingBox, RasterDataset
from torchgeo.samplers import RandomGeoSampler
from torchvision.transforms import ColorJitter, Normalize
from torchvision.transforms import functional_tensor as ft

from minerva.utils.utils import find_tensor_mode, mask_transform


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class ClassTransform:
    """Transform to be applied to a mask to convert from one labelling schema to another.

    Attributes:
        transform (dict[int, int]): Mapping from one labelling schema to another.

    Args:
        transform (dict[int, int]): Mapping from one labelling schema to another.
    """

    def __init__(self, transform: Dict[int, int]) -> None:
        self.transform = transform

    def __call__(self, mask: LongTensor) -> LongTensor:
        return self.forward(mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"

    def forward(self, mask: LongTensor) -> LongTensor:
        """Transforms the given mask from the original label schema to the new.

        Args:
            mask (~torch.LongTensor): Mask in the original label schema.

        Returns:
            ~torch.LongTensor: Mask transformed into new label schema.
        """
        transformed: LongTensor = mask_transform(mask, self.transform)
        return transformed


class PairCreate:
    """Transform that takes a sample and returns a pair of the same sample."""

    def __init__(self) -> None:
        pass

    def __call__(self, sample: Any) -> Tuple[Any, Any]:
        return self.forward(sample)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def forward(sample: Any) -> Tuple[Any, Any]:
        """Takes a sample and returns it and a copy as a :class:`tuple` pair.

        Args:
            sample (~typing.Any): Sample to duplicate.

        Returns:
            tuple[~typing.Any, ~typing.Any]: :class:`tuple` of two copies of the sample.
        """
        return sample, sample


class Normalise:
    """Transform that normalises an image tensor based on the bit size.

    Attributes:
        norm_value (int): Value to normalise image with.

    Args:
        norm_value (int): Value to normalise image with.
    """

    def __init__(self, norm_value: int) -> None:
        self.norm_value = norm_value

    def __call__(self, img: Tensor) -> Tensor:
        return self.forward(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm_value={self.norm_value})"

    def forward(self, img: Tensor) -> Tensor:
        """Normalises inputted image using ``norm_value``.

        Args:
            img (~torch.Tensor): Image tensor to be normalised. Should have a bit size
                that relates to ``norm_value``.

        Returns:
            ~torch.Tensor: Input image tensor normalised by ``norm_value``.
        """
        return img / self.norm_value


class AutoNorm(Normalize):
    """Transform that will automatically calculate the mean and standard deviation of the dataset
    to normalise the data with.

    Uses :class:`torchvision.transforms.Normalize` for the normalisation.

    Attributes:
        dataset (RasterDataset): Dataset to calculate the mean and standard deviation of.
        sampler (RandomGeoSampler): Sampler used to create valid queries for the dataset to find data files.

    Args:
        dataset (RasterDataset): Dataset to calculate the mean and standard deviation of.
        length (int): Optional; Number of samples from the dataset to calculate the mean and standard deviation of.
        roi (BoundingBox): Optional; Region of interest for sampler to sample from.
        inplace (bool): Optional; Performs the normalisation transform inplace on the tensor. Default False.

    .. versionadded:: 0.26
    """

    def __init__(
        self,
        dataset: RasterDataset,
        length: int = 128,
        roi: Optional[BoundingBox] = None,
        inplace=False,
    ):
        self.dataset = dataset
        self.sampler = RandomGeoSampler(dataset, 32, length, roi)

        mean, std = self._calc_mean_std()
        print(mean, std)

        super().__init__(mean, std, inplace)

    def _calc_mean_std(self):
        per_img_means = []
        per_img_stds = []
        for query in self.sampler:
            mean, std = self._get_tile_mean_std(query)
            per_img_means.append(mean)
            per_img_stds.append(std)

        per_band_means = list(zip(*per_img_means))
        per_band_stds = list(zip(*per_img_stds))

        per_band_mean = [np.mean(band) for band in per_band_means]
        per_band_std = [np.mean(band) for band in per_band_stds]

        return per_band_mean, per_band_std

    def _get_tile_mean_std(self, query: BoundingBox):
        hits = self.dataset.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.dataset.bounds}"
            )

        means: List[List[float]] = []
        stds: List[List[float]] = []
        if self.dataset.separate_files:
            filename_regex = re.compile(self.dataset.filename_regex, re.VERBOSE)
            for band in self.dataset.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = Path(filepath).name
                    directory = Path(filepath).parent
                    match = re.match(filename_regex, filename)
                    if match:
                        if "band" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                    filepath = str(directory / filename)
                    band_filepaths.append(filepath)
                mean, std = self._get_image_mean_std(band_filepaths)
                means.append(mean)
                stds.append(std)
        else:
            means, stds = self._get_image_mean_std(filepaths, self.dataset.band_indexes)

        return means, stds

    def _get_image_mean_std(
        self,
        filepaths: List[str],
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        stats = [self._get_meta_mean_std(fp, band_indexes) for fp in filepaths]

        means = list(np.mean([stat[0] for stat in stats], axis=0))
        stds = list(np.mean([stat[1] for stat in stats], axis=0))

        return means, stds

    def _get_meta_mean_std(
        self, filepath, band_indexes: Optional[Sequence[int]] = None
    ) -> Tuple[List[float], List[float]]:
        # Open the Tiff file and get the statistics from the meta (min, max, mean, std).
        means = []
        stds = []
        data = rasterio.open(filepath)
        if band_indexes:
            for band in band_indexes:
                mean, std = self._extract_meta(data, band)
                means.append(mean)
                stds.append(std)
        else:
            mean, std = self._extract_meta(data, 1)
            means, stds = [mean], [std]
        data.close()

        return means, stds

    def _extract_meta(self, data, band_index):
        stats = data.statistics(band_index)
        mean, std = stats.mean, stats.std

        return mean, std


class DetachedColorJitter(ColorJitter):
    """Sends RGB channels of multi-spectral images to be transformed by
    :class:`~torchvision.transforms.ColorJitter`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, img: Tensor) -> Tensor:
        """Detaches RGB channels of input image to be sent to :class:`~torchvision.transforms.ColorJitter`.

        All other channels bypass :class:`~torchvision.transforms.ColorJitter` and are
        concatenated onto the colour jittered RGB channels.

        Args:
            img (~torch.Tensor): Input image.

        Raises:
            ValueError: If number of channels of input ``img`` is 2.

        Returns:
            ~torch.Tensor: Color jittered image.
        """
        channels = ft.get_image_num_channels(img)

        jitter_img: Tensor
        if channels > 3:
            rgb_jitter = super().forward(img[:3])
            jitter_img = torch.cat((rgb_jitter, img[3:]), 0)  # type: ignore[attr-defined]

        elif channels in (1, 3):
            jitter_img = super().forward(img)

        else:
            raise ValueError(f"{channels} channel images are not supported!")

        return jitter_img

    def __call__(self, img: Tensor) -> Tensor:
        return self.forward(img)

    def __repr__(self) -> Any:
        return super().__repr__()


class ToRGB:
    """Reduces the number of channels down to RGB.

    Attributes:
        channels (tuple[int, int, int]): Optional; Tuple defining which channels in expected input images
            contain the RGB bands. If ``None``, it is assumed that the RGB bands are in the first 3 channels.

    Args:
        channels (tuple[int, int, int]): Optional; Tuple defining which channels in expected input images
            contain the RGB bands. If ``None``, it is assumed that the RGB bands are in the first 3 channels.

    .. versionadded:: 0.22

    """

    def __init__(self, channels: Optional[Tuple[int, int, int]] = None) -> None:
        self.channels = channels

    def __call__(self, img: Tensor) -> Tensor:
        return self.forward(img)

    def __repr__(self) -> str:
        if self.channels:
            return f"{self.__class__.__name__}(channels --> [{self.channels}])"
        else:
            return f"{self.__class__.__name__}(channels --> [0:3])"

    def forward(self, img: Tensor) -> Tensor:
        """Performs a forward pass of the transform, returning an RGB image.

        Args:
            img (~torch.Tensor): Image to convert to RGB.

        Returns:
            ~torch.Tensor: Image of only the RGB channels of ``img``.

        Raises:
            ValueError: If ``img`` has less channels than specified in :attr:`~ToRGB.channels`.
            ValueError: If ``img`` has less than 3 channels and :attr:`~ToRGB.channels` is ``None``.
        """
        # If a tuple defining the RGB channels was provided, select and concat together.
        if self.channels:
            if len(img) < len(self.channels):
                raise ValueError("Image has less channels that trying to reduce to!")

            return torch.stack([img[channel] for channel in self.channels])

        # If no channels were provided, assume that that the first 3 channels are the RGB channels.
        else:
            if len(img) < 3:
                raise ValueError("Image has less than 3 channels! Cannot be RGB!")

            return img[:3]


class SingleLabel:
    """Reduces a mask to a single label using transform mode provided.

    Attributes:
        mode (str): Mode of operation.

    Args:
        mode (str): Mode of operation. Currently only supports ``"modal"``.

    .. versionadded:: 0.22

    """

    def __init__(self, mode: str = "modal") -> None:
        self.mode = mode

    def __call__(self, mask: LongTensor) -> LongTensor:
        return self.forward(mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode})"

    def forward(self, mask: LongTensor) -> LongTensor:
        """Forward pass of the transform, reducing the input mask to a single label.

        Args:
            mask (~torch.LongTensor): Input mask to reduce to a single label.

        Raises:
            NotImplementedError: If :attr:`~SingleLabel.mode` is not ``"modal"``.

        Returns:
            ~torch.LongTensor: The single label as a 0D, 1-element tensor.
        """
        if self.mode == "modal":
            return LongTensor([find_tensor_mode(mask)])
        else:
            raise NotImplementedError(
                f"{self.mode} is not a recognised operating mode!"
            )


class MinervaCompose:
    """Adaption of :class:`torchvision.transforms.Compose`. Composes several transforms together.

    Designed to work with both :class:`~torch.Tensor` and :mod:`torchgeo` sample :class:`dict`.

    This transform does not support torchscript.

    Args:
        transforms (~typing.Sequence[~typing.Callable[..., ~typing.Any]] | ~typing.Callable[..., ~typing.Any]):
            List of transforms to compose.
        key (str): Optional; For use with :mod:`torchgeo` samples and must be assigned a value if using.
            The key of the data type in the sample dict to transform.

    Example:
        >>> transforms.MinervaCompose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])
    """

    def __init__(
        self,
        transforms: Union[Sequence[Callable[..., Any]], Callable[..., Any]],
        key: Optional[str] = None,
    ) -> None:
        self.transforms = transforms
        self.key = key

    @overload
    def __call__(self, sample: Tensor) -> Tensor:
        ...  # pragma: no cover

    @overload
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        ...  # pragma: no cover

    def __call__(
        self, sample: Union[Tensor, Dict[str, Any]]
    ) -> Union[Tensor, Dict[str, Any]]:
        if isinstance(sample, Tensor):
            return self._transform_input(sample)
        elif isinstance(sample, dict):
            assert self.key is not None
            sample[self.key] = self._transform_input(sample[self.key])
            return sample
        else:
            raise TypeError(f"Sample is {type(sample)=}, not Tensor or dict!")

    def _transform_input(self, img: Tensor) -> Tensor:
        if isinstance(self.transforms, Sequence):
            for t in self.transforms:
                img = t(img)
        elif callable(self.transforms):
            img = self.transforms(img)

        else:
            raise TypeError(
                f"`transforms` has type {type(self.transforms)}, not callable"
            )

        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("

        if isinstance(self.transforms, Sequence):
            for t in self.transforms:
                format_string += "\n"
                format_string += "    {0}".format(t)

        elif callable(self.transforms):
            format_string += "{0})".format(self.transforms)
            return format_string

        else:
            raise TypeError(
                f"`transforms` has type {type(self.transforms)}, not callable"
            )

        format_string += "\n)"

        return format_string


class SwapKeys:
    """Transform to set one key in a :mod:`torchgeo` sample :class:`dict` to another.

    Useful for testing autoencoders to predict their input.

    Attributes:
        from_key (str): Key for the value to set to ``to_key``.
        to_key (str): Key to set the value from ``from_key`` to.

    Args:
        from_key (str): Key for the value to set to ``to_key``.
        to_key (str): Key to set the value from ``from_key`` to.

    .. versionadded:: 0.22
    """

    def __init__(self, from_key: str, to_key: str) -> None:
        self.from_key = from_key
        self.to_key = to_key

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward(sample)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.from_key} -> {self.to_key})"

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Sets the ``to_key`` of ``sample`` to the ``from_key`` and returns.

        Args:
            sample (dict[str, ~typing.Any]): Sample dict from :mod:`torchgeo` containing ``from_key``.

        Returns:
            dict[str, ~typing.Any]: Sample with ``to_key`` set to the value of ``from_key``.
        """
        sample[self.to_key] = sample[self.from_key]
        return sample
