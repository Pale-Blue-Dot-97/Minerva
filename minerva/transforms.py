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
"""Module containing custom transforms to be used with :mod:`torchvision.transforms`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "ClassTransform",
    "PairCreate",
    "Normalise",
    "AutoNorm",
    "DetachedColorJitter",
    "SingleLabel",
    "ToRGB",
    "SelectChannels",
    "MinervaCompose",
    "SwapKeys",
    "SeasonTransform",
    "get_transform",
    "init_auto_norm",
    "make_transformations",
]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import re
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
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
from kornia.augmentation.base import _AugmentationBase
from torch import LongTensor, Tensor
from torchgeo.datasets import BoundingBox, RasterDataset
from torchgeo.samplers import RandomGeoSampler
from torchvision.transforms import (
    ColorJitter,
    ConvertImageDtype,
    Normalize,
    RandomApply,
)
from torchvision.transforms.v2 import functional as ft

from minerva.utils.utils import find_tensor_mode, func_by_str, mask_transform


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
        transformed: LongTensor = mask_transform(mask, self.transform)  # type: ignore[assignment]
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
        norm_img = img / self.norm_value
        assert isinstance(norm_img, Tensor)
        return norm_img


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

        super().__init__(mean, std, inplace)

    def _calc_mean_std(self) -> Tuple[List[float], List[float]]:
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

    def _get_tile_mean_std(self, query: BoundingBox) -> Tuple[List[float], List[float]]:
        hits = self.dataset.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:  # pragma: no cover
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.dataset.bounds}"
            )

        means: List[float]
        stds: List[float]
        if self.dataset.separate_files:
            filename_regex = re.compile(self.dataset.filename_regex, re.VERBOSE)

            band_means = []
            band_stds = []
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
                band_means.append(mean)
                band_stds.append(std)
            means = [np.mean(band) for band in band_means]  # type:ignore[misc]
            stds = [np.mean(band) for band in band_stds]  # type:ignore[misc]
        else:
            means, stds = self._get_image_mean_std(filepaths, self.dataset.band_indexes)

        return means, stds

    def _get_image_mean_std(
        self,
        filepaths: List[str],
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tuple[List[float], List[float]]:
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

    @staticmethod
    def _extract_meta(data, band_index):
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

            rgb_img = img[:3]
            assert isinstance(rgb_img, Tensor)
            return rgb_img


class SelectChannels:
    """Transform to select which channels to keep by passing a list of indices

    Attributes:
        channels (list[int]): Channel indices to keep.

    Args:
        channels (list[int]): Channel indices to keep.
    """

    def __init__(self, channels: List[int]) -> None:
        self.channels = channels

    def __call__(self, img: Tensor) -> Tensor:
        return self.forward(img)

    def forward(self, img: Tensor) -> Tensor:
        """Select the desired channels from the input image and return.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Selected channels of the input image.
        """
        return img[self.channels]


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

    Attributes:
        transforms (list[~typing.Callable[..., ~typing.Any]] | ~typing.Callable[..., ~typing.Any]):
            List of composed transforms.
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
        transforms: Union[
            List[Callable[..., Any]],
            Callable[..., Any],
            Dict[str, Union[List[Callable[..., Any]], Callable[..., Any]]],
        ],
    ) -> None:
        self.transforms: Union[
            List[Callable[..., Any]], Dict[str, List[Callable[..., Any]]]
        ]

        if isinstance(transforms, Sequence):
            self.transforms = list(transforms)
        elif callable(transforms):
            self.transforms = [transforms]
        elif isinstance(transforms, dict):
            self.transforms = transforms  # type: ignore[assignment]
            assert isinstance(self.transforms, dict)
            for key in transforms.keys():
                if callable(transforms[key]):
                    _transform = transforms[key]
                    assert callable(_transform)
                    self.transforms[key] = [_transform]
        else:
            raise TypeError(
                f"`transforms` has type {type(transforms)}, not callable or sequence of callables"
            )

    @overload
    def __call__(self, sample: Tensor) -> Tensor: ...  # pragma: no cover

    @overload
    def __call__(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]: ...  # pragma: no cover

    def __call__(
        self, sample: Union[Tensor, Dict[str, Any]]
    ) -> Union[Tensor, Dict[str, Any]]:
        if isinstance(sample, Tensor):
            assert not isinstance(self.transforms, dict)
            return self._transform_input(sample, self.transforms)
        elif isinstance(sample, dict):
            assert isinstance(self.transforms, dict)
            for key in self.transforms.keys():
                # Special case for "both". Indicates that these transforms should be applied
                # to both types of data in the dataset.
                # Assumes the keys must be "image" and "mask"
                if key == "both":
                    # Transform images with new random states (if applicable).
                    sample["image"] = self._transform_input(sample["image"], self.transforms["both"])

                    # We'll have to convert the masks to float for these transforms to work
                    # so need to store the current dtype to cast back to after.
                    prior_dtype = sample["mask"].dtype

                    # Transform masks reapplying the same random states.
                    sample["mask"] = self._transform_input(
                        sample["mask"].to(dtype=torch.float),
                        self.transforms["both"],
                        reapply=True,
                    ).to(dtype=prior_dtype)
                else:
                    sample[key] = self._transform_input(sample[key], self.transforms[key])
            return sample
        else:
            raise TypeError(f"Sample is {type(sample)=}, not Tensor or dict!")

    @staticmethod
    def _transform_input(img: Tensor, transforms: List[Callable[..., Any]], reapply: bool = False) -> Tensor:
        if isinstance(transforms, Sequence):
            for t in transforms:
                if isinstance(t, _AugmentationBase) and reapply:
                    img = t(img, params=t._params)

                else:
                    img = t(img)

        else:
            raise TypeError(
                f"`transforms` has type {type(transforms)}, not sequence of callables"
            )

        return img

    def _add(
        self,
        new_transform: Union[
            "MinervaCompose",
            Sequence[Callable[..., Any]],
            Callable[..., Any],
            Dict[str, Union[Sequence[Callable[..., Any]], Callable[..., Any]]],
        ],
    ) -> Union[Dict[str, List[Callable[..., Any]]], List[Callable[..., Any]]]:
        def add_transforms(
            _new_transform: Union[Sequence[Callable[..., Any]], Callable[..., Any]],
            old_transform: List[Callable[..., Any]],
        ) -> List[Callable[..., Any]]:
            if isinstance(_new_transform, Sequence):
                old_transform.extend(_new_transform)
                return old_transform
            elif callable(_new_transform):
                old_transform.append(_new_transform)
                return old_transform
            else:
                raise TypeError(
                    f"`new_transform` has type {type(new_transform)}, not callable or sequence of callables"
                )

        _transforms = deepcopy(self.transforms)

        if isinstance(new_transform, MinervaCompose):
            return self._add(new_transform.transforms)  # type: ignore[arg-type]

        if isinstance(new_transform, dict) and isinstance(_transforms, dict):
            for key in new_transform.keys():
                if key not in _transforms:
                    _transforms[key] = new_transform[key]  # type: ignore[assignment]
                else:
                    _transforms[key] = add_transforms(
                        new_transform[key], _transforms[key]
                    )
            return _transforms

        elif (
            isinstance(new_transform, Sequence) or callable(new_transform)
        ) and not isinstance(_transforms, dict):
            return add_transforms(new_transform, _transforms)

        else:
            raise TypeError(
                f"Cannot add together transforms of {type(new_transform)=} to the existing"
                + f" type of {type(self.transforms)=}"
            )

    def __add__(
        self,
        new_transform: Union[
            "MinervaCompose",
            Sequence[Callable[..., Any]],
            Callable[..., Any],
            Dict[str, Union[Sequence[Callable[..., Any]], Callable[..., Any]]],
        ],
    ) -> "MinervaCompose":
        new_compose = deepcopy(self)
        new_compose.transforms = self._add(new_transform)
        return new_compose

    def __iadd__(
        self,
        new_transform: Union[
            "MinervaCompose",
            Sequence[Callable[..., Any]],
            Callable[..., Any],
            Dict[str, Union[Sequence[Callable[..., Any]], Callable[..., Any]]],
        ],
    ) -> "MinervaCompose":
        self.transforms = self._add(new_transform)
        return self

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        if isinstance(self.transforms, dict):
            for key in self.transforms.keys():
                format_string += "\n"
                format_string += f"    {key}:"
                for t in self.transforms[key]:
                    format_string += "\n"
                    format_string += "        {0}".format(t)

        elif isinstance(self.transforms, list):
            if len(self.transforms) > 1:
                for t in self.transforms:
                    format_string += "\n"
                    format_string += "    {0}".format(t)

            else:
                format_string += "{0})".format(self.transforms[0])
                return format_string

        else:
            raise TypeError(
                f"`transforms` has type {type(self.transforms)}, not sequence of callables or dictionary"
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


class SeasonTransform:
    """Configure what seasons from a patch are parsed through to the model.

    Adapted from source: https://github.com/zhu-xlab/SSL4EO-S12/tree/main

    Args:
        season (str): How to handle what seasons to return:
        * ``pair``: Randomly pick 2 seasons to return that will form a pair.
        * ``random``: Randomly pick a single season to return.
    """

    def __init__(self, season: str = "random") -> None:
        self.season = season

    def __call__(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:

        if self.season == "pair":
            season1 = np.random.choice([0, 1, 2, 3])
            season2 = np.random.choice([0, 1, 2, 3])

            x1 = x[season1]
            x2 = x[season2]
            return x1, x2

        elif self.season == "random":
            season1 = np.random.choice([0, 1, 2, 3])

        else:
            raise ValueError(
                f"The value for season, {self.season}, is not a valid option! Choose from ``random`` or ``pair``"
            )

        image = x[season1]

        return image


class ConvertDtypeFromStr(ConvertImageDtype):
    """Wrapper for :class:~`torchvision.transforms.ConvertImageDtype` that accepts str.

    Args:
        dtype (str): A tensor type as :class:`str`.
    """

    def __init__(self, dtype: str) -> None:
        super().__init__(getattr(torch, dtype))


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def _construct_random_transforms(random_params: Dict[str, Any]) -> Any:
    p = random_params.pop("p", 0.5)

    random_transforms = []
    for ran_name in random_params:
        random_transforms.append(get_transform(ran_name, random_params[ran_name]))

    return RandomApply(random_transforms, p=p)


def init_auto_norm(
    dataset: RasterDataset, params: Dict[str, Any] = {}
) -> RasterDataset:
    """Uses :class:~`minerva.transforms.AutoNorm` to automatically find the mean and standard deviation of `dataset`
    to create a normalisation transform that is then added to the existing transforms of `dataset`.

    Args:
        dataset (RasterDataset): Dataset to find and apply the normalisation conditions to.
        params (Dict[str, Any]): Parameters for :class:~`minerva.transforms.AutoNorm`.

    Returns:
        RasterDataset: `dataset` with an additional :class:~`minerva.transforms.AutoNorm` transform
        added to it's :attr:~`torchgeo.datasets.RasterDataset.transforms` attribute.
    """
    # Creates the AutoNorm transform by sampling `dataset` for its mean and standard deviation stats.
    auto_norm = AutoNorm(dataset, **params)

    if dataset.transforms is None:
        dataset.transforms = MinervaCompose(auto_norm)
    else:
        # If the existing transforms are already `MinervaCompose`, we can just add the AutoNorm transform on.
        if isinstance(dataset.transforms, MinervaCompose):
            if isinstance(dataset.transforms.transforms, dict):
                dataset.transforms += {"image": auto_norm}
            else:
                dataset.transforms += auto_norm

        # If existing transforms are a callable, place in a list with AutoNorm and make in `MinervaCompose`.
        elif callable(dataset.transforms):
            dataset.transforms = MinervaCompose([dataset.transforms, auto_norm])
        else:
            raise TypeError(
                f"The type of datset.transforms, {type(dataset.transforms)}, is not supported"
            )

    return dataset


def get_transform(name: str, transform_params: Dict[str, Any]) -> Callable[..., Any]:
    """Creates a transform object based on config parameters.

    Args:
        name (str): Name of transform object to import e.g :class:`~torchvision.transforms.RandomResizedCrop`.
        transform_params (dict[str, ~typing.Any]): Arguements to construct transform with.
            Should also include ``"module"`` key defining the import path to the transform object.

    Returns:
        Initialised transform object specified by config parameters.

    .. note::
        If ``transform_params`` contains no ``"module"`` key, it defaults to ``torchvision.transforms``.

    Example:
        >>> name = "RandomResizedCrop"
        >>> params = {"module": "torchvision.transforms", "size": 128}
        >>> transform = get_transform(name, params)

    Raises:
        TypeError: If created transform object is itself not :class:`~typing.Callable`.
    """
    params = transform_params.copy()
    module = params.pop("module", "torchvision.transforms")

    # Gets the transform requested by config parameters.
    _transform: Callable[..., Any] = func_by_str(module, name)

    transform: Callable[..., Any] = _transform(**params)
    if callable(transform):
        return transform
    else:
        raise TypeError(f"Transform has type {type(transform)}, not a callable!")


def make_transformations(
    transform_params: Union[Dict[str, Any], Literal[False]]
) -> Optional[MinervaCompose]:
    """Constructs a transform or series of transforms based on parameters provided.

    Args:
        transform_params (dict[str, ~typing.Any] | ~typing.Literal[False]): Parameters defining transforms desired.
            The name of each transform should be the key, while the kwargs for the transform should
            be the value of that key as a dict.

    Example:
        >>> transform_params = {
        >>>    "CenterCrop": {"module": "torchvision.transforms", "size": 128},
        >>>     "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7}
        >>> }
        >>> transforms = make_transformations(transform_params)

    Returns:
        If no parameters are parsed, None is returned.
        If only one transform is defined by the parameters, returns a Transforms object.
        If multiple transforms are defined, a Compose object of Transform objects is returned.
    """

    def construct(type_transform_params: Dict[str, Any]) -> List[Callable[..., Any]]:
        type_transformations = []

        # Get each transform.
        for _name in type_transform_params:
            if _name == "RandomApply":
                random_params = type_transform_params[_name].copy()
                type_transformations.append(_construct_random_transforms(random_params))

            # AutoNorm needs to be handled separately.
            elif _name == "AutoNorm":
                continue

            else:
                type_transformations.append(
                    get_transform(_name, type_transform_params[_name])
                )

        return type_transformations

    # If no transforms are specified, return None.
    if not transform_params:
        return None
    if all(transform_params.values()) is None:
        return None

    transformations: Dict[str, Any] = {}

    for name in transform_params:
        if name in ("image", "mask", "label", "both"):
            if not transform_params[name]:
                pass
            else:
                transformations[name] = construct(transform_params[name])
        else:
            return MinervaCompose(construct(transform_params))

    return MinervaCompose(transformations)
