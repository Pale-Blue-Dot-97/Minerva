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
"""Module to handle various IO from `dataloaders` and to models."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "sup_tg",
    "ssl_pair_tg",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.testing import assert_array_equal, assert_raises
from torch import LongTensor, Tensor
from torchgeo.datasets.utils import BoundingBox

from minerva.models import MinervaModel
from minerva.utils.utils import check_substrings_in_string, mask_to_ohe


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def sup_tg(
    batch: Dict[Any, Any],
    model: MinervaModel,
    device: torch.device,  # type: ignore[name-defined]
    train: bool,
    **kwargs,
) -> Tuple[
    Tensor,
    Union[Tensor, Tuple[Tensor, ...]],
    Tensor,
    Optional[Union[Sequence[str], Sequence[BoundingBox]]],
]:
    """Provides IO functionality for a supervised model using :mod:`torchgeo` datasets.

    Args:
        batch (dict[~typing.Any, ~typing.Any]): Batch of data in a :class:`dict`.
            Must have ``"image"``, ``"mask"``/ ``"label"`` and ``"bbox"`` / ``"id"`` keys.
        model (MinervaModel): Model being fitted.
        device (~torch.device): `torch` device object to send data to (e.g. CUDA device).
        train (bool): True to run a step of the model in training mode. False for eval mode.

    Kwargs:
        mix_precision (bool): Use mixed-precision. Will set the floating tensors to 16-bit
            rather than the default 32-bit.
        target_key (str): Should be either ``"mask"`` or ``"label"``.

    Returns:
        tuple[
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~typing.Sequence[~torchgeo.datasets.utils.BoundingBox] | None
        ]:
        The ``loss``, the model output ``z``, the ground truth ``y`` supplied and the bounding boxes
        of the input images supplied.
    """
    float_dtype = _determine_float_dtype(device, kwargs.get("mix_precision", False))
    target_key = kwargs.get("target_key", "mask")

    model_type = kwargs.get("model_type", "")
    multilabel = True if check_substrings_in_string(model_type, "multilabel") else False

    # Extracts the x and y batches from the dict.
    x: Tensor = batch["image"]
    y: Tensor = batch[target_key]

    # Check that none of the data is NaN or infinity.
    if kwargs.get("validate_variables", False):
        assert not x.isnan().any()
        assert not x.isinf().any()
        assert not y.isnan().any()
        assert not y.isinf().any()

    # Re-arranges the x and y batches.
    x = x.to(float_dtype)  # type: ignore[attr-defined]

    # Squeeze out axis 1 if only 1 element wide.
    if target_key == "mask":
        y = y.squeeze()

    if isinstance(y, Tensor):
        y = y.detach().cpu()
    y = y.to(dtype=torch.float if multilabel else torch.long)  # type: ignore[attr-defined]

    # Transfer to GPU.
    x = x.to(device)
    y = y.to(device)

    # Runs a step of the epoch.
    loss, z = model.step(x, y, train=train)

    # Get the indices of the batch. Either bounding boxes or filenames.
    index: Optional[Union[Sequence[str], Sequence[BoundingBox]]]
    if "bbox" in batch:
        index = batch["bbox"]
    elif "id" in batch:
        index = batch["id"]
    else:
        index = None

    return loss, z, y, index


def autoencoder_io(
    batch: Dict[Any, Any],
    model: MinervaModel,
    device: torch.device,  # type: ignore[name-defined]
    train: bool,
    **kwargs,
) -> Tuple[
    Tensor,
    Union[Tensor, Tuple[Tensor, ...]],
    Tensor,
    Optional[Union[Sequence[str], Sequence[BoundingBox]]],
]:
    """Provides IO functionality for an autoencoder using :mod:`torchgeo` datasets by only using the same data
    for input and ground truth.

    Args:
        batch (dict[~typing.Any, ~typing.Any]): Batch of data in a :class:`dict`.
            Must have ``"image"``, ``"mask"``/ ``"label"`` and ``"bbox"`` / ``"id"`` keys.
        model (MinervaModel): Model being fitted.
        device (~torch.device): `torch` device object to send data to (e.g. CUDA device).
        train (bool): True to run a step of the model in training mode. False for eval mode.

    Keyword args:
        autoencoder_data_key (str): Key of the data type in the sample dict to use for both input and ground truth.
            Must be either ``"mask"`` or ``"image"``.
        mix_precision (bool): Use mixed-precision. Will set the floating tensors to 16-bit
            rather than the default 32-bit.

    Returns:
        tuple[
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~typing.Sequence[~torchgeo.datasets.utils.BoundingBox] | None
        ]:
        The ``loss``, the model output ``z``, the ground truth ``y`` supplied and the bounding boxes
        of the input images supplied.

    Raises:
        ValueError: If the value given for ``key`` is not ``"mask"`` or ``"image"``.

    .. versionadded:: 0.23
    """
    x: Tensor
    y: Tensor
    key = kwargs.get("autoencoder_data_key")
    float_dtype = _determine_float_dtype(device, kwargs.get("mix_precision", False))

    # Extracts the images and masks from the batch sample dict.
    images: Tensor = batch["image"]
    masks: LongTensor = batch["mask"]

    # Check that none of the data is NaN or infinity.
    if kwargs.get("validate_variables", False):
        assert not images.isnan().any()
        assert not images.isinf().any()
        assert not masks.isnan().any()
        assert not masks.isinf().any()

    if key == "mask":
        # Squeeze out axis 1 if only 1 element wide.
        if masks.shape[1] == 1:
            _masks = torch.tensor(
                np.squeeze(masks.detach().cpu().numpy(), axis=1), dtype=torch.long
            )
            assert isinstance(_masks, LongTensor)
            masks = _masks

        input_masks: Tensor = torch.stack(
            tuple([mask_to_ohe(mask, kwargs.get("n_classes", None)) for mask in masks])
        )
        output_masks: LongTensor = masks

        if isinstance(input_masks, Tensor):
            input_masks = input_masks.detach().cpu().numpy()

        if isinstance(output_masks, Tensor):
            output_masks = output_masks.detach().cpu().numpy()

        # Transfer to GPU and cast to correct dtypes.
        x = torch.tensor(input_masks, dtype=float_dtype, device=device)
        y = torch.tensor(output_masks, dtype=torch.long, device=device)

    elif key == "image":
        # Extract the images from the batch, set to float, transfer to GPU and make x and y.
        x = images.to(dtype=float_dtype, device=device)
        y = images.to(dtype=float_dtype, device=device)

    else:
        raise ValueError(
            f"The value of {key=} is not understood. Must be either 'mask' or 'image'"
        )

    # Runs a step of the epoch.
    loss, z = model.step(x, y, train=train)

    # Get the indices of the batch. Either bounding boxes or filenames.
    index: Optional[Union[Sequence[str], Sequence[BoundingBox]]]
    if "bbox" in batch:
        index = batch["bbox"]
    elif "id" in batch:
        index = batch["id"]
    else:
        index = None

    return loss, z, y, index


def ssl_pair_tg(
    batch: Tuple[Dict[str, Any], Dict[str, Any]],
    model: MinervaModel,
    device: torch.device,  # type: ignore[name-defined]
    train: bool,
    **kwargs,
) -> Tuple[
    Tensor,
    Union[Tensor, Tuple[Tensor, ...]],
    None,
    Optional[Union[Sequence[BoundingBox], Sequence[int]]],
]:
    """Provides IO functionality for a self-supervised Siamese model using :mod:`torchgeo` datasets.

    Args:
        batch (tuple[dict[str, ~typing.Any], dict[str, ~typing.Any]]): Pair of batches of data in :class:`dict` (s).
            Must have ``"image"`` and ``"bbox"`` keys.
        model (MinervaModel): Model being fitted.
        device (~torch.device): :mod:`torch` device object to send data to (e.g. ``CUDA`` device).
        train (bool): True to run a step of the model in training mode. False for eval mode.

    Kwargs:
        mix_precision (bool): Use mixed-precision. Will set the floating tensors to 16-bit
            rather than the default 32-bit.

    Returns:
        tuple[
            ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~typing.Sequence[~torchgeo.datasets.utils.BoundingBox] | None
        ]: The ``loss``, the model output ``z``, the ``y`` supplied and the bounding boxes
        of the original input images supplied.
    """
    float_dtype = _determine_float_dtype(device, kwargs.get("mix_precision", False))

    # Extracts both batches from the dict and ensures images are floats.
    x_i_batch: Tensor = batch[0]["image"].to(float_dtype)  # type: ignore[attr-defined]
    x_j_batch: Tensor = batch[1]["image"].to(float_dtype)  # type: ignore[attr-defined]

    if kwargs.get("validate_variables", False):
        try:
            assert_raises(AssertionError, assert_array_equal, x_i_batch, x_j_batch)
        except AssertionError:
            print("WARNING: Batches are the same!")

    # Stacks each side of the pair batches together and transfer to GPU.
    x = torch.stack([x_i_batch, x_j_batch]).to(device, non_blocking=True)

    # Check that none of the data is NaN or infinity.
    if kwargs.get("validate_variables", False):
        assert not x.isnan().any()
        assert not x.isinf().any()

    # Runs a step of the epoch.
    loss, z = model.step(x, train=train)

    if "bbox" in batch[0].keys():
        return loss, z, None, batch[0]["bbox"] + batch[1]["bbox"]
    elif "id" in batch[0].keys():
        return loss, z, None, batch[0]["id"] + batch[1]["id"]
    else:
        return loss, z, None, None


def _determine_float_dtype(device: torch.device, mix_precision: bool) -> torch.dtype:
    if mix_precision is True:
        return torch.bfloat16 if device.type == "cpu" else torch.float16
    else:
        return torch.float
