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
r"""Tests for :mod:`minerva.modelio`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
import importlib
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.modules as nn
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import NTXentLoss
except (OSError, NewConnectionError, MaxRetryError):
    NTXentLoss = getattr(importlib.import_module("lightly.loss"), "NTXentLoss")
import pytest
from numpy.testing import assert_array_equal
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox

from minerva.modelio import autoencoder_io, ssl_pair_tg, sup_tg
from minerva.models import FCN32ResNet18, SimCLR34


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_sup_tg(
    simple_bbox: BoundingBox,
    random_rgbi_batch: Tensor,
    random_mask_batch: Tensor,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
    default_device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=rgbi_input_size).to(default_device)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        bboxes = [simple_bbox] * std_batch_size
        batch: Dict[str, Union[Tensor, List[Any]]] = {
            "image": random_rgbi_batch,
            "mask": random_mask_batch,
            "bbox": bboxes,
        }

        results = sup_tg(batch, model, default_device, mode)

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (
            std_batch_size,
            std_n_classes,
            *rgbi_input_size[1:],
        )
        assert_array_equal(results[2].detach().cpu(), batch["mask"].detach().cpu())  # type: ignore[union-attr]
        assert results[3] == batch["bbox"]


def test_ssl_pair_tg(
    simple_bbox: BoundingBox,
    std_batch_size: int,
    rgbi_input_size: Tuple[int, int, int],
    default_device: torch.device,
) -> None:
    criterion = NTXentLoss(0.5)
    model = SimCLR34(criterion, input_size=rgbi_input_size).to(default_device)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val"):
        images_1 = torch.rand(size=(std_batch_size, *rgbi_input_size))
        bboxes_1 = [simple_bbox] * std_batch_size

        batch_1 = {
            "image": images_1,
            "bbox": bboxes_1,
        }

        images_2 = torch.rand(size=(std_batch_size, *rgbi_input_size))
        bboxes_2 = [simple_bbox] * std_batch_size

        batch_2 = {
            "image": images_2,
            "bbox": bboxes_2,
        }

        results = ssl_pair_tg((batch_1, batch_2), model, default_device, mode)

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (2 * std_batch_size, 128)
        assert results[2] is None
        assert isinstance(batch_1["bbox"], list)
        assert isinstance(batch_2["bbox"], list)
        assert results[3] == batch_1["bbox"] + batch_2["bbox"]


def test_mask_autoencoder_io(
    simple_bbox: BoundingBox,
    std_batch_size: int,
    std_n_classes: int,
    rgbi_input_size: Tuple[int, int, int],
    default_device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=(8, *rgbi_input_size[1:])).to(
        default_device
    )
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        images = torch.rand(size=(std_batch_size, *rgbi_input_size))
        masks = torch.randint(0, 8, (std_batch_size, *rgbi_input_size[1:]))  # type: ignore[attr-defined]
        bboxes = [simple_bbox] * std_batch_size
        batch: Dict[str, Union[Tensor, List[Any]]] = {
            "image": images,
            "mask": masks,
            "bbox": bboxes,
        }

        with pytest.raises(
            ValueError,
            match="The value of key='wrong' is not understood. Must be either 'mask' or 'image'",
        ):
            autoencoder_io(
                batch, model, default_device, mode, autoencoder_data_key="wrong"
            )

        results = autoencoder_io(
            batch, model, default_device, mode, autoencoder_data_key="mask"
        )

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (
            std_batch_size,
            std_n_classes,
            *rgbi_input_size[1:],
        )
        assert_array_equal(results[2].detach().cpu(), batch["mask"].detach().cpu())  # type: ignore[union-attr]
        assert results[3] == batch["bbox"]


def test_image_autoencoder_io(
    simple_bbox: BoundingBox,
    random_rgbi_batch: Tensor,
    random_mask_batch: Tensor,
    std_batch_size: int,
    rgbi_input_size: Tuple[int, int, int],
    default_device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=rgbi_input_size, n_classes=4).to(
        default_device
    )
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        bboxes = [simple_bbox] * std_batch_size
        batch: Dict[str, Union[Tensor, List[Any]]] = {
            "image": random_rgbi_batch,
            "mask": random_mask_batch,
            "bbox": bboxes,
        }

        results = autoencoder_io(
            batch, model, default_device, mode, autoencoder_data_key="image"
        )

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (std_batch_size, *rgbi_input_size)
        assert_array_equal(results[2].detach().cpu(), batch["image"].detach().cpu())  # type: ignore[union-attr]
        assert results[3] == batch["bbox"]
