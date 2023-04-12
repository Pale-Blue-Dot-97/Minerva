# -*- coding: utf-8 -*-
import importlib
from typing import Any, Dict, List, Union

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

from minerva.modelio import autoencoder_io, ssl_pair_tg, sup_tg
from minerva.models import FCN32ResNet18, SimCLR34

input_size = (4, 64, 64)
batch_size = 3
n_classes = 8
device = torch.device("cpu")  # type: ignore[attr-defined]


def test_sup_tg(simple_bbox) -> None:
    criterion = nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=input_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        images = torch.rand(size=(batch_size, *input_size))
        masks = torch.randint(0, n_classes, (batch_size, *input_size[1:]))  # type: ignore[attr-defined]
        bboxes = [simple_bbox] * batch_size
        batch: Dict[str, Union[Tensor, List[Any]]] = {
            "image": images,
            "mask": masks,
            "bbox": bboxes,
        }

        results = sup_tg(batch, model, device, mode)

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (batch_size, n_classes, *input_size[1:])
        assert_array_equal(results[2], batch["mask"])
        assert results[3] == batch["bbox"]


def test_ssl_pair_tg(simple_bbox) -> None:
    criterion = NTXentLoss(0.5)
    model = SimCLR34(criterion, input_size=input_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val"):
        images_1 = torch.rand(size=(batch_size, *input_size))
        bboxes_1 = [simple_bbox] * batch_size

        batch_1 = {
            "image": images_1,
            "bbox": bboxes_1,
        }

        images_2 = torch.rand(size=(batch_size, *input_size))
        bboxes_2 = [simple_bbox] * batch_size

        batch_2 = {
            "image": images_2,
            "bbox": bboxes_2,
        }

        results = ssl_pair_tg((batch_1, batch_2), model, device, mode)

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (2 * batch_size, 128)
        assert results[2] is None
        assert isinstance(batch_1["bbox"], list)
        assert isinstance(batch_2["bbox"], list)
        assert results[3] == batch_1["bbox"] + batch_2["bbox"]


def test_mask_autoencoder_io(simple_bbox) -> None:
    criterion = nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=(8, *input_size[1:]))
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        images = torch.rand(size=(batch_size, *input_size))
        masks = torch.randint(0, 8, (batch_size, *input_size[1:]))  # type: ignore[attr-defined]
        bboxes = [simple_bbox] * batch_size
        batch: Dict[str, Union[Tensor, List[Any]]] = {
            "image": images,
            "mask": masks,
            "bbox": bboxes,
        }

        with pytest.raises(
            ValueError,
            match="The value of key='wrong' is not understood. Must be either 'mask' or 'image'",
        ):
            autoencoder_io(batch, model, device, mode, autoencoder_data_key="wrong")

        results = autoencoder_io(
            batch, model, device, mode, autoencoder_data_key="mask"
        )

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (batch_size, n_classes, *input_size[1:])
        assert_array_equal(results[2], batch["mask"])
        assert results[3] == batch["bbox"]


def test_image_autoencoder_io(simple_bbox) -> None:
    criterion = nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=input_size, n_classes=4)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        images = torch.rand(size=(batch_size, *input_size))
        masks = torch.randint(0, 8, (batch_size, *input_size[1:]))  # type: ignore[attr-defined]
        bboxes = [simple_bbox] * batch_size
        batch: Dict[str, Union[Tensor, List[Any]]] = {
            "image": images,
            "mask": masks,
            "bbox": bboxes,
        }

        results = autoencoder_io(
            batch, model, device, mode, autoencoder_data_key="image"
        )

        assert isinstance(results[0], Tensor)
        assert isinstance(results[1], Tensor)
        assert results[1].size() == (batch_size, *input_size)
        assert_array_equal(results[2], batch["image"])
        assert results[3] == batch["bbox"]
