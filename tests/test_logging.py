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
r"""Tests for :mod:`minerva.logging`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

import importlib
import shutil
import tempfile
from pathlib import Path

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from typing import Any

import numpy as np
import torch
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import NTXentLoss
except (OSError, NewConnectionError, MaxRetryError):
    NTXentLoss = getattr(importlib.import_module("lightly.loss"), "NTXentLoss")
import pytest
from nptyping import NDArray, Shape
from numpy.testing import assert_array_equal
from torch import Tensor
from torch.nn.modules import Module
from torchgeo.datasets.utils import BoundingBox

from minerva.logger.steplog import SupervisedStepLogger
from minerva.logger.tasklog import SSLTaskLogger, SupervisedTaskLogger
from minerva.loss import SegBarlowTwinsLoss
from minerva.modelio import ssl_pair_torchgeo_io, supervised_torchgeo_io
from minerva.models import FCN16ResNet18, MinervaSiamese, SimCLR18, SimConvPSP
from minerva.utils import utils

n_epochs = 2


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
@pytest.mark.parametrize("train", (True, False))
@pytest.mark.parametrize("model_type", ("scene_classifier", "segmentation"))
def test_SupervisedStepLogger(
    simple_bbox: BoundingBox,
    x_entropy_loss,
    std_n_batches: int,
    std_n_classes: int,
    std_batch_size: int,
    small_patch_size: tuple[int, int],
    default_device: torch.device,
    train: bool,
    model_type: str,
) -> None:
    path = Path(tempfile.gettempdir(), "exp1")

    if not path.exists():
        path.mkdir()

    try:
        tensorboard_writer = utils._optional_import(
            "torch.utils.tensorboard.writer", name="SummaryWriter", package="tensorflow"
        )
    except ImportError as err:
        print(err)
        writer = None
    else:
        writer = tensorboard_writer(log_dir=path)

    input_size = (4, *small_patch_size)
    model = FCN16ResNet18(x_entropy_loss, input_size=input_size).to(default_device)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)
    model.determine_output_dim()

    output_shape = model.output_shape
    assert isinstance(output_shape, tuple)

    with pytest.raises(
        ValueError, match="`n_classes` must be specified for this type of logger!"
    ):
        _ = SupervisedStepLogger(
            task_name="pytest",
            n_batches=std_n_batches,
            batch_size=std_batch_size,
            input_size=input_size,
            output_size=output_shape,
            record_int=True,
            record_float=True,
            writer=writer,
            model_type=model_type,
        )

    logger = SupervisedTaskLogger(
        task_name="pytest",
        n_batches=std_n_batches,
        batch_size=std_batch_size,
        input_size=input_size,
        output_size=output_shape,
        n_classes=std_n_classes,
        record_int=True,
        record_float=True,
        writer=writer,
        model_type=model_type,
        step_logger_params={"n_classes": std_n_classes},
    )

    correct_loss: dict[str, list[float]] = {"x": [], "y": []}
    correct_acc: dict[str, list[float]] = {"x": [], "y": []}
    correct_miou: dict[str, list[float]] = {"x": [], "y": []}

    for epoch_no in range(n_epochs):
        data: list[dict[str, Tensor | list[Any]]] = []
        for i in range(std_n_batches):
            images = torch.rand(size=(std_batch_size, 4, *small_patch_size))
            masks = torch.randint(  # type: ignore[attr-defined]
                0, std_n_classes, (std_batch_size, *small_patch_size)
            )
            bboxes = [simple_bbox] * std_batch_size
            batch: dict[str, Tensor | list[Any]] = {
                "image": images,
                "mask": masks,
                "bbox": bboxes,
            }
            data.append(batch)

            logger.step(
                i,
                i,
                *supervised_torchgeo_io(
                    batch, model, device=default_device, train=train
                ),
            )  # type: ignore[arg-type]  # noqa: E501

        logs = logger.get_logs
        assert logs["batch_num"] == std_n_batches - 1
        assert isinstance(logs["total_loss"], float)
        assert isinstance(logs["total_correct"], float)

        if model_type == "segmentation":
            assert isinstance(logs["total_miou"], float)

        results = logger.get_results
        assert results["z"].shape == (
            std_n_batches,
            std_batch_size,
            *small_patch_size,
        )
        assert results["y"].shape == (
            std_n_batches,
            std_batch_size,
            *small_patch_size,
        )
        assert results["x"].shape == (
            std_n_batches,
            std_batch_size,
            *input_size,
        )
        assert np.array(results["ids"]).shape == (std_n_batches, std_batch_size)

        shape = f"{std_n_batches}, {std_batch_size}, {small_patch_size[0]}, {small_patch_size[1]}"
        y: NDArray[Shape[shape], Any] = np.empty(
            (std_n_batches, std_batch_size, *output_shape), dtype=np.uint8
        )
        for i in range(std_n_batches):
            mask: Tensor | list[Any] = data[i]["mask"]
            assert isinstance(mask, Tensor)
            y[i] = mask.cpu().numpy()

        assert_array_equal(results["y"], y)

        correct_loss["x"].append(epoch_no)
        correct_loss["y"].append(logs["total_loss"] / std_n_batches)

        correct_acc["x"].append(epoch_no)
        if utils.check_substrings_in_string(model_type, "segmentation"):
            correct_acc["y"].append(
                logs["total_correct"]
                / float(std_n_batches * std_batch_size * np.prod(small_patch_size))
            )
        else:
            correct_acc["y"].append(
                logs["total_correct"] / (std_n_batches * std_batch_size)
            )

        if utils.check_substrings_in_string(model_type, "segmentation"):
            correct_miou["x"].append(epoch_no)
            correct_miou["y"].append(
                logs["total_miou"] / (std_n_batches * std_batch_size)
            )

        logger.calc_metrics(epoch_no)
        logger.print_epoch_results(epoch_no)

        metrics = logger.get_metrics

        assert metrics["pytest_loss"] == pytest.approx(correct_loss)
        assert metrics["pytest_acc"] == pytest.approx(correct_acc)

        if model_type == "segmentation":
            assert metrics["pytest_miou"] == pytest.approx(correct_miou)

        logger._make_logger()

    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize(
    ("model_cls", "model_type", "criterion"),
    (
        (SimCLR18, "siamese", NTXentLoss()),
        (SimConvPSP, "siamese-segmentation", SegBarlowTwinsLoss()),
    ),
)
@pytest.mark.parametrize("train", (True, False))
@pytest.mark.parametrize("extra_metrics", (True, False))
def test_SSLStepLogger(
    simple_bbox: BoundingBox,
    std_n_batches: int,
    std_batch_size: int,
    small_patch_size: tuple[int, int],
    default_device: torch.device,
    model_cls: MinervaSiamese,
    model_type: str,
    criterion: Module,
    train: bool,
    extra_metrics: bool,
) -> None:
    path = Path(tempfile.gettempdir(), "exp2")

    if not path.exists():
        path.mkdir()

    try:
        tensorboard_writer = utils._optional_import(
            "torch.utils.tensorboard.writer", name="SummaryWriter", package="tensorflow"
        )
    except ImportError as err:
        print(err)
        writer = None
    else:
        writer = tensorboard_writer(log_dir=path)

    input_size = (4, *small_patch_size)
    model: MinervaSiamese = model_cls(criterion, input_size=input_size).to(
        default_device
    )
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    model.determine_output_dim(sample_pairs=True)
    output_shape = model.output_shape
    assert isinstance(output_shape, tuple)

    logger = SSLTaskLogger(
        task_name="pytest",
        n_batches=std_n_batches,
        batch_size=std_batch_size,
        input_size=input_size,
        output_size=small_patch_size,
        record_int=True,
        record_float=True,
        writer=writer,
        model_type=model_type,
        sample_pairs=True,
    )

    correct_loss: dict[str, list[float]] = {"x": [], "y": []}
    correct_collapse_level: dict[str, list[float]] = {"x": [], "y": []}
    correct_euc_dist: dict[str, list[float]] = {"x": [], "y": []}

    for epoch_no in range(n_epochs):
        for i in range(std_n_batches):
            images = torch.rand(size=(std_batch_size, 4, *small_patch_size))
            bboxes = [simple_bbox] * std_batch_size
            batch = {"image": images, "bbox": bboxes}

            logger.step(
                i,
                i,
                *ssl_pair_torchgeo_io(
                    (batch, batch), model, device=default_device, train=train
                ),  # type: ignore[arg-type]  # noqa: E501
            )

        logs = logger.get_logs
        assert logs["batch_num"] == std_n_batches - 1
        assert isinstance(logs["total_loss"], float)

        if extra_metrics:
            assert isinstance(logs["collapse_level"], float)
            assert isinstance(logs["euc_dist"], float)

        results = logger.get_results
        assert results == {}

        correct_loss["x"].append(epoch_no)
        correct_loss["y"].append(logs["total_loss"] / std_n_batches)

        if extra_metrics:
            correct_collapse_level["x"].append(epoch_no)
            correct_euc_dist["x"].append(epoch_no)
            correct_collapse_level["y"].append(logs["collapse_level"])
            correct_euc_dist["y"].append(logs["euc_dist"] / std_n_batches)

        logger.calc_metrics(epoch_no)
        logger.print_epoch_results(epoch_no)

        metrics = logger.get_metrics

        assert metrics["pytest_loss"] == pytest.approx(correct_loss)

        if extra_metrics:
            assert metrics["pytest_collapse_level"] == pytest.approx(
                correct_collapse_level
            )
            assert metrics["pytest_euc_dist"] == pytest.approx(correct_euc_dist)

        logger._make_logger()

    shutil.rmtree(path, ignore_errors=True)
