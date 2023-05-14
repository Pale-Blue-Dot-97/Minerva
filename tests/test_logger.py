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
r"""Tests for :mod:`minerva.logger`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

import importlib
import shutil
import tempfile
from pathlib import Path

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Needed to avoid connection error when importing lightly.
try:
    from lightly.loss import NTXentLoss
except (OSError, NewConnectionError, MaxRetryError):
    NTXentLoss = getattr(importlib.import_module("lightly.loss"), "NTXentLoss")
from nptyping import NDArray, Shape
from numpy.testing import assert_array_equal
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox

from minerva.logger import SSLLogger, STGLogger
from minerva.modelio import ssl_pair_tg, sup_tg
from minerva.models import FCN16ResNet18, SimCLR18
from minerva.utils import utils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_STGLogger(
    simple_bbox: BoundingBox,
    x_entropy_loss,
    std_n_batches: int,
    std_n_classes: int,
    std_batch_size: int,
    small_patch_size: Tuple[int, int],
    default_device: torch.device,
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

    model = FCN16ResNet18(x_entropy_loss, input_size=(4, *small_patch_size)).to(
        default_device
    )
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)
    model.determine_output_dim()

    output_shape = model.output_shape
    assert isinstance(output_shape, tuple)

    for mode in ("train", "val", "test"):
        for model_type in ("scene_classifier", "segmentation"):
            logger = STGLogger(
                n_batches=std_n_batches,
                batch_size=std_batch_size,
                n_samples=std_n_batches
                * std_batch_size
                * small_patch_size[0]
                * small_patch_size[1],
                out_shape=output_shape,
                n_classes=std_n_classes,
                record_int=True,
                record_float=True,
                model_type=model_type,
                writer=writer,
            )
            data: List[Dict[str, Union[Tensor, List[Any]]]] = []
            for i in range(std_n_batches):
                images = torch.rand(size=(std_batch_size, 4, *small_patch_size))
                masks = torch.randint(  # type: ignore[attr-defined]
                    0, std_n_classes, (std_batch_size, *small_patch_size)
                )
                bboxes = [simple_bbox] * std_batch_size
                batch: Dict[str, Union[Tensor, List[Any]]] = {
                    "image": images,
                    "mask": masks,
                    "bbox": bboxes,
                }
                data.append(batch)

                logger(mode, i, *sup_tg(batch, model, device=default_device, mode=mode))

            logs = logger.get_logs
            assert logs["batch_num"] == std_n_batches
            assert type(logs["total_loss"]) is float
            assert type(logs["total_correct"]) is float

            if model_type == "segmentation":
                assert type(logs["total_miou"]) is float

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
            assert np.array(results["ids"]).shape == (std_n_batches, std_batch_size)

            shape = f"{std_n_batches}, {std_batch_size}, {small_patch_size[0]}, {small_patch_size[1]}"
            y: NDArray[Shape[shape], Any] = np.empty(
                (std_n_batches, std_batch_size, *output_shape), dtype=np.uint8
            )
            for i in range(std_n_batches):
                mask: Union[Tensor, List[Any]] = data[i]["mask"]
                assert isinstance(mask, Tensor)
                y[i] = mask.cpu().numpy()

            assert_array_equal(results["y"], y)

    shutil.rmtree(path, ignore_errors=True)


def test_SSLLogger(
    simple_bbox: BoundingBox,
    std_n_batches: int,
    std_batch_size: int,
    small_patch_size: Tuple[int, int],
    default_device: torch.device,
) -> None:
    criterion = NTXentLoss(0.5)

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

    model = SimCLR18(criterion, input_size=(4, *small_patch_size)).to(default_device)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        for extra_metrics in (True, False):
            logger = SSLLogger(
                n_batches=std_n_batches,
                batch_size=std_batch_size,
                n_samples=std_n_batches * std_batch_size,
                record_int=True,
                record_float=True,
                collapse_level=extra_metrics,
                euclidean=extra_metrics,
                writer=writer,
            )
            data = []
            for i in range(std_n_batches):
                images = torch.rand(size=(std_batch_size, 4, *small_patch_size))
                bboxes = [simple_bbox] * std_batch_size
                batch = {
                    "image": images,
                    "bbox": bboxes,
                }
                data.append((batch, batch))

                logger(
                    mode,
                    i,
                    *ssl_pair_tg(
                        (batch, batch), model, device=default_device, mode=mode
                    ),
                )

            logs = logger.get_logs
            assert logs["batch_num"] == std_n_batches
            assert type(logs["total_loss"]) is float
            assert type(logs["total_correct"]) is float
            assert type(logs["total_top5"]) is float

            if extra_metrics:
                assert type(logs["collapse_level"]) is float
                assert type(logs["euc_dist"]) is float

            results = logger.get_results
            assert results == {}

    shutil.rmtree(path, ignore_errors=True)
