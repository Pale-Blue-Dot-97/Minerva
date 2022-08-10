# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Module to handle the logging of results from various model types."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets.utils import BoundingBox

from minerva.utils import utils

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaLogger(ABC):
    """Base abstract class for all `minerva` logger classes to ensure intercompatibility with `trainer`.

    Attributes:
        record_int (bool): Whether to record the integer values from an epoch of model fitting.
        record_float (bool): Whether to record the floating point values from an epoch of model fitting.
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        logs (Dict[str, Any]): Dictionary to hold the logs from the epoch.
            Logs should be more lightweight than `results`.
        results (Dict[str, Any]): Dictionary to hold the results from the epoch.

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to True.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to False.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        n_samples: int,
        record_int: bool = True,
        record_float: bool = False,
    ) -> None:

        super(MinervaLogger, self).__init__()
        self.record_int = record_int
        self.record_float = record_float
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.logs: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def __call__(
        self, mode: str, step_num: int, writer: SummaryWriter, loss: _Loss, *args
    ) -> None:
        """Call :func:`log`.

        Args:
            mode (str): Mode of model fitting.
            step_num (int): The global step number of for the mode of model fitting.
            writer (SummaryWriter): Writer object from `tensorboard`.
            loss (_Loss): Loss from this step of model fitting.

        Returns:
            None
        """
        self.log(mode, step_num, writer, loss, *args)

    @abc.abstractmethod
    def log(
        self,
        mode: str,
        step_num: int,
        writer: SummaryWriter,
        loss: _Loss,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Abstract logging method, the core functionality of a logger. Must be overwritten.

        Args:
            mode (str): Mode of model fitting.
            step_num (int): The global step number of for the mode of model fitting.
            writer (SummaryWriter): Writer object from `tensorboard`.
            loss (_Loss): Loss from this step of model fitting.
            z (Tensor): Optional; Output tensor from the model.
            y (Tensor): Optional; Labels to assess model output against.
            bbox (BoundingBox): Optional; Bounding boxes of the input samples.

        Returns:
            None
        """
        pass

    @property
    def get_logs(self) -> Dict[str, Any]:
        """Gets the logs dictionary.

        Returns:
            Dict[str, Any]: Log dictionary of the logger.
        """
        return self.logs

    @property
    def get_results(self) -> Dict[str, Any]:
        """Gets the results dictionary.

        Returns:
            Dict[str,Any]: Results dictionary of the logger.
        """
        return self.results


class STG_Logger(MinervaLogger):
    """Logger designed for supervised learning using `torchgeo` datasets.

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        out_shape (Tuple[int, ...]): Shape of the model output.
        n_classes (int): Number of classes in dataset.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to True.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to False.

    Raises:
        MemoryError: If trying to allocate memory to hold the probabilites of predictions
            from the model exceeds capacity.
        MemoryError: If trying to allocate memory to hold the bounding boxes of samples would exceed capacity.
    """

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        n_samples: int,
        out_shape: Tuple[int, ...],
        n_classes: int,
        record_int: bool = True,
        record_float: bool = False,
    ) -> None:

        super(STG_Logger, self).__init__(
            n_batches, batch_size, n_samples, record_int, record_float
        )

        self.logs: Dict[str, Any] = {
            "batch_num": 0,
            "total_loss": 0.0,
            "total_correct": 0.0,
        }

        self.results: Dict[str, Any] = {
            "y": None,
            "z": None,
            "probs": None,
            "ids": [],
            "bounds": None,
        }

        # Allocate memory for the integer values to be recorded.
        if self.record_int:
            self.results["y"] = np.empty(
                (self.n_batches, self.batch_size, *out_shape), dtype=np.uint8
            )
            self.results["z"] = np.empty(
                (self.n_batches, self.batch_size, *out_shape), dtype=np.uint8
            )

        # Allocate memory for the floating point values to be recorded.
        if self.record_float:
            try:
                self.results["probs"] = np.empty(
                    (self.n_batches, self.batch_size, n_classes, *out_shape),
                    dtype=np.float16,
                )
            except MemoryError:
                raise MemoryError(
                    "Dataset too large to record probabilities of predicted classes!"
                )

            try:
                self.results["bounds"] = np.empty(
                    (self.n_batches, self.batch_size), dtype=object
                )
            except MemoryError:
                raise MemoryError(
                    "Dataset too large to record bounding boxes of samples!"
                )

    def log(
        self,
        mode: str,
        step_num: int,
        writer: SummaryWriter,
        loss: _Loss,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Logs the outputs and results from a step of model fitting. Overwrites abstract method.

        Args:
            mode (str): Mode of model fitting.
            step_num (int): The global step number of for the mode of model fitting.
            writer (SummaryWriter): Writer object from `tensorboard`.
            loss (_Loss): Loss from this step of model fitting.
            z (Tensor): Output tensor from the model.
            y (Tensor): Labels to assess model output against.
            bbox (BoundingBox): Bounding boxes of the input samples.

        Returns:
            None
        """
        assert z is not None

        if self.record_int:
            assert y is not None
            # Arg max the estimated probabilities and add to predictions.
            self.results["z"][self.logs["batch_num"]] = torch.argmax(z, 1).cpu().numpy()

            # Add the labels and sample IDs to lists.
            self.results["y"][self.logs["batch_num"]] = y.cpu().numpy()
            batch_ids = []
            for i in range(
                self.logs["batch_num"] * self.batch_size,
                (self.logs["batch_num"] + 1) * self.batch_size,
            ):
                batch_ids.append(str(i).zfill(len(str(self.n_samples))))
            self.results["ids"].append(batch_ids)

        if self.record_float:
            assert bbox is not None
            # Add the estimated probabilities to probs.
            self.results["probs"][self.logs["batch_num"]] = z.detach().cpu().numpy()
            self.results["bounds"][self.logs["batch_num"]] = bbox

        # Computes the loss and the correct predictions from this step.
        ls = loss.item()
        correct = (torch.argmax(z, 1) == y).sum().item()

        # Adds loss and correct predictions to logs.
        self.logs["total_loss"] += ls
        self.logs["total_correct"] += correct

        # Writes loss and correct predictions to the writer.
        writer.add_scalar(tag=f"{mode}_loss", scalar_value=ls, global_step=step_num)
        writer.add_scalar(
            tag=f"{mode}_acc",
            scalar_value=correct / len(torch.flatten(y)),
            global_step=step_num,
        )

        # Adds 1 to batch number (step number).
        self.logs["batch_num"] += 1


class SSL_Logger(MinervaLogger):
    """Logger designed for self-supervised learning.

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        out_shape (Tuple[int, ...]): Shape of the model output.
        n_classes (int): Number of classes in dataset.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to True.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to False.
    """

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        n_samples: int,
        out_shape: Optional[Tuple[int, ...]] = None,
        n_classes: Optional[int] = None,
        record_int: bool = True,
        record_float: bool = False,
    ) -> None:

        super(SSL_Logger, self).__init__(
            n_batches, batch_size, n_samples, record_int, record_float=False
        )

        self.logs: Dict[str, Any] = {
            "batch_num": 0,
            "total_loss": 0.0,
            "total_correct": 0.0,
            "total_top5": 0.0,
            "avg_loss": 0.0,
            "avg_output_std": 0.0,
        }

    def log(
        self,
        mode: str,
        step_num: int,
        writer: SummaryWriter,
        loss: _Loss,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Logs the outputs and results from a step of model fitting. Overwrites abstract method.

        Args:
            mode (str): Mode of model fitting.
            step_num (int): The global step number of for the mode of model fitting.
            writer (SummaryWriter): Writer object from `tensorboard`.
            loss (_Loss): Loss from this step of model fitting.
            z (Tensor): Optional; Output tensor from the model.
            y (Tensor): Optional; Labels to assess model output against.
            bbox (BoundingBox): Optional; Bounding boxes of the input samples.
        """
        assert z is not None

        # Adds the loss for this step to the logs.
        ls = loss.item()
        self.logs["total_loss"] += ls

        # Compute the TOP1 and TOP5 accuracies.
        sim_argsort = utils.calc_contrastive_acc(z)
        correct = float((sim_argsort == 0).float().mean().cpu().numpy())
        top5 = float((sim_argsort < 5).float().mean().cpu().numpy())

        if kwargs["collapse_level"]:
            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = torch.split(z, 0.5 * len(z), 0)[0].detach()
            output = torch.nn.functional.normalize(output, dim=1)

            output_std = torch.std(output, 0)
            output_std = output_std.mean()

            # use moving averages to track the loss and standard deviation
            w = 0.9
            self.logs["avg_loss"] = w * self.logs["avg_loss"] + (1 - w) * ls
            self.logs["avg_output_std"] = (
                w * self.logs["avg_output_std"] + (1 - w) * output_std.item()
            )

            # the level of collapse is large if the standard deviation of the l2
            # normalized output is much smaller than 1 / sqrt(dim)
            collapse_level = max(
                0.0, 1 - math.sqrt(len(output)) * self.logs["avg_output_std"]
            )

            self.logs["collapse_level"] = collapse_level

        # Add accuracies to log.
        self.logs["total_correct"] += correct
        self.logs["total_top5"] += top5

        # Writes the loss to the writer.
        writer.add_scalar(tag=f"{mode}_loss", scalar_value=ls, global_step=step_num)
        writer.add_scalar(
            tag=f"{mode}_acc",
            scalar_value=correct / 2 * len(z[0]),
            global_step=step_num,
        )
        writer.add_scalar(
            tag=f"{mode}_top5_acc",
            scalar_value=top5 / 2 * len(z[0]),
            global_step=step_num,
        )

        # Adds 1 to the batch number (step number).
        self.logs["batch_num"] += 1
