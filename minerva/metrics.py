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
"""Module to calculate the metrics of a model's fitting."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "MinervaMetrics",
    "SPMetrics",
    "SSLMetrics",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

from torch import Tensor
from torchgeo.datasets.utils import BoundingBox
from wandb.sdk.wandb_run import Run

from minerva.logger import MinervaLogger, SSLLogger, STGLogger, get_logger


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaMetrics(ABC):
    """Abstract class for metric logging within the :mod:`minerva` framework.

    Attributes:
        n_batches (dict[str, int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (tuple[int, int, int]): Shape of the input data in ``C x H x W``.
        metrics (dict[str, ~typing.Any]): Dictionary to hold the metrics to assess the model with
            for each mode of fitting.
        model_type (str): Type of the model.

    Args:
        n_batches (dict[str, int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (tuple[int, int, int]): Shape of the input data in ``C x H x W``.
        logger_params (dict[str, ~typing.Any]): Optional; Parameters for a logger other than the default for these metrics.
    """

    __metaclass__ = abc.ABCMeta

    metric_types: List[str] = []
    special_metric_types: List[str] = []
    logger_cls: Callable[..., MinervaLogger]

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        data_size: Tuple[int, int, int],
        task_name: str,
        logger_params: Optional[Dict[str, Any]] = None,
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        **params,
    ) -> None:
        super(MinervaMetrics, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_samples = self.n_batches * self.batch_size
        self.data_size = data_size
        self.task_name = task_name

        self.model_type = params.get("model_type", "scene_classifier")
        self.sample_pairs = params.get("sample_pairs", False)

        self.writer = writer

        if logger_params:
            if logger_params.get("name", None) is not None:
                self.logger_cls = get_logger(logger_params["name"])

        else:
            logger_params = {}

        self.logger = self.logger_cls(
            self.n_batches,
            self.batch_size,
            self.n_samples,
            record_int,
            record_float,
            self.writer,
            **logger_params["params"],
        )

        if self.sample_pairs:
            self.metric_types += self.special_metric_types

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics: Dict[str, Any] = {}
        for metric in self.metric_types:
            self.metrics[f"{self.task_name}_{metric}"] = {"x": [], "y": []}

    def step(
        self,
        step_num: int,
        loss: Tensor,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Abstract method to log a step, using the logger. Must be overwritten.

        Args:
            mode (str): Mode of model fitting.
            step_num (int): The global step number of for the mode of model fitting.
            loss (~torch.Tensor): Loss from this step of model fitting.
            z (~torch.Tensor): Optional; Output tensor from the model.
            y (~torch.Tensor): Optional; Labels to assess model output against.
            bbox (~torchgeo.datasets.utils.BoundingBox): Optional; Bounding boxes of the input samples.

        Returns:
            None
        """
        self.logger.log(
            step_num,
            loss,
            z,
            y,
            bbox,
            *args,
            **kwargs,
        )

    def calc_metrics(self) -> None:
        """Updates metrics with epoch results."""
        self._calc_metrics(self.logger.get_logs)

    @abc.abstractmethod
    def _calc_metrics(self, logs: Dict[str, Any]) -> None:
        """Updates metrics with epoch results.

        Must be defined before use.

        Args:
            logs (dict[str, ~typing.Any]): Logs of the results from the epoch of the task to calculate metrics from.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def log_epoch_number(self, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            epoch_no (int): Epoch number to log.
        """
        pass  # pragma: no cover

    @property
    def get_metrics(self) -> Dict[str, Any]:
        """Get the ``metrics`` dictionary.

        Returns:
            dict[str, Any]: Metrics dictionary.
        """
        return self.metrics

    def get_sub_metrics(
        self, pattern: Tuple[str, ...] = ("train", "val")
    ) -> Dict[str, Any]:
        """Gets a subset of the metrics dictionary with keys containing strings in the pattern.

        Useful for getting the train and validation metrics for plotting for example.

        Args:
            pattern (tuple[str, ...]): Optional; Strings to pattern match the metric keys to be returned.
                Defaults to ``("train", "val")``.

        Returns:
            dict[str, ~typing.Any]: Subset of ``metrics`` with keys that contained strings in ``pattern``.
        """
        sub_metrics = {}
        for key in self.metrics.keys():
            if key.split("_")[0] in pattern:
                sub_metrics[key] = self.metrics[key]

        return sub_metrics

    @abc.abstractmethod
    def print_epoch_results(self, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            epoch_no (int): Epoch number to print results from.
        """
        pass  # pragma: no cover


class SPMetrics(MinervaMetrics):
    """Metric logging for supervised models.

    Attributes:
        n_batches (dict[str, int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (tuple[int, int, int]): Shape of the input data in ``C x H x W``.
        metrics (dict[str, ~typing.Any]): Dictionary to hold the metrics to assess the model with
            for each mode of fitting.
        model_type (str): Type of the model.

    Args:
        n_batches (dict[str, int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (tuple[int, int, int]): Shape of the input data in ``C x H x W``.
        model_type (str): Optional; Type of the model.
    """

    metric_types: List[str] = ["loss", "acc", "miou"]
    logger_cls = STGLogger

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        data_size: Tuple[int, int, int],
        task_name: str,
        logger_params: Optional[Dict[str, Any]] = None,
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        model_type: str = "segmentation",
        **params,
    ) -> None:
        super(SPMetrics, self).__init__(
            n_batches,
            batch_size,
            data_size,
            task_name,
            logger_params,
            record_int,
            record_float,
            writer,
            model_type=model_type,
        )

    def _calc_metrics(self, logs: Dict[str, Any]) -> None:
        """Updates metrics with epoch results.

        Args:
            mode (str): Mode of model fitting.
            logs (dict[str, ~typing.Any]): Logs of the results from the epoch of fitting to calculate metrics from.
        """
        self.metrics[f"{self.task_name}_loss"]["y"].append(
            logs["total_loss"] / self.n_batches
        )

        if self.model_type == "segmentation":
            self.metrics[f"{self.task_name}_acc"]["y"].append(
                logs["total_correct"]
                / (
                    self.n_batches
                    * self.batch_size
                    * self.data_size[1]
                    * self.data_size[2]
                )
            )
            if logs.get("total_miou") is not None:
                self.metrics[f"{self.task_name}_miou"]["y"].append(
                    logs["total_miou"] / (self.n_samples)
                )

        else:
            self.metrics[f"{self.task_name}_acc"]["y"].append(
                logs["total_correct"] / (self.n_samples)
            )

    def log_epoch_number(self, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            epoch_no (int): Epoch number to log.
        """
        self.metrics[f"{self.task_name}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{self.task_name}_acc"]["x"].append(epoch_no + 1)
        self.metrics[f"{self.task_name}_miou"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            epoch_no (int): Epoch number to print results from.
        """
        msg = "{} | Loss: {} | Accuracy: {}%".format(
            self.task_name,
            self.metrics[f"{self.task_name}_loss"]["y"][epoch_no],
            self.metrics[f"{self.task_name}_acc"]["y"][epoch_no] * 100.0,
        )

        if self.model_type == "segmentation":
            msg += " | mIoU: {}".format(
                self.metrics[f"{self.task_name}_miou"]["y"][epoch_no]
            )

        msg += "\n"
        print(msg)


class SSLMetrics(MinervaMetrics):
    """Metric logging for self-supervised models.

    Attributes:
        n_batches (dict[str, int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (tuple[int, int, int]): Shape of the input data in ``C x H x W``.
        metrics (dict[str, ~typing.Any]): Dictionary to hold the metrics to assess the model with
            for each mode of fitting.
        model_type (str): Type of the model.

    Args:
        n_batches (dict[str, int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (tuple[int, int, int]): Shape of the input data in ``C x H x W``.
        model_type (str): Optional; Type of the model.
    """

    metric_types = ["loss", "acc", "top5_acc"]
    special_metric_types = ["collapse_level", "euc_dist"]
    logger_cls = SSLLogger

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        data_size: Tuple[int, int, int],
        task_name: str,
        logger_params: Optional[Dict[str, Any]] = None,
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        model_type: str = "segmentation",
        sample_pairs: bool = False,
        **params,
    ) -> None:
        super(SSLMetrics, self).__init__(
            n_batches,
            batch_size,
            data_size,
            task_name,
            logger_params,
            record_int,
            record_float,
            writer,
            model_type=model_type,
            sample_pairs=sample_pairs,
        )

    def _calc_metrics(self, logs: Dict[str, Any]) -> None:
        """Updates metrics with epoch results.

        Args:
            logs (dict[str, ~typing.Any]): Logs of the results from the epoch of fitting to calculate metrics from.
        """
        self.metrics[f"{self.task_name}_loss"]["y"].append(
            logs["total_loss"] / self.n_batches
        )

        if self.model_type == "segmentation":
            self.metrics[f"{self.task_name}_acc"]["y"].append(
                logs["total_correct"]
                / (
                    self.n_batches
                    * self.batch_size
                    * self.data_size[1]
                    * self.data_size[2]
                )
            )
            self.metrics[f"{self.task_name}_top5_acc"]["y"].append(
                logs["total_top5"]
                / (
                    self.n_batches
                    * self.batch_size
                    * self.data_size[1]
                    * self.data_size[2]
                )
            )

        else:
            self.metrics[f"{self.task_name}_acc"]["y"].append(
                logs["total_correct"] / (self.n_samples)
            )
            self.metrics[f"{self.task_name}_top5_acc"]["y"].append(
                logs["total_top5"] / (self.n_samples)
            )

        if self.sample_pairs:
            self.metrics[f"{self.task_name}_collapse_level"]["y"].append(
                logs["collapse_level"]
            )
            self.metrics[f"{self.task_name}_euc_dist"]["y"].append(
                logs["euc_dist"] / self.n_batches
            )

    def log_epoch_number(self, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            epoch_no (int): Epoch number to log.
        """
        self.metrics[f"{self.task_name}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{self.task_name}_acc"]["x"].append(epoch_no + 1)
        self.metrics[f"{self.task_name}_top5_acc"]["x"].append(epoch_no + 1)

        if self.sample_pairs:
            self.metrics[f"{self.task_name}_collapse_level"]["x"].append(epoch_no + 1)
            self.metrics[f"{self.task_name}_euc_dist"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            epoch_no (int): Epoch number to print results from.
        """
        msg = "{} | Loss: {} | Accuracy: {}% | Top5 Accuracy: {}% ".format(
            self.task_name,
            self.metrics[f"{self.task_name}_loss"]["y"][epoch_no],
            self.metrics[f"{self.task_name}_acc"]["y"][epoch_no] * 100.0,
            self.metrics[f"{self.task_name}_top5_acc"]["y"][epoch_no] * 100.0,
        )

        if self.sample_pairs:
            msg += "\n"

            msg += "| Collapse Level: {}%".format(
                self.metrics[f"{self.task_name}_collapse_level"]["y"][epoch_no] * 100.0
            )
            msg += "| Avg. Euclidean Distance: {}".format(
                self.metrics[f"{self.task_name}_euc_dist"]["y"][epoch_no]
            )

        msg += "\n"
        print(msg)
