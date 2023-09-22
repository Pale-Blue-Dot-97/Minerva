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
from typing import Any, Dict, List, Optional, Tuple

from minerva.logger import MinervaLogger


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

    def __init__(
        self,
        n_batches: int,
        batch_size: int,
        data_size: Tuple[int, int, int],
        task_name: str,
        logger_params: Optional[Dict[str, Any]] = None,
        **params,
    ) -> None:
        super(MinervaMetrics, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.data_size = data_size

        self.model_type = params.get("model_type", "scene_classifier")
        self.sample_pairs = params.get("sample_pairs", False)

        self.logger = MinervaLogger()

        if self.sample_pairs:
            self.metric_types += self.special_metric_types

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics: Dict[str, Any] = {}
        for metric in self.metric_types:
            self.metrics[f"{task_name}_{metric}"] = {"x": [], "y": []}

    def __call__(self, logs: Dict[str, Any]) -> None:
        self.calc_metrics(logs)

    @abc.abstractmethod
    def calc_metrics(self, logs: Dict[str, Any]) -> None:
        """Updates metrics with epoch results.

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

    def __init__(
        self,
        n_batches: Dict[str, int],
        batch_size: int,
        data_size: Tuple[int, int, int],
        model_type: str = "segmentation",
        **params,
    ) -> None:
        super(SPMetrics, self).__init__(
            n_batches, batch_size, data_size, model_type=model_type
        )

    def calc_metrics(self, mode: str, logs: Dict[str, Any]) -> None:
        """Updates metrics with epoch results.

        Args:
            mode (str): Mode of model fitting.
            logs (dict[str, ~typing.Any]): Logs of the results from the epoch of fitting to calculate metrics from.
        """
        self.metrics[f"{mode}_loss"]["y"].append(
            logs["total_loss"] / self.n_batches[mode]
        )

        if self.model_type == "segmentation":
            self.metrics[f"{mode}_acc"]["y"].append(
                logs["total_correct"]
                / (
                    self.n_batches[mode]
                    * self.batch_size
                    * self.data_size[1]
                    * self.data_size[2]
                )
            )
            if logs.get("total_miou") is not None:
                self.metrics[f"{mode}_miou"]["y"].append(
                    logs["total_miou"] / (self.n_batches[mode] * self.batch_size)
                )

        else:
            self.metrics[f"{mode}_acc"]["y"].append(
                logs["total_correct"] / (self.n_batches[mode] * self.batch_size)
            )

    def log_epoch_number(self, mode: str, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            mode (str): Mode of model fitting.
            epoch_no (int): Epoch number to log.
        """
        self.metrics[f"{mode}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_acc"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_miou"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            mode (str): Mode of fitting to print results from.
            epoch_no (int): Epoch number to print results from.
        """
        msg = "{} | Loss: {} | Accuracy: {}%".format(
            mode,
            self.metrics[f"{mode}_loss"]["y"][epoch_no],
            self.metrics[f"{mode}_acc"]["y"][epoch_no] * 100.0,
        )

        if self.model_type == "segmentation":
            msg += " | mIoU: {}".format(self.metrics[f"{mode}_miou"]["y"][epoch_no])

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

    def __init__(
        self,
        n_batches: Dict[str, int],
        batch_size: int,
        data_size: Tuple[int, int, int],
        model_type: str = "segmentation",
        sample_pairs: bool = False,
        **params,
    ) -> None:
        super(SSLMetrics, self).__init__(
            n_batches,
            batch_size,
            data_size,
            model_type=model_type,
            sample_pairs=sample_pairs,
        )

    def calc_metrics(self, mode: str, logs) -> None:
        """Updates metrics with epoch results.

        Args:
            mode (str): Mode of model fitting.
            logs (dict[str, ~typing.Any]): Logs of the results from the epoch of fitting to calculate metrics from.
        """
        self.metrics[f"{mode}_loss"]["y"].append(
            logs["total_loss"] / self.n_batches[mode]
        )

        if self.model_type == "segmentation":
            self.metrics[f"{mode}_acc"]["y"].append(
                logs["total_correct"]
                / (
                    self.n_batches[mode]
                    * self.batch_size
                    * self.data_size[1]
                    * self.data_size[2]
                )
            )
            self.metrics[f"{mode}_top5_acc"]["y"].append(
                logs["total_top5"]
                / (
                    self.n_batches[mode]
                    * self.batch_size
                    * self.data_size[1]
                    * self.data_size[2]
                )
            )

        else:
            self.metrics[f"{mode}_acc"]["y"].append(
                logs["total_correct"] / (self.n_batches[mode] * self.batch_size)
            )
            self.metrics[f"{mode}_top5_acc"]["y"].append(
                logs["total_top5"] / (self.n_batches[mode] * self.batch_size)
            )

        if self.sample_pairs and mode == "train":
            self.metrics[f"{mode}_collapse_level"]["y"].append(logs["collapse_level"])
            self.metrics[f"{mode}_euc_dist"]["y"].append(
                logs["euc_dist"] / self.n_batches[mode]
            )

    def log_epoch_number(self, mode: str, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            mode (str): Mode of model fitting.
            epoch_no (int): Epoch number to log.
        """
        self.metrics[f"{mode}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_acc"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_top5_acc"]["x"].append(epoch_no + 1)

        if self.sample_pairs and mode == "train":
            self.metrics[f"{mode}_collapse_level"]["x"].append(epoch_no + 1)
            self.metrics[f"{mode}_euc_dist"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            mode (str): Mode of fitting to print results from.
            epoch_no (int): Epoch number to print results from.
        """
        msg = "{} | Loss: {} | Accuracy: {}% | Top5 Accuracy: {}% ".format(
            mode,
            self.metrics[f"{mode}_loss"]["y"][epoch_no],
            self.metrics[f"{mode}_acc"]["y"][epoch_no] * 100.0,
            self.metrics[f"{mode}_top5_acc"]["y"][epoch_no] * 100.0,
        )

        if self.sample_pairs and mode == "train":
            msg += "\n"

            msg += "| Collapse Level: {}%".format(
                self.metrics[f"{mode}_collapse_level"]["y"][epoch_no] * 100.0
            )
            msg += "| Avg. Euclidean Distance: {}".format(
                self.metrics[f"{mode}_euc_dist"]["y"][epoch_no]
            )

        msg += "\n"
        print(msg)
