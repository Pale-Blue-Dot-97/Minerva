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
"""Module to calculate the metrics of a model's fitting."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
from typing import Any, Dict, Literal, Tuple

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
class MinervaMetrics(ABC):
    """Abstract class for metric logging within the :mod:`minerva` framework.

    Attributes:
        n_batches (Dict[Literal["train", "val", "test"], int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (Tuple[int, int, int]): Shape of the input data in C x H x W.
        metrics (Dict[str, Any]): Dictionary to hold the metrics to assess the model with for each mode of fitting.
        model_type (str): Type of the model.

    Args:
        n_batches (Dict[Literal["train", "val", "test"], int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (Tuple[int, int, int]): Shape of the input data in C x H x W.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        n_batches: Dict[Literal["train", "val", "test"], int],
        batch_size: int,
        data_size: Tuple[int, int, int],
        **params,
    ) -> None:
        super(MinervaMetrics, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.data_size = data_size

        # To be overwritten.
        self.metrics: Dict[str, Any] = {}

        self.model_type = params.get("model_type", "scene_classifier")

    def __call__(
        self, mode: Literal["train", "val", "test"], logs: Dict[str, Any]
    ) -> None:
        self.calc_metrics(mode, logs)

    @abc.abstractmethod
    def calc_metrics(
        self, mode: Literal["train", "val", "test"], logs: Dict[str, Any]
    ) -> None:
        """Updates metrics with epoch results.

        Args:
            mode (Literal["train", "val", "test"]): Mode of model fitting.
            logs (Dict[str, Any]): Logs of the results from the epoch of fitting to calculate metrics from.
        """
        pass

    @abc.abstractmethod
    def log_epoch_number(
        self, mode: Literal["train", "val", "test"], epoch_no: int
    ) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            mode (Literal["train", "val", "test"]): Mode of model fitting.
            epoch_no (int): Epoch number to log.
        """
        pass

    @property
    def get_metrics(self) -> Dict[str, Any]:
        """Get the ``metrics`` dictionary.

        Returns:
            Dict[str, Any]: Metrics dictionary.
        """
        return self.metrics

    def get_sub_metrics(
        self, pattern: Tuple[str, ...] = ("train", "val")
    ) -> Dict[str, Any]:
        """Gets a subset of the metrics dictionary with keys containing strings in the pattern.

        Useful for getting the train and validation metrics for plotting for example.

        Args:
            pattern (Tuple[str, ...]): Optional; Strings to pattern match the metric keys to be returned.
                Defaults to ("train", "val").

        Returns:
            Dict[str, Any]: Subset of ``metrics`` with keys that contained strings in ``pattern``.
        """
        sub_metrics = {}
        for key in self.metrics.keys():
            if key.split("_")[0] in pattern:
                sub_metrics[key] = self.metrics[key]

        return sub_metrics

    @abc.abstractmethod
    def print_epoch_results(
        self, mode: Literal["train", "val", "test"], epoch_no: int
    ) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            mode (Literal["train", "val", "test"]): Mode of fitting to print results from.
            epoch_no (int): Epoch number to print results from.
        """
        pass


class SP_Metrics(MinervaMetrics):
    """Metric logging for supervised models.

    Attributes:
        n_batches (Dict[Literal["train", "val", "test"], int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (Tuple[int, int, int]): Shape of the input data in C x H x W.
        metrics (Dict[str, Any]): Dictionary to hold the metrics to assess the model with for each mode of fitting.
        model_type (str): Type of the model.

    Args:
        n_batches (Dict[Literal["train", "val", "test"], int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (Tuple[int, int, int]): Shape of the input data in C x H x W.
        model_type (str): Optional; Type of the model.
    """

    def __init__(
        self,
        n_batches: Dict[Literal["train", "val", "test"], int],
        batch_size: int,
        data_size: Tuple[int, int, int],
        model_type: str = "segmentation",
    ) -> None:
        super(SP_Metrics, self).__init__(
            n_batches, batch_size, data_size, model_type=model_type
        )

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            "train_loss": {"x": [], "y": []},
            "val_loss": {"x": [], "y": []},
            "test_loss": {"x": [], "y": []},
            "train_acc": {"x": [], "y": []},
            "val_acc": {"x": [], "y": []},
            "test_acc": {"x": [], "y": []},
        }

    def calc_metrics(
        self, mode: Literal["train", "val", "test"], logs: Dict[str, Any]
    ) -> None:
        """Updates metrics with epoch results.

        Args:
            mode (Literal["train", "val", "test"]): Mode of model fitting.
            logs (Dict[str, Any]): Logs of the results from the epoch of fitting to calculate metrics from.
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
        else:
            self.metrics[f"{mode}_acc"]["y"].append(
                logs["total_correct"] / (self.n_batches[mode] * self.batch_size)
            )

    def log_epoch_number(self, mode: str, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            mode (Literal["train", "val", "test"]): Mode of model fitting.
            epoch_no (int): Epoch number to log.
        """
        self.metrics[f"{mode}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_acc"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            mode (Literal["train", "val", "test"]): Mode of fitting to print results from.
            epoch_no (int): Epoch number to print results from.
        """
        print(
            "{} | Loss: {} | Accuracy: {}% \n".format(
                mode,
                self.metrics[f"{mode}_loss"]["y"][epoch_no],
                self.metrics[f"{mode}_acc"]["y"][epoch_no] * 100.0,
            )
        )


class SSL_Metrics(MinervaMetrics):
    """Metric logging for self-supervised models.

    Attributes:
        n_batches (Dict[Literal["train", "val", "test"], int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (Tuple[int, int, int]): Shape of the input data in C x H x W.
        metrics (Dict[str, Any]): Dictionary to hold the metrics to assess the model with for each mode of fitting.
        model_type (str): Type of the model.

    Args:
        n_batches (Dict[Literal["train", "val", "test"], int]): Dictionary of the number of batches in each mode of fitting.
        batch_size (int): Batch size.
        data_size (Tuple[int, int, int]): Shape of the input data in C x H x W.
        model_type (str): Optional; Type of the model.
    """

    def __init__(
        self,
        n_batches: Dict[Literal["train", "val", "test"], int],
        batch_size: int,
        data_size: Tuple[int, int, int],
        model_type: str = "segmentation",
    ) -> None:
        super(SSL_Metrics, self).__init__(
            n_batches, batch_size, data_size, model_type=model_type
        )

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            "train_loss": {"x": [], "y": []},
            "val_loss": {"x": [], "y": []},
            "train_acc": {"x": [], "y": []},
            "val_acc": {"x": [], "y": []},
            "train_top5_acc": {"x": [], "y": []},
            "val_top5_acc": {"x": [], "y": []},
        }

        if model_type == "siamese":
            self.metrics["train_collapse_level"] = {"x": [], "y": []}
            self.metrics["val_collapse_level"] = {"x": [], "y": []}

    def calc_metrics(self, mode: Literal["train", "val", "test"], logs) -> None:
        """Updates metrics with epoch results.

        Args:
            mode (Literal["train", "val", "test"]): Mode of model fitting.
            logs (Dict[str, Any]): Logs of the results from the epoch of fitting to calculate metrics from.
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

        if self.model_type == "siamese":
            self.metrics[f"{mode}_collapse_level"]["y"].append(logs["collapse_level"])

    def log_epoch_number(self, mode: str, epoch_no: int) -> None:
        """Logs the epoch number to ``metrics``.

        Args:
            mode (Literal["train", "val", "test"]): Mode of model fitting.
            epoch_no (int): Epoch number to log.
        """
        self.metrics[f"{mode}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_acc"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_top5_acc"]["x"].append(epoch_no + 1)

        if self.model_type == "siamese":
            self.metrics[f"{mode}_collapse_level"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        """Prints the results from an epoch to ``stdout``.

        Args:
            mode (Literal["train", "val", "test"]): Mode of fitting to print results from.
            epoch_no (int): Epoch number to print results from.
        """
        msg = "{} | Loss: {} | Accuracy: {}% | Top5 Accuracy: {}% ".format(
            mode,
            self.metrics[f"{mode}_loss"]["y"][epoch_no],
            self.metrics[f"{mode}_acc"]["y"][epoch_no] * 100.0,
            self.metrics[f"{mode}_top5_acc"]["y"][epoch_no] * 100.0,
        )

        if self.model_type == "siamese":
            msg += "| Collapse Level: {}%".format(
                self.metrics[f"{mode}_collapse_level"]["y"][epoch_no] * 100.0
            )

        msg += "\n"
        print(msg)
