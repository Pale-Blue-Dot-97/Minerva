#!/usr/bin/env python
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
from typing import Dict, Any
from abc import ABC
import abc
import re as regex


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
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_batches: Dict[str, int], batch_size: int, data_size) -> None:
        super(MinervaMetrics, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size
        self.data_size = data_size

        # To be overwritten.
        self.metrics = None

    @abc.abstractmethod
    def calc_metrics(self, mode: str, logs: Dict[str, Any], **params) -> None:
        # Updates metrics with epoch results.
        pass

    @abc.abstractmethod
    def log_epoch_number(self, mode: str, epoch_no: int) -> None:
        pass

    @property
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_sub_metrics(self, pattern: str = r"[train|val]") -> Dict[str, Any]:
        reg_pattern = regex.compile(pattern)
        sub_metrics = {}
        for key in self.metrics.keys():
            if reg_pattern.search(key):
                sub_metrics[key] = self.metrics[key]

        return sub_metrics

    @abc.abstractmethod
    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        pass


class SP_Metrics(MinervaMetrics):
    def __init__(self, n_batches: Dict[str, int], batch_size: int, data_size) -> None:
        super(SP_Metrics, self).__init__(n_batches, batch_size, data_size)

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            "train_loss": {"x": [], "y": []},
            "val_loss": {"x": [], "y": []},
            "test_loss": {"x": [], "y": []},
            "train_acc": {"x": [], "y": []},
            "val_acc": {"x": [], "y": []},
            "test_acc": {"x": [], "y": []},
        }

    def calc_metrics(self, mode: str, logs: Dict[str, Any], **params) -> None:
        # Updates metrics with epoch results.
        self.metrics[f"{mode}_loss"]["y"].append(
            logs["total_loss"] / self.n_batches[mode]
        )

        if params["model_type"] == "segmentation":
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
        self.metrics[f"{mode}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_acc"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        print(
            "{} | Loss: {} | Accuracy: {}% \n".format(
                mode,
                self.metrics[f"{mode}_loss"]["y"][epoch_no],
                self.metrics[f"{mode}_acc"]["y"][epoch_no] * 100.0,
            )
        )


class SSL_Metrics(MinervaMetrics):
    def __init__(self, n_batches: Dict[str, int], batch_size: int, data_size) -> None:
        super(SSL_Metrics, self).__init__(n_batches, batch_size, data_size)

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            "train_loss": {"x": [], "y": []},
            "val_loss": {"x": [], "y": []},
            "train_acc": {"x": [], "y": []},
            "val_acc": {"x": [], "y": []},
            "train_top5_acc": {"x": [], "y": []},
            "val_top5_acc": {"x": [], "y": []},
        }

    def calc_metrics(self, mode: str, logs, **params) -> None:
        # Updates metrics with epoch results.
        self.metrics[f"{mode}_loss"]["y"].append(
            logs["total_loss"] / self.n_batches[mode]
        )

        if params["model_type"] == "segmentation":
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

    def log_epoch_number(self, mode: str, epoch_no: int) -> None:
        self.metrics[f"{mode}_loss"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_acc"]["x"].append(epoch_no + 1)
        self.metrics[f"{mode}_top5_acc"]["x"].append(epoch_no + 1)

    def print_epoch_results(self, mode: str, epoch_no: int) -> None:
        print(
            "{} | Loss: {} | Accuracy: {}% | Top5 Accuracy: {}% \n".format(
                mode,
                self.metrics[f"{mode}_loss"]["y"][epoch_no],
                self.metrics[f"{mode}_acc"]["y"][epoch_no] * 100.0,
                self.metrics[f"{mode}_top5_acc"]["y"][epoch_no] * 100.0,
            )
        )
