# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
r"""Tests for :mod:`minerva.metrics`.
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
import random
from typing import Dict, List

import pytest

from minerva.metrics import MinervaMetrics, SPMetrics, SSLMetrics


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_minervametrics() -> None:
    assert issubclass(SPMetrics, MinervaMetrics)


def test_sp_metrics() -> None:
    def get_random_logs() -> Dict[str, float]:
        logs = {
            "total_loss": random.random(),
            "total_correct": random.random(),
            "total_miou": random.random(),
        }

        return logs

    n_epochs = 2

    n_batches: Dict[str, int] = {
        "train": 12,
        "val": 4,
        "test": 2,
    }

    metric_loggers: List[MinervaMetrics] = []

    metric_loggers.append(
        SPMetrics(n_batches, 16, (4, 224, 224), model_type="segmentation")
    )
    metric_loggers.append(
        SPMetrics(n_batches, 16, (4, 224, 224), model_type="scene_classifier")
    )

    epochs = [k + 1 for k in range(n_epochs)]

    for mode in n_batches.keys():
        logs = [get_random_logs() for i in range(n_epochs)]

        correct_loss_1 = {
            "x": epochs,
            "y": [log["total_loss"] / n_batches[mode] for log in logs],
        }

        correct_loss = [correct_loss_1, correct_loss_1]

        correct_acc = [
            {
                "x": epochs,
                "y": [
                    log["total_correct"] / (n_batches[mode] * 16 * 224 * 224)
                    for log in logs
                ],
            },
            {
                "x": epochs,
                "y": [log["total_correct"] / (n_batches[mode] * 16) for log in logs],
            },
        ]

        correct_miou = {
            "x": epochs,
            "y": [log["total_miou"] / (n_batches[mode] * 16) for log in logs],
        }

        for i, metric_logger in enumerate(metric_loggers):
            for j in range(len(logs)):
                metric_logger(mode, logs[j])
                metric_logger.log_epoch_number(mode, j)
                metric_logger.print_epoch_results(mode, j)

            metrics = metric_loggers[i].get_metrics

            assert metrics[f"{mode}_loss"] == pytest.approx(correct_loss[i])
            assert metrics[f"{mode}_acc"] == pytest.approx(correct_acc[i])

            if i == 0:
                assert metrics[f"{mode}_miou"] == pytest.approx(correct_miou)

            if mode in ("train", "val"):
                sub_metrics = metric_logger.get_sub_metrics()

                assert sub_metrics[f"{mode}_loss"] == pytest.approx(correct_loss[i])
                assert sub_metrics[f"{mode}_acc"] == pytest.approx(correct_acc[i])

                if i == 0:
                    assert sub_metrics[f"{mode}_miou"] == pytest.approx(correct_miou)


def test_ssl_metrics() -> None:
    def get_random_logs() -> Dict[str, float]:
        logs = {
            "total_loss": random.random(),
            "total_correct": random.random(),
            "total_top5": random.random(),
            "collapse_level": random.random(),
            "euc_dist": random.random(),
        }

        return logs

    n_epochs = 2

    epochs = [k + 1 for k in range(n_epochs)]

    n_batches: Dict[str, int] = {"train": 12, "val": 4}

    metric_loggers: List[MinervaMetrics] = []
    metric_loggers.append(
        SSLMetrics(
            n_batches, 16, (4, 224, 224), model_type="segmentation", sample_pairs=True
        )
    )
    metric_loggers.append(
        SSLMetrics(
            n_batches,
            16,
            (4, 224, 224),
            model_type="scene_classifier",
            sample_pairs=True,
        )
    )

    for mode in n_batches.keys():
        logs = [get_random_logs(), get_random_logs()]

        correct_loss = {
            "x": epochs,
            "y": [log["total_loss"] / n_batches[mode] for log in logs],
        }

        correct_acc = []
        correct_acc.append(
            {
                "x": epochs,
                "y": [
                    log["total_correct"] / (n_batches[mode] * 16 * 224 * 224)
                    for log in logs
                ],
            }
        )

        correct_acc.append(
            {
                "x": epochs,
                "y": [log["total_correct"] / (n_batches[mode] * 16) for log in logs],
            }
        )

        correct_top5 = []
        correct_top5.append(
            {
                "x": epochs,
                "y": [
                    log["total_top5"] / (n_batches[mode] * 16 * 224 * 224)
                    for log in logs
                ],
            }
        )

        correct_top5.append(
            {
                "x": epochs,
                "y": [log["total_top5"] / (n_batches[mode] * 16) for log in logs],
            }
        )

        correct_collapse_level = {
            "x": epochs,
            "y": [log["collapse_level"] for log in logs],
        }

        correct_euc_dist = {
            "x": epochs,
            "y": [log["euc_dist"] / n_batches[mode] for log in logs],
        }

        for i, metric_logger in enumerate(metric_loggers):
            for j in range(len(logs)):
                metric_logger(mode, logs[j])
                metric_logger.log_epoch_number(mode, j)
                metric_logger.print_epoch_results(mode, j)

            metrics = metric_logger.get_metrics

            assert metrics[f"{mode}_loss"] == pytest.approx(correct_loss)
            assert metrics[f"{mode}_acc"] == pytest.approx(correct_acc[i])
            assert metrics[f"{mode}_top5_acc"] == pytest.approx(correct_top5[i])

            if mode == "train":
                assert metrics[f"{mode}_collapse_level"] == pytest.approx(
                    correct_collapse_level
                )
                assert metrics[f"{mode}_euc_dist"] == pytest.approx(correct_euc_dist)

            sub_metrics = metric_logger.get_sub_metrics()

            assert sub_metrics[f"{mode}_loss"] == pytest.approx(correct_loss)
            assert sub_metrics[f"{mode}_acc"] == pytest.approx(correct_acc[i])
            assert sub_metrics[f"{mode}_top5_acc"] == pytest.approx(correct_top5[i])

            if mode == "train":
                assert sub_metrics[f"{mode}_collapse_level"] == pytest.approx(
                    correct_collapse_level
                )
                assert sub_metrics[f"{mode}_euc_dist"] == pytest.approx(
                    correct_euc_dist
                )
