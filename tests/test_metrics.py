import random
from typing import Dict, List

import pytest

from minerva.metrics import MinervaMetrics, SP_Metrics, SSL_Metrics


def test_minervametrics() -> None:
    assert issubclass(SP_Metrics, MinervaMetrics)


def test_sp_metrics() -> None:
    def get_random_logs() -> Dict[str, float]:
        logs = {
            "total_loss": random.random(),
            "total_correct": random.random(),
            "total_miou": random.random(),
        }

        return logs

    n_batches: Dict[str, int] = {
        "train": 12,
        "val": 4,
        "test": 2,
    }

    metric_loggers: List[MinervaMetrics] = []

    metric_loggers.append(
        SP_Metrics(n_batches, 16, (4, 224, 224), model_type="segmentation")
    )
    metric_loggers.append(
        SP_Metrics(n_batches, 16, (4, 224, 224), model_type="scene_classifier")
    )

    for mode in n_batches.keys():
        logs = [get_random_logs(), get_random_logs()]

        epochs = [j + 1 for j in range(len(logs))]
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
    n_batches: Dict[str, int] = {"train": 12, "val": 4}
    metric_logger_1 = SSL_Metrics(
        n_batches, 16, (4, 224, 224), model_type="segmentation"
    )
    metric_logger_2 = SSL_Metrics(
        n_batches, 16, (4, 224, 224), model_type="scene_classifier"
    )

    for mode in n_batches.keys():
        logs_1 = {
            "total_loss": random.random(),
            "total_correct": random.random(),
            "total_top5": random.random(),
        }

        logs_2 = {
            "total_loss": random.random(),
            "total_correct": random.random(),
            "total_top5": random.random(),
        }

        metric_logger_1(mode, logs_1)
        metric_logger_1.log_epoch_number(mode, 0)
        metric_logger_1.print_epoch_results(mode, 0)

        metric_logger_1(mode, logs_2)
        metric_logger_1.log_epoch_number(mode, 1)
        metric_logger_1.print_epoch_results(mode, 1)

        metric_logger_2(mode, logs_1)
        metric_logger_2.log_epoch_number(mode, 0)
        metric_logger_2.print_epoch_results(mode, 0)

        metric_logger_2(mode, logs_2)
        metric_logger_2.log_epoch_number(mode, 1)
        metric_logger_2.print_epoch_results(mode, 1)

        metrics_1 = metric_logger_1.get_metrics
        metrics_2 = metric_logger_2.get_metrics

        correct_loss_1 = {
            "x": [1, 2],
            "y": [
                logs_1["total_loss"] / n_batches[mode],
                logs_2["total_loss"] / n_batches[mode],
            ],
        }
        correct_acc_1 = {
            "x": [1, 2],
            "y": [
                logs_1["total_correct"] / (n_batches[mode] * 16 * 224 * 224),
                logs_2["total_correct"] / (n_batches[mode] * 16 * 224 * 224),
            ],
        }

        correct_acc_2 = {
            "x": [1, 2],
            "y": [
                logs_1["total_correct"] / (n_batches[mode] * 16),
                logs_2["total_correct"] / (n_batches[mode] * 16),
            ],
        }
        correct_top5_1 = {
            "x": [1, 2],
            "y": [
                logs_1["total_top5"] / (n_batches[mode] * 16 * 224 * 224),
                logs_2["total_top5"] / (n_batches[mode] * 16 * 224 * 224),
            ],
        }

        correct_top5_2 = {
            "x": [1, 2],
            "y": [
                logs_1["total_top5"] / (n_batches[mode] * 16),
                logs_2["total_top5"] / (n_batches[mode] * 16),
            ],
        }

        assert metrics_1[f"{mode}_loss"] == pytest.approx(correct_loss_1)
        assert metrics_1[f"{mode}_acc"] == pytest.approx(correct_acc_1)
        assert metrics_1[f"{mode}_top5_acc"] == pytest.approx(correct_top5_1)

        assert metrics_2[f"{mode}_loss"] == pytest.approx(correct_loss_1)
        assert metrics_2[f"{mode}_acc"] == pytest.approx(correct_acc_2)
        assert metrics_2[f"{mode}_top5_acc"] == pytest.approx(correct_top5_2)

        sub_metrics_1 = metric_logger_1.get_sub_metrics()
        sub_metrics_2 = metric_logger_2.get_sub_metrics()

        assert sub_metrics_1[f"{mode}_loss"] == pytest.approx(correct_loss_1)
        assert sub_metrics_1[f"{mode}_acc"] == pytest.approx(correct_acc_1)
        assert sub_metrics_1[f"{mode}_top5_acc"] == pytest.approx(correct_top5_1)

        assert sub_metrics_2[f"{mode}_loss"] == pytest.approx(correct_loss_1)
        assert sub_metrics_2[f"{mode}_acc"] == pytest.approx(correct_acc_2)
        assert sub_metrics_2[f"{mode}_top5_acc"] == pytest.approx(correct_top5_2)
