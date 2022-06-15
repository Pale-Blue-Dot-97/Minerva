import random

import pytest

from minerva.metrics import MinervaMetrics, SP_Metrics, SSL_Metrics


def test_minervametrics() -> None:
    n_batches = {"train": 12, "val": 4, "test": 2}
    with pytest.raises(TypeError):
        _ = MinervaMetrics(n_batches, 16, (4, 224, 224), model_type="scene_classifier")


def test_sp_metrics() -> None:
    n_batches = {"train": 12, "val": 4, "test": 2}
    metric_logger_1 = SP_Metrics(
        n_batches, 16, (4, 224, 224), model_type="segmentation"
    )
    metric_logger_2 = SP_Metrics(
        n_batches, 16, (4, 224, 224), model_type="scene_classifier"
    )

    for mode in n_batches.keys():
        logs_1 = {
            "total_loss": random.random(),
            "total_correct": random.random(),
        }

        logs_2 = {
            "total_loss": random.random(),
            "total_correct": random.random(),
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

        assert metrics_1[f"{mode}_loss"] == pytest.approx(correct_loss_1)
        assert metrics_1[f"{mode}_acc"] == pytest.approx(correct_acc_1)

        assert metrics_2[f"{mode}_loss"] == pytest.approx(correct_loss_1)
        assert metrics_2[f"{mode}_acc"] == pytest.approx(correct_acc_2)

        if mode in ("train", "val"):
            sub_metrics_1 = metric_logger_1.get_sub_metrics()
            sub_metrics_2 = metric_logger_2.get_sub_metrics()

            assert sub_metrics_1[f"{mode}_loss"] == pytest.approx(correct_loss_1)
            assert sub_metrics_1[f"{mode}_acc"] == pytest.approx(correct_acc_1)

            assert sub_metrics_2[f"{mode}_loss"] == pytest.approx(correct_loss_1)
            assert sub_metrics_2[f"{mode}_acc"] == pytest.approx(correct_acc_2)


def test_ssl_metrics() -> None:
    n_batches = {"train": 12, "val": 4}
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
