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
"""Loggers to handle the logging from each step of a task."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
from __future__ import annotations

__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "MinervaStepLogger",
    "SupervisedGeoStepLogger",
    "SSLStepLogger",
    "KNNStepLogger",
    "get_logger",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
import math
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    SupportsFloat,
    Tuple,
    Union,
)

import mlflow
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import jaccard_score
from torch import Tensor
from torchmetrics.regression.cosine_similarity import CosineSimilarity

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter
else:  # pragma: no cover
    SummaryWriter = None

from torchgeo.datasets.utils import BoundingBox
from wandb.sdk.wandb_run import Run

from minerva.utils import utils
from minerva.utils.utils import check_substrings_in_string, func_by_str

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
_tensorflow_exist = utils.check_optional_import_exist("tensorflow")
TENSORBOARD_WRITER: Optional[Callable[..., Any]]
try:
    TENSORBOARD_WRITER = utils._optional_import(
        "torch.utils.tensorboard.writer",
        name="SummaryWriter",
        package="tensorflow",
    )
except ImportError as err:  # pragma: no cover
    print(err)
    print("Disabling TensorBoard logging")
    TENSORBOARD_WRITER = None


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaStepLogger(ABC):
    """Base abstract class for all :mod:`minerva` step logger classes to ensure intercompatibility with
    :class:`~trainer.Trainer`.

    Attributes:
        record_int (bool): Whether to record the integer values from an epoch of model fitting.
        record_float (bool): Whether to record the floating point values from an epoch of model fitting.
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        logs (dict[str, ~typing.Any]): Dictionary to hold the logs from the epoch.
            Logs should be more lightweight than ``results``.
        results (dict[str, ~typing.Any]): Dictionary to hold the results from the epoch.

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to ``True``.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to ``False``.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wandb_run.Run): Optional; Writer object
            from :mod:`tensorboard`, a :mod:`wandb` :class:`~wandb.sdk.wandb_run.Run` object or ``None``.

    .. versionadded:: 0.27
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        task_name: str,
        n_batches: int,
        batch_size: int,
        output_size: Tuple[int, int],
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        model_type: str = "",
        **kwargs,
    ) -> None:
        super(MinervaStepLogger, self).__init__()
        self.record_int = record_int
        self.record_float = record_float
        self.n_batches = n_batches
        self.output_size = output_size
        self.batch_size = batch_size

        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover
            self.batch_size = batch_size // dist.get_world_size()  # type: ignore[attr-defined]

        self.n_samples = self.batch_size * self.n_batches

        self.task_name = task_name
        self.writer = writer

        self.model_type = model_type

        self.logs: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def __call__(self, step_num: int, loss: Tensor, *args) -> None:
        """Call :meth:`log`.

        Args:
            step_num (int): The global step number of for the mode of model fitting.
            loss (~torch.Tensor): Loss from this step of model fitting.

        Returns:
            None
        """
        self.log(step_num, loss, *args)

    @abc.abstractmethod
    def log(
        self,
        step_num: int,
        loss: Tensor,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Abstract logging method, the core functionality of a logger. Must be overwritten.

        Args:
            step_num (int): The global step number of for the mode of model fitting.
            loss (~torch.Tensor): Loss from this step of model fitting.
            z (~torch.Tensor): Optional; Output tensor from the model.
            y (~torch.Tensor): Optional; Labels to assess model output against.
            bbox (~torchgeo.datasets.utils.BoundingBox): Optional; Bounding boxes of the input samples.

        Returns:
            None
        """
        pass  # pragma: no cover

    def write_metric(
        self, key: str, value: SupportsFloat, step_num: Optional[int] = None
    ):
        """Write metric values to logging backends after calculation.

        Args:
            key (str): Key for the metric that ``value`` belongs to.
            value (SupportsFloat): Metric to write to logger.
            step_num (int): Optional; Global step number for this ``mode`` of fitting.

        """
        # TODO: Are values being reduced across nodes / logged from rank 0?
        if self.writer:
            if _tensorflow_exist:
                if (
                    isinstance(
                        self.writer, utils.extract_class_type(TENSORBOARD_WRITER)
                    )
                    and self.writer
                ):
                    self.writer.add_scalar(  # type: ignore[attr-defined]
                        tag=f"{self.task_name}_{key}",
                        scalar_value=value,  # type: ignore[attr-defined]
                        global_step=step_num,
                    )
            if isinstance(self.writer, Run):
                self.writer.log(
                    {
                        f"{self.task_name}/step": step_num,
                        f"{self.task_name}/{key}": value,
                    }
                )

        if mlflow.active_run():
            # If running in Azure Machine Learning, tracking URI / experiment ID set already
            # https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-cli-runs?tabs=python%2Cmlflow#creating-a-training-routine  # noqa: E501
            mlflow.log_metric(key, value)  # pragma: no cover

    @property
    def get_logs(self) -> Dict[str, Any]:
        """Gets the logs dictionary.

        Returns:
            dict[str, ~typing.Any]: Log dictionary of the logger.
        """
        return self.logs

    @property
    def get_results(self) -> Dict[str, Any]:
        """Gets the results dictionary.

        Returns:
            dict[str, ~typing.Any]: Results dictionary of the logger.
        """
        return self.results


class SupervisedGeoStepLogger(MinervaStepLogger):
    """Logger designed for supervised learning using :mod:`torchgeo` datasets.

    Attributes:
        logs (dict[str, ~typing.Any]): The main logs from the KNN with these metrics:

            * ``batch_num``
            * ``total_loss``
            * ``total_correct``
            * ``total_top5``

        results (dict[str, ~typing.Any]): Hold these additional, full results from the KNN:

            * ``y``
            * ``z``
            * ``probs``
            * ``ids``
            * ``bounds``

        calc_miou (bool): Activates the calculating and logging of :term:`MIoU` for segmentation models.
            Places the metric in the ``total_miou`` key of ``logs``.

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        out_shape (int | tuple[int, ...]): Shape of the model output.
        n_classes (int): Number of classes in dataset.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to ``True``.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to ``False``.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wandb_run.Run): Optional; Writer object
            from :mod:`tensorboard`, a :mod:`wandb` :class:`~wandb.sdk.wandb_run.Run` object or ``None``.

    Raises:
        MemoryError: If trying to allocate memory to hold the probabilites of predictions
            from the model exceeds capacity.
        MemoryError: If trying to allocate memory to hold the bounding boxes of samples would exceed capacity.

    .. versionadded:: 0.27
    """

    def __init__(
        self,
        task_name: str,
        n_batches: int,
        batch_size: int,
        output_size: Tuple[int, int],
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        model_type: str = "",
        n_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super(SupervisedGeoStepLogger, self).__init__(
            task_name,
            n_batches,
            batch_size,
            output_size,
            record_int,
            record_float,
            writer,
            model_type,
        )
        if n_classes is None:
            raise ValueError("`n_classes` must be specified for this type of logger!")

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
        self.calc_miou = (
            True
            if check_substrings_in_string(self.model_type, "segmentation")
            else False
        )

        if self.calc_miou:
            self.logs["total_miou"] = 0.0

        # Allocate memory for the integer values to be recorded.
        if self.record_int:
            int_log_shape: Tuple[int, ...]
            if check_substrings_in_string(self.model_type, "scene classifier"):
                int_log_shape = (self.n_batches, self.batch_size)
            else:
                int_log_shape = (self.n_batches, self.batch_size, *self.output_size)

            self.results["z"] = np.empty(int_log_shape, dtype=np.uint8)
            self.results["y"] = np.empty(int_log_shape, dtype=np.uint8)

        # Allocate memory for the floating point values to be recorded.
        if self.record_float:
            float_log_shape: Tuple[int, ...]
            if check_substrings_in_string(self.model_type, "scene classifier"):
                float_log_shape = (self.n_batches, self.batch_size, n_classes)
            else:
                float_log_shape = (
                    self.n_batches,
                    self.batch_size,
                    n_classes,
                    *self.output_size,
                )

            try:
                self.results["probs"] = np.empty(float_log_shape, dtype=np.float16)
            except MemoryError:  # pragma: no cover
                raise MemoryError(
                    "Dataset too large to record probabilities of predicted classes!"
                )

            try:
                self.results["bounds"] = np.empty(
                    (self.n_batches, self.batch_size), dtype=object
                )
            except MemoryError:  # pragma: no cover
                raise MemoryError(
                    "Dataset too large to record bounding boxes of samples!"
                )

    def log(
        self,
        step_num: int,
        loss: Tensor,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Logs the outputs and results from a step of model fitting. Overwrites abstract method.

        Args:
            step_num (int): The global step number of for the mode of model fitting.
            loss (~torch.Tensor): Loss from this step of model fitting.
            z (~torch.Tensor): Output tensor from the model.
            y (~torch.Tensor): Labels to assess model output against.
            bbox (~torchgeo.datasets.utils.BoundingBox): Bounding boxes of the input samples.

        Returns:
            None
        """

        assert z is not None
        assert y is not None

        if self.record_int:
            # Arg max the estimated probabilities and add to predictions.
            self.results["z"][self.logs["batch_num"]] = torch.argmax(z, 1).cpu().numpy()  # type: ignore[attr-defined]

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
        correct = (torch.argmax(z, 1) == y).sum().item()  # type: ignore[attr-defined]

        # Adds loss and correct predictions to logs.
        self.logs["total_loss"] += ls
        self.logs["total_correct"] += correct

        if self.calc_miou:
            assert y is not None
            y_true = y.detach().cpu().numpy()
            y_pred = torch.argmax(z, 1).detach().cpu().numpy()  # type: ignore[attr-defined]
            miou = 0.0
            for i in range(len(y)):
                miou += float(
                    jaccard_score(
                        y_true[i].flatten(), y_pred[i].flatten(), average="macro"
                    )
                )  # noqa: E501 type: ignore[attr-defined]
            self.logs["total_miou"] += miou

            self.write_metric("miou", miou / len(y), step_num=step_num)

        # Writes loss and correct predictions to the writer.
        self.write_metric("loss", ls, step_num=step_num)
        self.write_metric("acc", correct / len(torch.flatten(y)), step_num=step_num)

        # Adds 1 to batch number (step number).
        self.logs["batch_num"] += 1


class KNNStepLogger(MinervaStepLogger):
    """Logger specifically designed for use with the KNN validation in
    :meth:`trainer.Trainer.weighted_knn_validation`.

    Attributes:
        logs (dict[str, ~typing.Any]): The main logs from the KNN with these metrics:

            * ``batch_num``
            * ``total_loss``
            * ``total_correct``
            * ``total_top5``

        results (dict[str, ~typing.Any]): Hold these additional, full results from the KNN:

            * ``y``
            * ``z``
            * ``probs``
            * ``ids``
            * ``bounds``

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to ``True``.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to ``False``.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wand_run.Run): Optional; Writer object
            from :mod:`tensorboard`, a :mod:`wandb` :class:`~wandb.sdk.wandb_run.Run` object or ``None``.

    .. versionadded:: 0.27
    """

    def __init__(
        self,
        task_name: str,
        n_batches: int,
        batch_size: int,
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        model_type: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            task_name,
            n_batches,
            batch_size,
            record_int=record_int,
            record_float=record_float,
            writer=writer,
            model_type=model_type,
            **kwargs,
        )

        self.logs: Dict[str, Any] = {
            "batch_num": 0,
            "total_loss": 0.0,
            "total_correct": 0.0,
            "total_top5": 0.0,
        }

        self.results: Dict[str, Any] = {
            "y": None,
            "z": None,
            "probs": None,
            "ids": [],
            "bounds": None,
        }

    def log(
        self,
        step_num: int,
        loss: Tensor,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        assert isinstance(z, Tensor)
        assert isinstance(y, Tensor)

        # Extract loss.
        ls = loss.item()

        # Calculate the top-1 (standard) accuracy.
        top1 = torch.sum((z[:, :1] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        # Calculate the top-5 accuracy
        top5 = torch.sum((z[:, :5] == y.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        # Add results to logs.
        self.logs["total_loss"] += ls
        self.logs["total_correct"] += top1
        self.logs["total_top5"] += top5

        # Write results to the writer.
        self.write_metric("loss", loss, step_num)
        self.write_metric("acc", top1, step_num)
        self.write_metric("top5", top5, step_num)

        # Adds 1 to batch number (step number).
        self.logs["batch_num"] += 1


class SSLStepLogger(MinervaStepLogger):
    """Logger designed for self-supervised learning.

    Attributes:
        logs (dict[str, ~typing.Any]): Dictionary to hold these logged metrics:

            * ``batch_num``
            * ``total_loss``
            * ``total_correct``
            * ``total_top5``
            * ``avg_loss``
            * ``avg_output_std``

        collapse_level (bool): Adds calculation and logging of the :term:`collapse level` to the metrics.
            Only to be used with Siamese type models.
        euclidean (bool): Adds calculation and logging of the :term:`euclidean distance` to the metrics.
            Only to be used with Siamese type models.

    Args:
        n_batches (int): Number of batches in the epoch.
        batch_size (int): Size of the batch.
        n_samples (int): Total number of samples in the epoch.
        out_shape (tuple[int, ...]): Shape of the model output.
        n_classes (int): Number of classes in dataset.
        record_int (bool): Optional; Whether to record the integer values from an epoch of model fitting.
            Defaults to ``True``.
        record_float (bool): Optional; Whether to record the floating point values from an epoch of model fitting.
            Defaults to ``False``.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wand_run.Run): Optional; Writer object
            from :mod:`tensorboard`, a :mod:`wandb` :class:`~wandb.sdk.wandb_run.Run` object or ``None``.

    .. versionadded:: 0.27
    """

    def __init__(
        self,
        task_name: str,
        n_batches: int,
        batch_size: int,
        output_size: Tuple[int, int],
        record_int: bool = True,
        record_float: bool = False,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        model_type: str = "",
        **kwargs,
    ) -> None:
        super(SSLStepLogger, self).__init__(
            task_name,
            n_batches,
            batch_size,
            output_size,
            record_int=record_int,
            record_float=record_float,
            writer=writer,
            model_type=model_type,
            **kwargs,
        )

        self.logs: Dict[str, Any] = {
            "batch_num": 0,
            "total_loss": 0.0,
            "total_correct": 0.0,
            "total_top5": 0.0,
            "avg_loss": 0.0,
            "avg_output_std": 0.0,
        }

        self.collapse_level = kwargs.get("collapse_level", False)
        self.euclidean = kwargs.get("euclidean", False)

        if self.collapse_level:
            self.logs["collapse_level"] = 0
        if self.euclidean:
            self.logs["euc_dist"] = 0

    def log(
        self,
        step_num: int,
        loss: Tensor,
        z: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        bbox: Optional[BoundingBox] = None,
        *args,
        **kwargs,
    ) -> None:
        """Logs the outputs and results from a step of model fitting. Overwrites abstract method.

        Args:
            step_num (int): The global step number of for the mode of model fitting.
            loss (~torch.Tensor): Loss from this step of model fitting.
            z (~torch.Tensor): Optional; Output tensor from the model.
            y (~torch.Tensor): Optional; Labels to assess model output against.
            bbox (~torchgeo.datasets.utils.BoundingBox): Optional; Bounding boxes of the input samples.
        """
        assert z is not None

        if check_substrings_in_string(self.model_type, "segmentation"):
            z = z.flatten(1, -1)

        # Adds the loss for this step to the logs.
        ls = loss.item()
        self.logs["total_loss"] += ls

        # Compute the TOP1 and TOP5 accuracies.
        cosine_sim = CosineSimilarity(reduction=None)
        sim_argsort = cosine_sim(*torch.split(z, int(0.5 * len(z)), 0))
        correct = float((sim_argsort == 0).float().mean().cpu().numpy())
        top5 = float((sim_argsort < 5).float().mean().cpu().numpy())

        if self.euclidean:
            z_a, z_b = torch.split(z, int(0.5 * len(z)), 0)

            euc_dists = []
            for i in range(len(z_a)):
                euc_dists.append(
                    utils.calc_norm_euc_dist(
                        torch.nan_to_num(z_a[i]).detach().cpu().numpy(),
                        torch.nan_to_num(z_b[i]).detach().cpu().numpy(),
                    )
                )

            euc_dist = sum(euc_dists) / len(euc_dists)
            self.write_metric("euc_dist", euc_dist, step_num)
            self.logs["euc_dist"] += euc_dist

        if self.collapse_level:
            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = torch.split(z, int(0.5 * len(z)), 0)[0].detach()
            output = torch.nn.functional.normalize(output, dim=1)

            output_std = torch.std(output, 0)  # type: ignore[attr-defined]
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

            self.write_metric("collapse_level", collapse_level, step_num)

            self.logs["collapse_level"] = collapse_level

        # Add accuracies to log.
        self.logs["total_correct"] += correct
        self.logs["total_top5"] += top5

        # Writes the loss to the writer.
        self.write_metric("loss", ls, step_num=step_num)
        self.write_metric("acc", correct / 2 * len(z[0]), step_num)
        self.write_metric("top5_acc", top5 / 2 * len(z[0]), step_num)

        # Adds 1 to the batch number (step number).
        self.logs["batch_num"] += 1


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_logger(name) -> Callable[..., Any]:
    """Gets the constructor for a step logger to log the results from each step of model fitting during an epoch.

    Returns:
        ~typing.Callable[..., ~typing.Any]: The constructor of :class:`~logging.step.log.MinervaStepLogger`
        to be intialised within the epoch.

    .. versionadded:: 0.27
    """
    logger: Callable[..., Any] = func_by_str("minerva.logger.steplog", name)
    return logger
