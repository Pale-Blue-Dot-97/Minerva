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
"""Core functionality of :mod:`tasks`, defining the abstract :class:`MinervaTask` class"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.distributed as dist
from wandb.sdk.wandb_run import Run

from minerva.datasets import make_loaders
from minerva.logger import MinervaLogger
from minerva.metrics import MinervaMetrics
from minerva.models import MinervaDataParallel, MinervaModel
from minerva.utils.utils import func_by_str


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaTask(ABC):
    """An abstract definition of a task to fit or evalulate a model within :mod:`minerva`.

    Attributes:
        params (dict[str, ~typing.Any]): Dictionary describing all the parameters that define how the model will be
            constructed, trained and evaluated. These should be defined via config ``YAML`` files.
        model (MinervaModel): Model to be fitted of a class contained within :mod:`~minerva.models`.
        batch_size (int): Size of each batch of samples supplied to the model.
        loaders (dict[str, ~torch.utils.data.DataLoader]): :class:`dict` containing
            :class:`~torch.utils.data.DataLoader` (s) for each dataset.
        n_batches (dict[str, int]): Dictionary of the number of batches to supply to the model for train,
            validation and testing.
        metrics (dict[str, ~typing.Any]): Dictionary to hold the loss and accuracy results from training,
            validation and testing.
        device: The CUDA device on which to fit the model.
        verbose (bool): Provides more prints to stdout if ``True``.
        class_dist (~typing.Any): Distribution of classes within the data.
        sample_pairs (bool): Whether samples are paired together for Siamese learning.
        modes (tuple[str, ...]): The different *modes* of fitting in this experiment specified by the config.
        writer (~torch.utils.tensorboard.writer.SummaryWriter | ~wandb.sdk.wandb_run.Run | None): The *writer*
            to perform logging for this experiment. For use with either :mod:`tensorboard` or :mod:`wandb`.
        stopper (~pytorchtools.EarlyStopping | None): Early stopping function.
        early_stop (bool): Whether early stopping has been triggered. Will end model training if ``True``.
        n_samples (dict[str, int]): Number of samples in each mode of model fitting.
        metric_logger (~logger.MinervaLogger): Object to calculate and log metrics to track the performance
            of the model.
        modelio_func (~typing.Callable[..., ~typing.Any]): Function to handle the input/ output to the model.
        steps (dict[str, int]): :class:`dict` to hold the global step number for each mode of model fitting.
        model_type (str): Type of the model that determines how to handle IO, metric calculations etc.

    Args:
        model (MinervaModel): Model to be fitted of a class contained within :mod:`~minerva.models`.
        rank (int): Optional; The rank of this process across all devices in the distributed run.
        world_size (int): Optional; The total number of processes across the distributed run.
        writer (~wandb.sdk.wandb_run.Run | RunDisabled): Optional; Run object for Weights and Biases.
        params (dict[str, ~typing.Any]): Dictionary describing all the parameters that define how the model will be
            constructed, trained and evaluated. These should be defined via config ``YAML`` files.

    Keyword Args:
        batch_size (int): Number of samples in each batch.
        elim (bool): Will eliminate classes that have no samples in and reorder the class labels so they
            still run from ``0`` to ``n-1`` classes where ``n`` is the reduced number of classes.
            :mod:`minerva` ensures that labels are converted between the old and new schemes seamlessly.
        model_type (str): Defines the type of the model. If ``siamese``, ensures inappropiate functionality is not used.
        dataset_params (dict[str, ~typing.Any]): Parameters to construct each dataset.
            See documentation on structure of these.
        collator (dict[str, ~typing.Any]): Defines the collator to use that will collate samples together into batches.
            Contains the ``module`` key to define the import path and the ``name`` key
            for name of the collation function.
        sample_pairs (bool): Activates paired sampling for Siamese models. Only used for ``train`` datasets.
        stopping (dict[str, ~typing.Any]): Dictionary to hold the parameters defining the early stopping functionality.
            If no dictionary is given, it is assumed that there will be no early stopping.
        pre_train_name (str): Name of the pre-trained model to use.
        reload (bool): Reloads the weights in the cache matching ``pre_train_name`` to continue model fitting.
        loss_func (str): Name of the loss function to use.
        optim_func (str): Name of the optimiser function to use.
        lr (float): Learning rate of optimiser.
        optim_params (dict[str, ~typing.Any]): :class:`dict` to hold any additional parameters for the optimiser,
            other than the already handled learning rate -- ``lr``. Place them in the ``params`` key.
            If using a non-torch optimiser, use the ``module`` key to specify the import path to the optimiser function.
        loss_params (dict[str, ~typing.Any]): :class:`dict` to hold any additional parameters for the loss function
            in the ``params`` key. If using a non-torch loss function, you need to specify the import path
            with the ``module`` key.
        balance (bool): Activates class balancing. For ``model_type="scene classifer"`` or ``model_type="mlp"``,
            over and under sampling will be used. For ``model_type="segmentation"``, class weighting will be
            used on the loss function.
        patch_size (tuple[float, float]): Defines the shape of the patches in the dataset.
        input_size (tuple[int, ...]): Shape of the input to the model. Typically in CxHxW format.
            Should align with the values given for ``patch_size``.
        metrics (str): Specify the metric logger to use. Must be the name of a :class:`~metrics.MinervaMetric` class
            within :mod:`metrics`.
        logger (str): Specify the logger to use. Must be the name of a :class:`~logger.MinervaLogger` class
            within :mod:`logger`.
        modelio (str): Specify the IO function to use to handle IO for the model during fitting. Must be the name
            of a function within :mod:`modelio`.
        record_int (bool): Store the integer results of each epoch in memory such the predictions, ground truth etc.
        record_float (bool): Store the floating point results of each epoch in memory
            such as the raw predicted probabilities.
    """

    def __init__(
        self,
        model: Union[MinervaModel, MinervaDataParallel],
        batch_size: int,
        device: torch.device,
        exp_fn: Path,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        record_int: bool = True,
        record_float: bool = False,
        **params,
    ) -> None:
        self.model = model

        # Gets the datasets, number of batches, class distribution and the modfied parameters for the experiment.
        loaders, n_batches, class_dist, new_params = make_loaders(
            rank, world_size, **params
        )

        self.exp_fn = exp_fn

        self.gpu = gpu

        self.loaders = loaders
        self.params = new_params
        self.class_dist = class_dist

        # Corrects the batch size if this is a distributed job to account for batches being split across devices.
        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover
            self.batch_size = batch_size // dist.get_world_size()  # type: ignore[attr-defined]

        self.n_batches = n_batches
        self.model_type = self.params["model_type"]
        self.sample_pairs = self.params.get("sample_pairs", False)

        self.metric_logger: MinervaMetrics = self.make_metric_logger()
        self.modelio = self.get_io_func()
        self.logger: MinervaLogger

        self.loaders = loaders
        self.device = device

        self.record_int = record_int
        self.record_float = record_float

        self.writer = writer

        self.step_num = 0

    def make_metric_logger(self) -> MinervaMetrics:
        """Creates an object to calculate and log the metrics from the experiment, selected by config parameters.

        Returns:
            MinervaMetrics: Constructed metric logger.
        """

        # Gets the size of the input data to the network (without batch dimension).
        data_size = self.params["input_size"]

        # Gets constructor of the metric logger from name in the config.
        _metric_logger: Callable[..., Any] = func_by_str(
            "minerva.metrics", self.params["metrics"]
        )

        # Initialises the metric logger with arguments.
        metric_logger: MinervaMetrics = _metric_logger(
            self.n_batches,
            batch_size=self.batch_size,
            data_size=data_size,
            model_type=self.model_type,
            sample_pairs=self.sample_pairs,
        )

        return metric_logger

    def get_logger(self) -> Callable[..., Any]:
        """Creates an object to log the results from each step of model fitting during an epoch.

        Returns:
            ~typing.Callable[..., ~typing.Any]: The constructor of :class:`~logger.MinervaLogger`
            to be intialised within the epoch.
        """
        logger: Callable[..., Any] = func_by_str(
            "minerva.logger", self.params["logger"]
        )
        return logger

    def get_io_func(self) -> Callable[..., Any]:
        """Fetches a func to handle IO for the type of model used in the experiment.

        Returns:
            ~typing.Callable[..., ~typing.Any]: Model IO function requested from parameters.
        """
        io_func: Callable[..., Any] = func_by_str(
            "minerva.modelio", self.params["model_io"]
        )
        return io_func

    @abc.abstractmethod
    def step(self, mode: str) -> None:
        raise NotImplementedError

    def _generic_step(self, mode: str) -> Optional[Dict[str, Any]]:
        self.step(mode)

        # Send the logs to the metric logger.
        self.metric_logger(mode, self.logger.get_logs)

        if self.record_int or self.record_float:
            return self.logger.get_results
        else:
            return None

    def __call__(self, mode: str) -> Any:
        return self._generic_step(mode)

    @property
    def get_logs(self) -> Dict[str, Any]:
        return self.logger.get_logs

    def __repr__(self) -> str:
        return self.__class__.__name__


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_task(task: str, *args, **params) -> MinervaTask:
    """Get the requested :class:`MinervaTask` by name.

    Args:
        task (str): Name of the task.
        params (Dict[str, Any]): Parameters for the task to be initialised.

    Returns:
        MinervaTask: Constructed :class:`MinervaTask` object.
    """
    _task = func_by_str("minerva.tasks", task)

    task = _task(*args, **params)
    assert isinstance(task, MinervaTask)
    return task
