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
""""""
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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from minerva.logger import MinervaLogger
from minerva.metrics import MinervaMetrics
from minerva.models import MinervaDataParallel, MinervaModel
from minerva.utils.utils import func_by_str


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaTask(ABC):
    def __init__(
        self,
        model: Union[MinervaModel, MinervaDataParallel],
        batch_size: int,
        n_batches: int,
        model_type: str,
        loader: DataLoader[Iterable[Any]],
        device: torch.device,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        record_int: bool = True,
        record_float: bool = False,
        **params,
    ) -> None:
        self.model = model
        self.params = params

        # Corrects the batch size if this is a distributed job to account for batches being split across devices.
        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover
            self.batch_size = batch_size // dist.get_world_size()  # type: ignore[attr-defined]

        self.n_batches = n_batches
        self.model_type = model_type
        self.sample_pairs = self.params.get("sample_pairs", False)

        self.metric_logger: MinervaMetrics = self.make_metric_logger()
        self.logger: MinervaLogger = self.get_logger()
        self.modelio = self.get_io_func()

        self.loader = loader
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
        pass

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
