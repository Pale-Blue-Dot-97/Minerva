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
__all__ = ["StandardEpoch"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
import torch.distributed as dist
from alive_progress import alive_bar
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

from minerva.logger import MinervaLogger
from minerva.models import MinervaDataParallel, MinervaModel
from minerva.utils import utils

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class StandardEpoch(MinervaTask):
    def step(
        self,
        mode: str,
    ) -> None:
        """All encompassing function for any type of epoch, be that train, validation or testing.

        Args:
            mode (str): Either train, val or test.
                Defines the type of epoch to run on the model.
        """
        batch_size = self.batch_size
        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover
            batch_size = self.batch_size // dist.get_world_size()  # type: ignore[attr-defined]

        # Calculates the number of samples
        n_samples = self.n_batches[mode] * batch_size

        # Creates object to log the results from each step of this epoch.
        _epoch_logger: Callable[..., Any] = self.get_logger()
        epoch_logger: MinervaLogger = _epoch_logger(
            self.n_batches[mode],
            batch_size,
            n_samples,
            self.model.output_shape,
            self.model.n_classes,
            record_int=self.record_int,
            record_float=self.record_float,
            collapse_level=self.sample_pairs,
            euclidean=self.sample_pairs,
            model_type=self.model_type,
            writer=self.writer,
        )

        # Initialises a progress bar for the epoch.
        with alive_bar(
            self.n_batches[mode], bar="blocks"
        ) if self.gpu == 0 else nullcontext() as bar:
            # Sets the model up for training or evaluation modes.
            if mode == "train":
                self.model.train()
            else:
                self.model.eval()

            # Core of the epoch.
            for batch in self.loaders[mode]:
                results = self.modelio(
                    batch, self.model, self.device, mode, **self.params
                )

                if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover
                    loss = results[0].data.clone()
                    dist.all_reduce(loss.div_(dist.get_world_size()))  # type: ignore[attr-defined]
                    results = (loss, *results[1:])

                epoch_logger.log(mode, self.step_num, *results)

                self.step_num += 1

                # Updates progress bar that batch has been processed.
                if self.gpu == 0:
                    bar()  # type: ignore

        # Updates metrics with epoch results.
        self.metric_logger(mode, epoch_logger.get_logs)

        # If configured to do so, calculates the grad norms.
        if self.params.get("calc_norm", False):
            _ = utils.calc_grad(self.model)
