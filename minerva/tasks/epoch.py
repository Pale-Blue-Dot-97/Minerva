# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
"""A standard epoch for use with generic model fitting.

.. versionadded:: 0.27
"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["StandardEpoch"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from contextlib import nullcontext

import torch
import torch.distributed as dist
from tqdm import tqdm

from minerva.utils import utils

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class StandardEpoch(MinervaTask):
    """A standard epoch for use with generic model fitting.

    .. versionadded:: 0.27
    """

    logger_cls = "minerva.logger.tasklog.SupervisedTaskLogger"

    def step(self) -> None:
        # Initialises a progress bar for the epoch.
        with tqdm(total=self.n_batches) if self.gpu == 0 else nullcontext() as bar:
            # Sets the model up for training or evaluation modes.
            if self.train:
                self.model.train()
            else:
                self.model.eval()

            # Ensure gradients will not be calculated if this is not a training task.
            with torch.no_grad() if not self.train else nullcontext():  # type: ignore[attr-defined]
                # Core of the epoch.
                for batch in self.loaders:
                    results = self.modelio(
                        batch,
                        self.model,
                        self.device,
                        self.train,
                        **self.params,
                    )

                    if self.local_step_num % self.log_rate == 0:
                        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]  # pragma: no cover  # noqa: E501
                            loss = results[0].data.clone()
                            dist.all_reduce(loss.div_(dist.get_world_size()))  # type: ignore[attr-defined]
                            results = (loss, *results[1:])

                        self.logger.step(
                            self.global_step_num, self.local_step_num, *results
                        )

                    self.global_step_num += 1
                    self.local_step_num += 1

                    # Updates progress bar that batch has been processed.
                    if self.gpu == 0:
                        bar.update()  # type: ignore

        # If configured to do so, calculates the grad norms.
        if self.params.get("calc_norm", False):
            _ = utils.calc_grad(self.model)
