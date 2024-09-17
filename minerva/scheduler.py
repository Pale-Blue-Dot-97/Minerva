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
"""Module containing custom learning rate schedulers for use in :mod:`minerva`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["CosineLR"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import warnings

import numpy as np
from torch.optim.lr_scheduler import LRScheduler


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class CosineLR(LRScheduler):
    """Cosine learning rate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        min_lr (int): Minimum learning rate.
        max_lr (int): Maximum learning rate.
        max_epochs (int): Epoch number to run learning rate cosine oscilation up to.
        n_periods (int): Optional; Number of periods of the cosine oscilation over the scheduler. Default 1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    """

    def __init__(
        self,
        optimizer,
        min_lr: int,
        max_lr: int,
        max_epochs: int,
        n_periods: int = 1,
        last_epoch: int = -1,
        verbose="deprecated",
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_epochs = max_epochs
        self.n_periods = n_periods
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        if not hasattr(self, "_get_lr_called_within_step"):
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [self._cosine() for _ in self.optimizer.param_groups]

    def _cosine(self) -> float:
        x = self.max_lr + 0.5 * (self.min_lr - self.max_lr) * (
            1 + np.cos(np.pi * self.n_periods * self.last_epoch / self.max_epochs)
        )

        assert isinstance(x, float)
        return x
