# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2018 Bjarte Mehus Sunde
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Module containing :class:`EarlyStopping` to track when the training of a model should stop.

Source: https://github.com/Bjarten/early-stopping-pytorch
"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Bjarte Mehus Sunde"
__license__ = "MIT"
__copyright__ = "Copyright (C) 2018 Bjarte Mehus Sunde"

# =====================================================================================================================
#                                                    IMPORTS
# =====================================================================================================================
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.nn.modules import Module


# =====================================================================================================================
#                                                    CLASSES
# =====================================================================================================================
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
        patience (int): How long to wait after last time validation loss improved.
        verbose (bool): If ``True``, prints a message for each validation loss improvement.
        counter (int): Number of epochs of worsening validation loss since last improvement.
        best_score (float): Best validation loss score recorded.
        early_stop (bool): Will be ``True`` if early stopping is triggered by ``patience`` number of validation epochs
            with worsening validation losses consecutively.
        val_loss_min (float): The lowest validation loss recorded.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path for the checkpoint to be saved to.
        trace_func (~typing.Callable[..., None]): Trace print function.

    Args:
        patience (int): How long to wait after last time validation loss improved.
            Default: ``7``
        verbose (bool): If ``True``, prints a message for each validation loss improvement.
            Default: ``False``
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            Default: ``0``
        path (str): Path for the checkpoint to be saved to.
            Default: ``'checkpoint.pt'``
        trace_func (~typing.Callable[..., None]): Trace print function.
            Default: :func:`print`
        external_save (bool): If True, will not save the model here, but will activate a :attr:`save_model` flag
            indicating that the model should be saved by the user. If False, will save the model automaticallly
            using :mod:`torch`.
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: str | Path = "checkpoint.pt",
        trace_func: Callable[..., None] = print,
        external_save: bool = False,
    ):
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf
        self.delta: float = delta
        self.path: str | Path = path
        self.trace_func: Callable[..., None] = trace_func
        self.external_save: bool = external_save
        self.save_model: bool = False

    def __call__(self, val_loss: float, model: Module) -> None:
        # Reset save model flag.
        self.save_model = False

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: Module) -> None:
        """Saves model when validation loss decrease.

        Args:
            val_loss (float): Validation loss.
            model (~torch.nn.Module): The model to save checkpoint of.
        """
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        # If externally saving model, activate flag.
        if self.external_save:
            self.save_model = True

        # Else, save the model state dict using torch.
        else:
            torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
