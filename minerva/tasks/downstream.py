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
"""

.. versionadded:: 0.28
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch._dynamo.eval_frame import OptimizedModule
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

from minerva.datasets import make_loaders
from minerva.logger.tasklog import MinervaTaskLogger
from minerva.models import (
    FilterOutputs,
    MinervaDataParallel,
    MinervaModel,
    MinervaWrapper,
    extract_wrapped_model,
    wrap_model,
)
from minerva.tasks import MinervaTask
from minerva.utils import utils, visutils
from minerva.utils.utils import fallback_params, func_by_str


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MultiLabelSceneClassification(MinervaTask):
    logger_cls: str = "SupervisedTaskLogger"
    model_io_name: str = "sup_tg"

    def __init__(
        self,
        name: str,
        model: Union[MinervaModel, MinervaDataParallel, OptimizedModule],
        device: torch.device,
        exp_fn: Path,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        writer: Optional[Union[SummaryWriter, Run]] = None,
        record_int: bool = True,
        record_float: bool = False,
        train: bool = False,
        n_classes: int = 19,
        **global_params,
    ) -> None:
        model = torch.nn.Sequential(
            model.backbone.encoder.requires_grad_(False),
            FilterOutputs(-1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 19),
        ).to(device)

        model = MinervaWrapper(model, criterion=torch.nn.BCELoss())

        super(
            name,
            model,
            device,
            exp_fn,
            gpu,
            rank,
            world_size,
            writer,
            record_int,
            record_float,
            train,
            **global_params,
        )
