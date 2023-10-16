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
"""TSNE Clustering task."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["TSNEVis"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from wandb.sdk.wandb_run import Run

from minerva.models import MinervaDataParallel, MinervaModel
from minerva.utils.visutils import plot_embedding

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class TSNEVis(MinervaTask):
    """TSNE clustering task.

    Passes a batch of data through the model in eval mode to get the embeddings.
    Passes these embeddings to :mod:`visutils` to train a TSNE algorithm and then visual the cluster.
    """

    def __init__(
        self,
        name: str,
        model: Union[MinervaModel, MinervaDataParallel],
        device: torch.device,
        exp_fn: Path,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        writer: Union[SummaryWriter, Run, None] = None,
        record_int: bool = True,
        record_float: bool = False,
        **params,
    ) -> None:
        backbone = model.get_backbone()  # type: ignore[assignment, operator]

        # Set dummy optimiser. It won't be used as this is a test.
        backbone.set_optimser(torch.optim.SGD(backbone.parameters(), lr=1.0e-3))

        super().__init__(
            name,
            backbone,
            device,
            exp_fn,
            gpu,
            rank,
            world_size,
            writer,
            record_int,
            record_float,
            **params,
        )

    def step(self) -> None:
        """Perform TSNE clustering on the embeddings from the model and visualise.

        Passes a batch from the test dataset through the model in eval mode to get the embeddings.
        Passes these embeddings to :mod:`visutils` to train a TSNE algorithm and then visual the cluster.
        """
        # Get a batch of data.
        data = next(iter(self.loaders))

        # Make sure the model is in evaluation mode.
        self.model.eval()

        # Pass the batch of data through the model to get the embeddings.
        embeddings: Tensor = self.model.step(data["image"].to(self.device))[0]

        # Flatten embeddings.
        embeddings = embeddings.flatten(start_dim=1)

        plot_embedding(
            embeddings.detach().cpu(),
            data["bbox"],
            self.name,
            show=True,
            filename=str(self.task_fn / "tsne_cluster_vis.png"),
        )
