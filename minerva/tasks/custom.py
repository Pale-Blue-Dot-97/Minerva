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
"""KNN Validation task.

.. versionadded:: 0.27
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["WeightedKNN"]

from pathlib import Path

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as ptfunc
import torch.nn
from torch import Tensor
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter
else:  # pragma: no cover
    SummaryWriter = None

from minerva.models import (
    MinervaDataParallel,
    MinervaModel,
    MinervaSiamese,
    extract_wrapped_model,
    is_minerva_subtype,
    FCN32ResNet18,
    wrap_model,
)
from minerva.utils import utils

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class CustomTask(MinervaTask):
    """A KNN Validation task.

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
        record_int (bool): Store the integer results of each epoch in memory such the predictions, ground truth etc.
        record_float (bool): Store the floating point results of each epoch in memory
            such as the raw predicted probabilities.

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
        modelio (str): Specify the IO function to use to handle IO for the model during fitting. Must be the name
            of a function within :mod:`modelio`.

    .. versionadded:: 0.27
    """

    logger_cls = "SupervisedTaskLogger"

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
        backbone_weight_path: str = None,
        record_int: bool = True,
        record_float: bool = False,
        n_classes: int = 5,
        **params,
    ) -> None:
        
        self.backbone_weight_path = backbone_weight_path

        model = FCN32ResNet18(criterion=torch.nn.CrossEntropyLoss(),
                              n_classes=n_classes)
        
        model.determine_output_dim()
        
        super().__init__(
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
            **params,
        )

    def step(self) -> None:

        self.model.backbone.load_state_dict(torch.load(self.backbone_weight_path))

        self.model.set_optimiser(torch.optim.SGD(self.model.parameters(), lr=1.0e-3))

        # Transfer to GPU.

        self.model.to(self.device)
        
        # If writer is `wandb`, `watch` the model to log gradients.

        if isinstance(self.writer, Run):

            self.writer.watch(self.model)

        # Checks if multiple GPUs detected. If so, wraps model in DistributedDataParallel for multi-GPU use.

        # Will also wrap the model in torch.compile if specified to do so in params.

        self.model = wrap_model(
            self.model, self.gpu, self.params.get("torch_compile", False))

        # Make sure the model is in evaluation mode.
 
        self.model.train()

        self.model.backbone.requires_grad_(False)

        for batch in tqdm(self.loaders["features"]):
            
            val_data: Tensor = batch["image"].to(self.device, non_blocking=True)
            val_target: Tensor = batch["mask"].to(self.device, non_blocking=True).squeeze()

            _, _ = self.model.step(val_data,val_target,train=True)

        # Puts the model in evaluation mode so no back passes are made.
        self.model.eval()

        # Loop test data to predict the label by weighted KNN search.
        for batch in tqdm(self.loaders["test"]):
            test_data: Tensor = batch["image"].to(self.device, non_blocking=True)
            test_target: Tensor = batch["mask"].to(self.device, non_blocking=True).squeeze()

            # Calculate loss between predicted and ground truth labels by KNN.
            loss, z = self.model.step(test_data, test_target, train=False)
            #loss = criterion(pred_scores, test_target.to(dtype=torch.long))

            # Pack results together for the logger.
            results = (loss, z, test_target, None)

            # Gathers the losses across devices together if a distributed job.
            if dist.is_available() and dist.is_initialized():  # pragma: no cover
                loss = results[0].data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))
                results = (loss, *results[1:])

            # Sends results to logger.
            self.logger.step(self.global_step_num, self.local_step_num, *results)

            # Update global step number for this mode of model fitting.
            self.global_step_num += 1
            self.local_step_num += 1