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
"""KNN Validation task.

.. versionadded:: 0.27
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["WeightedKNN"]

from pathlib import Path

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as ptfunc
from alive_progress import alive_it
from torch import Tensor
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter
else:  # pragma: no cover
    SummaryWriter = None

from minerva.logging import SSLTaskLogger
from minerva.models import MinervaDataParallel, MinervaModel, MinervaSiamese
from minerva.utils import utils

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class WeightedKNN(MinervaTask):
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

    .. versionadded:: 0.27
    """

    logger_cls = SSLTaskLogger

    def __init__(
        self,
        name: str,
        model: Union[MinervaModel, MinervaDataParallel],
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

        self.temp = self.params.get("temp")
        self.k = self.params.get("k")

    def generate_feature_bank(self) -> Tuple[Tensor, Tensor]:
        feature_list = []
        target_list = []

        feat_bar = alive_it(self.loaders["features"])
        for batch in feat_bar:
            val_data: Tensor = batch["image"].to(self.device, non_blocking=True)
            val_target: Tensor = batch["mask"].to(self.device, non_blocking=True)
            target_list.append(
                torch.mode(torch.flatten(val_target, start_dim=1)).values
            )

            # Get features from passing the input data through the model.
            if utils.check_substrings_in_string(self.model_type, "siamese"):
                # Checks that the model is of type ``MinervaSiamese`` so a call to `forward_single` will work.
                if isinstance(self.model, MinervaDataParallel):  # pragma: no cover
                    assert isinstance(self.model.model.module, MinervaSiamese)
                else:
                    assert isinstance(self.model, MinervaSiamese)

                # Ensures that the data is parsed through a single head of the model rather than paired.
                feature, _ = self.model.forward_single(val_data)  # type: ignore[operator]
            else:
                feature, _ = self.model(val_data)

            # The masks from segmentation models will need to be flattened.
            if utils.check_substrings_in_string(self.model_type, "segmentation"):
                feature = feature.flatten(1, -1)

            feature_list.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_list, dim=0).t().contiguous()

        # [N]
        feature_labels = (
            torch.cat(target_list, dim=0).contiguous().to(feature_bank.device)
        )

        return feature_bank, feature_labels

    def step(self) -> None:
        """Trains a KNN using the model to validate a SSL model.

        Adapted from https://github.com/yaohungt/Barlow-Twins-HSIC for use in :mod:`minerva`.
        """

        # Puts the model in evaluation mode so no back passes are made.
        self.model.eval()

        total_num = 0

        with torch.no_grad():
            # Generate feature bank and target bank.
            feature_bank, feature_labels = self.generate_feature_bank()

            # Loop test data to predict the label by weighted KNN search.
            test_bar = alive_it(self.loaders["test"])
            for batch in test_bar:
                test_data: Tensor = batch["image"].to(self.device, non_blocking=True)
                test_target: Tensor = torch.mode(
                    torch.flatten(
                        batch["mask"].to(self.device, non_blocking=True), start_dim=1
                    )
                ).values

                # Get features from passing the input data through the model.
                if utils.check_substrings_in_string(self.model_type, "siamese"):
                    # Checks that the model is of type ``MinervaSiamese`` so a call to `forward_single` will work.
                    if isinstance(self.model, MinervaDataParallel):  # pragma: no cover
                        assert isinstance(self.model.model.module, MinervaSiamese)
                    else:
                        assert isinstance(self.model, MinervaSiamese)

                    # Ensures that the data is parsed through a single head of the model rather than paired.
                    feature, _ = self.model.forward_single(test_data)  # type: ignore[operator]
                else:
                    feature, _ = self.model(test_data)

                # The masks from segmentation models will need to be flattened.
                if utils.check_substrings_in_string(self.model_type, "segmentation"):
                    feature = feature.flatten(1, -1)

                total_num += self.batch_size

                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)

                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)

                # [B, K]
                sim_labels = torch.gather(
                    feature_labels.expand(test_data.size(0), -1),
                    dim=-1,
                    index=sim_indices,
                )

                sim_weight = (sim_weight / self.temp).exp()

                # Counts for each class
                one_hot_label = torch.zeros(
                    test_data.size(0) * self.k, self.n_classes, device=sim_labels.device
                )

                # [B*K, C]
                one_hot_label = one_hot_label.scatter(
                    dim=-1, index=sim_labels.view(-1, 1), value=1.0
                )

                # Weighted score ---> [B, C]
                pred_scores = torch.sum(
                    one_hot_label.view(test_data.size(0), -1, self.n_classes)
                    * sim_weight.unsqueeze(dim=-1),
                    dim=1,
                )
                pred_scores = ptfunc.normalize(
                    pred_scores.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0),
                )

                # Calculate loss between predicted and ground truth labels by KNN.
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(pred_scores, test_target)

                # Pack results together for the logger.
                results = (loss, pred_scores, test_target, _)

                # Gathers the losses across devices together if a distributed job.
                if dist.is_available() and dist.is_initialized():  # pragma: no cover
                    loss = results[0].data.clone()
                    dist.all_reduce(loss.div_(dist.get_world_size()))
                    results = (loss, *results[1:])

                # Sends results to logger.
                self.logger.step(self.step_num, *results)

                # Update global step number for this mode of model fitting.
                self.step_num += 1
