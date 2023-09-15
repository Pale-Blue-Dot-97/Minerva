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
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as ptfunc
from alive_progress import alive_it
from torch import Tensor
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter

from minerva.logger import KNNLogger
from minerva.models import MinervaDataParallel, MinervaModel, MinervaSiamese
from minerva.utils import AUX_CONFIGS, utils

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class WeightedKNN(MinervaTask):
    def __init__(
        self,
        model: MinervaModel,
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
        super().__init__(
            model,
            batch_size,
            n_batches,
            model_type,
            loader,
            device,
            **params,
        )

        self.temp = self.params["temp"]
        self.k = self.params["k"]

    def step(
        self,
        mode: str,
    ) -> None:
        """Trains a KNN using the model to validate a SSL model.

        Adapted from https://github.com/yaohungt/Barlow-Twins-HSIC for use in :mod:`minerva`.

        Args:
            temp (float, optional): Temperature of the similarity loss. Defaults to 0.5.
            k (int, optional): Number of similar images to use to predict images. Defaults to 200.
            mode (str, optional): Mode of model fitting this has been called on. Defaults to "val".
            record_int (bool, optional): Whether to record integer values. Defaults to True.
            record_float (bool, optional): Whether to record floating point values. Warning!
                This may result in memory issues on large amounts of data! Defaults to False.

        Returns:
            dict[str, ~typing.Any] | None: Results dictionary from the epoch logger if ``record_int``
            or ``record_float`` are ``True``.
        """

        # Puts the model in evaluation mode so no back passes are made.
        self.model.eval()

        # Get the number of classes from the data config.
        n_classes = len(AUX_CONFIGS["data_config"]["classes"])

        # Calculates the number of samples.
        n_samples = self.n_batches * self.batch_size

        # Uses the special `KNNLogger` to log the results from the KNN.
        epoch_logger = KNNLogger(
            self.n_batches,
            self.batch_size,
            n_samples,
            record_int=self.record_int,
            record_float=self.record_float,
            writer=self.writer,
        )

        total_num = 0
        feature_list = []
        target_list = []

        with torch.no_grad():
            # Generate feature bank and target bank.
            feat_bar = alive_it(self.loaders["val"])
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
                    test_data.size(0) * self.k, n_classes, device=sim_labels.device
                )

                # [B*K, C]
                one_hot_label = one_hot_label.scatter(
                    dim=-1, index=sim_labels.view(-1, 1), value=1.0
                )

                # Weighted score ---> [B, C]
                pred_scores = torch.sum(
                    one_hot_label.view(test_data.size(0), -1, n_classes)
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
                epoch_logger.log(mode, self.step_num, *results)

                # Update global step number for this mode of model fitting.
                self.step_num += 1
