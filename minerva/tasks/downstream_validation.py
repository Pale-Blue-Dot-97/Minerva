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
"""Custom task for downstreaming a model on a validation task.

.. versionadded:: 0.29
"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["DownstreamTask"]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import re
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import hydra
import torch
import torch.distributed as dist
from tqdm import tqdm

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard.writer import SummaryWriter
else:  # pragma: no cover
    SummaryWriter = None
from omegaconf import OmegaConf
from torch._dynamo.eval_frame import OptimizedModule
from wandb.sdk.wandb_run import Run

from minerva.models import MinervaDataParallel, MinervaModel

from .core import MinervaTask


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class DownstreamTask(MinervaTask):
    """Custom task for validating a model during fitting.

    .. versionadded:: 0.29
    """

    logger_cls = "minerva.logger.tasklog.SupervisedTaskLogger"

    def __init__(
        self,
        name: str,
        model: MinervaModel | MinervaDataParallel | OptimizedModule,
        device: torch.device,
        exp_fn: Path,
        gpu: int = 0,
        rank: int = 0,
        world_size: int = 1,
        writer: Optional[SummaryWriter | Run] = None,
        backbone_weight_path: Optional[str | Path] = None,
        record_int: bool = True,
        record_float: bool = False,
        train: bool = False,
        **global_params,
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
            backbone_weight_path,
            record_int,
            record_float,
            train,
            **global_params,
        )

        downstream_model_params = self.params["downstream_model_params"]
        assert isinstance(downstream_model_params, dict)
        self.downstream_model_params = downstream_model_params

        self.n_epochs = self.params["n_epochs"]

        self.freeze_backbone = self.params.get("freeze_backbone", True)

        # Overwrite and re-setup model as a downstream model.
        self.make_model()
        self.setup_model(rebuild=True)
        self.update_encoder_weights()

        # Rebuild loggers to account for new model structure.
        self.logger = self.make_logger()

    def make_model(self) -> None:
        """Creates a model from the parameters specified by config.

        Returns:
            MinervaModel: Initialised model.
        """
        model_params: dict[str, Any] = deepcopy(self.downstream_model_params)
        if OmegaConf.is_config(model_params):
            model_params = OmegaConf.to_object(model_params)  # type: ignore[assignment]

        is_minerva = True if re.search(r"minerva", model_params["_target_"]) else False

        if "n_classes" in model_params.keys():
            # Updates the number of classes in case it has been altered by class balancing.
            model_params["n_classes"] = self.params["n_classes"]

        if "num_classes" in model_params.keys():
            # Updates the number of classes in case it has been altered by class balancing.
            model_params["num_classes"] = self.params["n_classes"]

        if self.params.get("mix_precision", False):
            model_params["scaler"] = torch.cuda.amp.grad_scaler.GradScaler()

        # Initialise model.
        model: MinervaModel
        if is_minerva:
            model = hydra.utils.instantiate(
                model_params, criterion=self.make_criterion()
            )
        else:
            model_params["model"] = hydra.utils.get_method(model_params["_target_"])
            model_params["_target_"] = "minerva.models.MinervaWrapper"
            model = hydra.utils.instantiate(
                model_params,
                criterion=self.make_criterion(),
            )

        self.model = model

    def update_encoder_weights(self) -> None:
        assert hasattr(self.model, "encoder")
        assert self.backbone_weight_path is not None
        self.model.encoder.load_state_dict(
            torch.load(self.backbone_weight_path, map_location=self.device)
        )
        if self.freeze_backbone:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def step(self) -> None:
        self.make_model()
        self.update_encoder_weights()
        self.setup_model(rebuild=True)

        for i in range(self.n_epochs):
            self.training_step(i)

        self.validation_step()

    def training_step(self, epoch_no: int) -> None:
        assert isinstance(self.loaders, dict)

        total_loss = 0.0

        # Initialises a progress bar for the epoch.
        with tqdm(
            total=len(self.loaders["train"])
        ) if self.gpu == 0 else nullcontext() as bar:
            # Sets the model up for training.
            self.model.train()

            if self.freeze_backbone:
                self.model.encoder.eval()
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

            # Core of the epoch.
            for batch in self.loaders["train"]:
                results = self.modelio(
                    batch,
                    self.model,
                    self.device,
                    True,
                    **self.params,
                )

                # Updates progress bar that batch has been processed.
                if self.gpu == 0:
                    total_loss += results[0]
                    bar.update()  # type: ignore

        if self.gpu == 0:
            avg_loss = total_loss * self.batch_size / len(self.loaders["train"])
            print(
                f"Downstream training, Sub-Epoch {epoch_no} | loss (on rank 0): {avg_loss}"
            )

    def validation_step(self) -> None:
        assert isinstance(self.loaders, dict)

        # Initialises a progress bar for the epoch.
        with tqdm(
            total=len(self.loaders["val"])
        ) if self.gpu == 0 else nullcontext() as bar:
            # Sets the model to evaluation modes.
            self.model.eval()

            # Ensure gradients will not be calculated if this is not a training task.
            with torch.no_grad():  # type: ignore[attr-defined]
                # Core of the epoch.
                for batch in self.loaders["val"]:
                    results = self.modelio(
                        batch,
                        self.model,
                        self.device,
                        False,
                        **self.params,
                    )

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
