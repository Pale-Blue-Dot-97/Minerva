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
r"""Tests for :mod:`minerva.trainer`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from minerva.models import MinervaModel, MinervaOnnxModel, is_minerva_subtype
from minerva.trainer import Trainer
from minerva.utils import runner, utils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
@runner.distributed_run
def run_trainer(gpu: int, wandb_run: Optional[Run | RunDisabled], cfg: DictConfig):
    params = deepcopy(cfg)
    params["calc_norm"] = True

    trainer = Trainer(
        gpu=gpu,
        wandb_run=wandb_run,
        **params,
    )
    assert isinstance(trainer, Trainer)

    trainer.fit()

    trainer.test()

    if gpu == 0:
        trainer.save_model()

        trainer.save_backbone()

    assert trainer.exp_fn.parent.exists()
    shutil.rmtree(trainer.exp_fn.parent)


def test_trainer_1(default_config: DictConfig) -> None:
    with runner.WandbConnectionManager():
        if torch.distributed.is_available():  # type: ignore
            # Disable wandb logging on Windows in CI/CD due to pwd.
            if os.name == "nt":
                OmegaConf.update(default_config, "wandb_log", False, force_add=True)
            else:
                # Configure the arguments and environment variables.
                OmegaConf.update(default_config, "log_all", False, force_add=True)
                OmegaConf.update(default_config, "entity", None, force_add=True)
                OmegaConf.update(default_config, "project", "pytest", force_add=True)
                OmegaConf.update(default_config, "wandb_log", True, force_add=True)

            # Run the specified main with distributed computing and the arguments provided.
            run_trainer(default_config)

        else:
            # Disable wandb logging on Windows in CI/CD due to pwd.
            if os.name == "nt":
                OmegaConf.update(default_config, "wandb_log", False, force_add=True)

            run_trainer(default_config)


def test_trainer_2(default_config: DictConfig, cache_dir: Path) -> None:
    params1 = deepcopy(default_config)
    params1["elim"] = False

    trainer1 = Trainer(0, **params1)

    with pytest.raises(ValueError):
        trainer1.save_model(fmt="unkown")

    suffix = "onnx"
    try:
        utils._optional_import("onnx2torch", package="onnx2torch")
    except ValueError:
        suffix = "pt"

    trainer1.save_model(fn=trainer1.get_model_cache_path(), fmt=suffix)

    params2 = deepcopy(params1)
    OmegaConf.update(
        params2,
        "pre_train_name",
        cache_dir / f"{params1['model_name'].split('-')[0]}.{suffix}",
        force_add=True,
    )
    OmegaConf.update(params2, "sample_pairs", "false", force_add=True)
    params2.plot_last_epoch = False
    params2.wandb_log = False
    params2.max_epochs = 2
    del params2.stopping

    trainer2 = Trainer(0, **params2)
    if suffix == "onnx":
        assert is_minerva_subtype(trainer2.model, MinervaOnnxModel)
    else:
        assert is_minerva_subtype(trainer2.model, MinervaModel)

    trainer2.fit()
    trainer2.test()

    assert isinstance(repr(trainer2.model), str)


def test_trainer_3(default_config: DictConfig) -> None:
    params1 = deepcopy(default_config)

    trainer1 = Trainer(0, **params1)

    pre_train_path = trainer1.get_model_cache_path()
    trainer1.save_model(fn=pre_train_path, fmt="onnx")

    params2 = deepcopy(default_config)
    OmegaConf.update(
        params2,
        "pre_train_name",
        str(pre_train_path.with_suffix(".onnx")),
        force_add=True,
    )
    params2["fine_tune"] = True
    params2["max_epochs"] = 2
    params2["elim"] = False
    del params2.stopping

    trainer2 = Trainer(0, **params2)
    trainer2.fit()


@pytest.mark.parametrize(
    ["cfg_name", "cfg_args", "kwargs"],
    [
        ("example_CNN_config.yaml", {}, {}),
        ("example_GeoCLR_config.yaml", {}, {"tsne_cluster": True}),
        ("example_GeoCLR_config.yaml", {"plot_last_epoch": False}, {}),
        ("example_3rd_party.yaml", {}, {"test": True}),
        ("example_autoencoder_config.yaml", {}, {"test": True}),
        ("example_UNetR_config.yaml", {}, {"test": True}),
        ("example_GeoSimConvNet.yaml", {}, {}),
        ("example_GSConvNet-II.yaml", {}, {}),
        ("example_PSP.yaml", {}, {"test": True}),
        ("example_SceneClassifier.yaml", {}, {"test": True}),
        ("example_MultiLabel.yaml", {}, {"test": True}),
        # ("example_ChangeDetector.yaml", {}, {"test": True}), Disabled
    ],
)
def test_trainer_4(
    inbuilt_cfg_root: Path,
    cfg_name: str,
    cfg_args: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    with hydra.initialize(version_base="1.3", config_path=str(inbuilt_cfg_root)):
        cfg = hydra.compose(config_name=cfg_name)

        for key in cfg_args.keys():
            cfg[key] = cfg_args[key]

        trainer = Trainer(0, **cfg)

        trainer.fit()

        if kwargs.get("tsne_cluster"):
            trainer.tsne_cluster("test-test")

        if kwargs.get("test"):
            trainer.test()


def test_trainer_resume(default_config: DictConfig) -> None:
    params1 = OmegaConf.to_object(default_config)
    assert isinstance(params1, dict)

    params1["checkpoint_experiment"] = True
    del params1["stopping"]

    trainer1 = Trainer(0, **params1)
    while trainer1.epoch_no < trainer1.max_epochs - 1:
        trainer1.fit()

    params2 = deepcopy(params1)
    params2["exp_name"] = trainer1.params["exp_name"]
    params2["resume"] = True

    trainer2 = Trainer(0, **params2)

    trainer2.fit()
    trainer2.test()
