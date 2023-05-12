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
r"""Tests for :mod:`minerva.trainer`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
import argparse
import shutil
from pathlib import Path

import pytest
import torch

from minerva.models import MinervaModel, MinervaOnnxModel
from minerva.trainer import Trainer
from minerva.utils import CONFIG, config_load, runner, utils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def run_trainer(gpu: int, args: argparse.Namespace):
    args.gpu = gpu
    params = CONFIG.copy()
    params["calc_norm"] = True

    trainer = Trainer(
        gpu=args.gpu,
        rank=args.rank,
        world_size=args.world_size,
        wandb_run=args.wandb_run,
        **params,
    )
    assert isinstance(trainer, Trainer)

    trainer.fit()

    trainer.test()

    if args.gpu == 0:
        trainer.save_model()

        trainer.save_backbone()

    assert trainer.exp_fn.parent.exists()
    shutil.rmtree(trainer.exp_fn.parent)


def test_trainer_1() -> None:
    args = argparse.Namespace()

    with runner.WandbConnectionManager():
        if torch.distributed.is_available():  # type: ignore
            # Configure the arguments and environment variables.
            runner.config_args(args)

            args.log_all = False
            args.entity = None
            args.project = "pytest"
            args.wandb_log = True

            # Run the specified main with distributed computing and the arguments provided.
            runner.distributed_run(run_trainer, args)

        else:
            args.gpu = 0
            args.wandb_run = None
            run_trainer(args.gpu, args)


def test_trainer_2() -> None:
    params1 = CONFIG.copy()

    trainer1 = Trainer(0, **params1)

    with pytest.raises(ValueError):
        trainer1.save_model(fmt="unkown")

    suffix = "onnx"
    try:
        utils._optional_import("onnx2torch", package="onnx2torch")
    except ValueError:
        suffix = "pt"

    trainer1.save_model(fn=trainer1.get_model_cache_path(), fmt=suffix)
    params2 = CONFIG.copy()
    params2["pre_train_name"] = f"{params1['model_name'].split('-')[0]}.{suffix}"
    params2["sample_pairs"] = "false"
    params2["plot_last_epoch"] = False
    params2["wandb_log"] = False
    params2["project"] = False
    params2["max_epochs"] = 2

    trainer2 = Trainer(0, **params2)
    if suffix == "onnx":
        assert isinstance(trainer2.model, MinervaOnnxModel)
    else:
        assert isinstance(trainer2.model, MinervaModel)

    trainer2.fit()
    trainer2.test()

    assert type(repr(trainer2.model)) is str


def test_trainer_3() -> None:
    params1 = CONFIG.copy()

    trainer1 = Trainer(0, **params1)
    trainer1.save_model(fn=trainer1.get_model_cache_path())

    params2 = CONFIG.copy()
    params2["pre_train_name"] = params1["model_name"]
    params2["fine_tune"] = True
    # params2["reload"] = True
    params2["max_epochs"] = 2
    params2["elim"] = False

    trainer2 = Trainer(0, **params2)
    trainer2.fit()


def test_cnn_train() -> None:
    cfg_path = Path(__file__).parent.parent / "inbuilt_cfgs" / "example_CNN_config.yml"

    with config_load.ToDefaultConfDir():
        cfg, _ = config_load.load_configs(cfg_path)

    trainer = Trainer(0, **cfg)

    trainer.fit()
    trainer.test()


def test_ssl_trainer() -> None:
    ssl_cfg_path = (
        Path(__file__).parent.parent / "inbuilt_cfgs" / "example_GeoCLR_config.yml"
    )

    with config_load.ToDefaultConfDir():
        ssl_cfg, _ = config_load.load_configs(ssl_cfg_path)

    trainer = Trainer(0, **ssl_cfg)

    trainer.fit()

    trainer.model = trainer.model.get_backbone()  # type: ignore[assignment, operator]

    trainer.tsne_cluster()


def test_ssl_trainer_2() -> None:
    ssl_cfg_path = (
        Path(__file__).parent.parent / "inbuilt_cfgs" / "example_GeoCLR_config.yml"
    )

    with config_load.ToDefaultConfDir():
        ssl_cfg, _ = config_load.load_configs(ssl_cfg_path)

    ssl_cfg["plot_last_epoch"] = False

    trainer = Trainer(0, **ssl_cfg)

    trainer.fit()


def test_third_party_model() -> None:
    cfg_path = Path(__file__).parent.parent / "inbuilt_cfgs" / "example_3rd_party.yml"

    with config_load.ToDefaultConfDir():
        cfg, _ = config_load.load_configs(cfg_path)

    trainer = Trainer(0, **cfg)

    trainer.fit()


def test_autoencoder() -> None:
    cfg_path = (
        Path(__file__).parent.parent / "inbuilt_cfgs" / "example_autoencoder_config.yml"
    )

    with config_load.ToDefaultConfDir():
        cfg, _ = config_load.load_configs(cfg_path)

    trainer = Trainer(0, **cfg)

    trainer.fit()
