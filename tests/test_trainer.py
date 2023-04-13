# -*- coding: utf-8 -*-
import argparse
import shutil
from pathlib import Path

import pytest
import torch

from minerva.models import MinervaOnnxModel
from minerva.trainer import Trainer
from minerva.utils import config_load, runner
from minerva.utils.utils import CONFIG


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
        trainer1.save_model(format="unkown")

    trainer1.save_model(fn=trainer1.get_model_cache_path(), format="onnx")

    params2 = CONFIG.copy()
    params2["pre_train_name"] = f"{params1['model_name'].split('-')[0]}.onnx"
    params2["sample_pairs"] = "false"
    params2["plot_last_epoch"] = False
    params2["wandb_log"] = False
    params2["project"] = False
    params2["max_epochs"] = 2

    trainer2 = Trainer(0, **params2)
    assert isinstance(trainer2.model, MinervaOnnxModel)

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
