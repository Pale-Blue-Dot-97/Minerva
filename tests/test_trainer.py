import argparse

import pytest
import torch

from minerva.models import MinervaOnnxModel
from minerva.trainer import Trainer
from minerva.utils import runner
from minerva.utils.utils import CONFIG, set_seeds

set_seeds(42)


def run_trainer(gpu: int, args: argparse.Namespace):
    args.gpu = gpu
    params = CONFIG.copy()
    params["calc_norm"] = True

    trainer = Trainer(gpu=args.gpu, **params)
    assert isinstance(trainer, Trainer)

    trainer.fit()

    trainer.test()

    if args.gpu == 0:
        trainer.save_model()

        trainer.save_backbone()


def test_trainer_1() -> None:
    args = argparse.Namespace()

    if torch.distributed.is_available():  # type: ignore
        # Assumes distributed tests are single node
        args.rank = 0
        args.dist_url = "tcp://localhost:58472"
        args.world_size = torch.cuda.device_count()
        args.ngpus_per_node = args.world_size
        args.distributed = True
        args.log_all = False
        args.entity = None
        args.project = "pytest"
        args.wandb_log = True
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

    trainer2 = Trainer(0, **params2)
    assert isinstance(trainer2.model, MinervaOnnxModel)

    trainer2.fit()
    trainer2.test()
