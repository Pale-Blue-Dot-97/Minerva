import argparse
import shutil

import pytest
import torch

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
        trainer.save_model(format="onnx")

        with pytest.raises(ValueError):
            trainer.save_model(format="unkown")

        trainer.save_backbone()

    assert trainer.exp_fn.parent.exists()
    shutil.rmtree(trainer.exp_fn.parent)


def test_trainer() -> None:
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
