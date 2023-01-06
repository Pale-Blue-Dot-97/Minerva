import argparse
import torch
from minerva.trainer import Trainer
from minerva.utils.utils import CONFIG, set_seeds
from minerva.utils import runner

set_seeds(42)


def test_trainer() -> None:
    args = argparse.Namespace()

    if torch.distributed.is_available():  # type: ignore
        # Assumes distributed tests are single node
        args.rank = 0
        args.dist_url = "tcp://localhost:58472"
        args.world_size = torch.cuda.device_count()
        args.ngpus_per_node = args.world_size
        args.distributed = True
        runner.distributed_run(run_trainer, args)

    else:
        args.gpu = 0
        run_trainer(args.gpu, args)


def run_trainer(gpu: int, args: argparse.Namespace):
    args.gpu = gpu
    params = CONFIG.copy()

    trainer = Trainer(gpu=args.gpu, **params)
    assert isinstance(trainer, Trainer)

    trainer.fit()

    trainer.test()

    if args.gpu == 0:
        trainer.save_backbone()
