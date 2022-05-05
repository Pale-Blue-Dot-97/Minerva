"""Distributed Computing version of `MinervaExp`.

    Copyright (C) 2022 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

TODO:
    * Add arg parsing from CLI
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from minerva.utils import config
from minerva.trainer import Trainer
import os
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
torch.manual_seed(0)


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(gpu: int, args) -> None:
    # Calculates the global rank of this process.
    rank = args.nr * args.gpus + gpu

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=args.world_size, rank=rank
    )

    trainer = Trainer(gpu=gpu, rank=rank, world_size=args.world_size, **config)

    trainer.fit()

    if config["pre_train"] and gpu == 0:
        trainer.save_backbone()

    else:
        trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    os.environ["MASTER_ADDR"] = "10.57.23.164"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.gpus, args=(args,))
