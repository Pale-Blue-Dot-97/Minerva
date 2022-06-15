#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Distributed Computing version of ``MinervaExp``.

Designed for use in SLURM clusters.

Some code derived from Barlow Twins implementation of distributed computing:
https://github.com/facebookresearch/barlowtwins


TODO:
    * Add arg parsing from CLI
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import os
import signal
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minerva.trainer import Trainer
from minerva.utils import config, master_parser

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
torch.manual_seed(0)


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def run(gpu: int, args) -> None:
    # Calculates the global rank of this process.
    args.rank += gpu

    dist.init_process_group(
        backend="gloo",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    print(f"INITIALISED PROCESS ON {args.rank}")

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(gpu=gpu, rank=args.rank, world_size=args.world_size, **config)

    trainer.fit()

    if config["pre_train"] and gpu == 0:
        trainer.save_backbone()
        trainer.close()

    if not config["pre_train"]:
        trainer.test()


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def main(args):
    try:
        mp.spawn(run, (args,), args.ngpus_per_node)
    except KeyboardInterrupt:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[master_parser])
    args = parser.parse_args()

    args.ngpus_per_node = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        # Single-node and multi-node distributed training on SLURM cluster.
        # Requeue job on SLURM preemption.
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # Find a common host name on all nodes.
        # Assume scontrol returns hosts in the same order on all nodes.
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:58472"

    else:
        # Single-node distributed training.
        args.rank = 0
        args.dist_url = "tcp://localhost:58472"
        args.world_size = args.ngpus_per_node

    main(args)
