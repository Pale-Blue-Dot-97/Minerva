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
"""Adaptation of ``MinervaExp.py`` for cluster visualisation of a model.

Designed for use in SLURM clusters and with distributed computing support.

Some code derived from Barlow Twins implementation of distributed computing:
https://github.com/facebookresearch/barlowtwins
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import os
import signal
import subprocess
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minerva.trainer import Trainer
from minerva.utils import config, master_parser, utils

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def run(gpu: int, args) -> None:
    # Calculates the global rank of this process.
    args.rank += gpu

    if args.world_size > 1:
        dist.init_process_group(
            backend="gloo",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print(f"INITIALISED PROCESS ON {args.rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(gpu=gpu, rank=args.rank, world_size=args.world_size, **config)

    trainer.tsne_cluster()

    if gpu == 0:
        trainer.close()


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def main(args):
    if args.world_size <= 1:
        run(gpu=0, args=args)

    else:
        try:
            mp.spawn(run, (args,), args.ngpus_per_node)
        except KeyboardInterrupt:
            dist.destroy_process_group()


if __name__ == "__main__":
    # ---+ CLI +--------------------------------------------------------------+
    parser = argparse.ArgumentParser(parents=[master_parser])

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed number",
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        help="Name of model."
        + " Sub-string before hyphen is taken as model class name."
        + " Sub-string past hyphen can be used to differeniate between versions.",
    )

    parser.add_argument(
        "--model-type",
        dest="model_type",
        type=str,
        help="Type of model. Should be 'segmentation', 'scene_classifier', 'siamese' or 'mlp'",
    )

    parser.add_argument(
        "--pre-train",
        action="store_false",
        help="Sets experiment type to pre-train. Will save model to cache at end of training.",
    )

    parser.add_argument(
        "--fine-tune",
        action="store_false",
        help="Sets experiment type to fine-tune. Will load pre-trained backbone from file.",
    )

    parser.add_argument(
        "--balance",
        action="store_false",
        help="Activates class balancing."
        + " Depending on `model_type`, this will either be via sampling or weighting of the loss function.",
    )

    parser.add_argument(
        "--class-elim",
        dest="elim",
        action="store_false",
        help="Eliminates classes that are specified in config but not present in the data.",
    )

    parser.add_argument(
        "--sample-pairs",
        dest="sample_pairs",
        action="store_false",
        help="Used paired sampling. E.g. For Siamese models.",
    )

    parser.add_argument(
        "--save-model",
        dest="save_model",
        type=str,
        default=False,
        help="Whether to save the model at end of testing. Must be 'true', 'false' or 'auto'."
        + " Setting 'auto' will automatically save the model to file."
        + " 'true' will ask the user whether to or not at runtime."
        + " 'false' will not save the model and will not ask the user at runtime.",
    )

    parser.add_argument(
        "--run-tensorboard",
        dest="run_tensorboard",
        type=str,
        default=False,
        help="Whether to run the Tensorboard logs at end of testing. Must be 'true', 'false' or 'auto'."
        + " Setting 'auto' will automatically locate and run the logs on a local browser."
        + " 'true' will ask the user whether to or not at runtime."
        + " 'false' will not save the model and will not ask the user at runtime.",
    )

    parser.add_argument(
        "--save-plots",
        dest="save",
        action="store_false",
        help="Whether to save plots created to file or not.",
    )

    parser.add_argument(
        "--show-plots",
        dest="show",
        action="store_false",
        help="Whether to show plots created in a window or not."
        + " Warning: Do not use with a terminal-less operation, e.g. SLURM.",
    )

    parser.add_argument(
        "--print-dist",
        dest="p_dist",
        action="store_false",
        help="Whether to print the distribution of classes within the data to `stdout`.",
    )

    parser.add_argument(
        "--plot-last-epoch",
        dest="plot_last_epoch",
        action="store_false",
        help="Whether to plot the results from the final validation epoch.",
    )

    args = parser.parse_args()

    args.ngpus_per_node = torch.cuda.device_count()

    # Convert CLI arguments to dict.
    args_dict = vars(args)

    # Find which CLI arguments are not in the config.
    new_args = {key: args_dict[key] for key in args_dict if key not in config}

    # Updates the config with new arguments from the CLI.
    config.update(new_args)

    # Get seed from config.
    seed = config.get("seed", 42)

    # Set torch, numpy and inbuilt seeds for reproducibility.
    utils.set_seeds(seed)

    if "SLURM_JOB_ID" in os.environ:
        # Single-node and multi-node distributed training on SLURM cluster.
        # Requeue job on SLURM preemption.
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

        # Get SLURM variables.
        slurm_job_nodelist: Optional[str] = os.getenv("SLURM_JOB_NODELIST")
        slurm_nodeid: Optional[str] = os.getenv("SLURM_NODEID")
        slurm_nnodes: Optional[str] = os.getenv("SLURM_NNODES")

        # Check that SLURM variables have been found.
        assert slurm_job_nodelist is not None
        assert slurm_nodeid is not None
        assert slurm_nnodes is not None

        # Find a common host name on all nodes.
        # Assume scontrol returns hosts in the same order on all nodes.
        cmd = "scontrol show hostnames " + slurm_job_nodelist
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(slurm_nodeid) * args.ngpus_per_node
        args.world_size = int(slurm_nnodes) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:58472"

    else:
        # Single-node distributed training.
        args.rank = 0
        args.dist_url = "tcp://localhost:58472"
        args.world_size = args.ngpus_per_node

    main(args)
