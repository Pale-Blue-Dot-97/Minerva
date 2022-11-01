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
"""Script to execute the creation, fitting and testing of a computer vision neural network model.

Designed for use in SLURM clusters and with distributed computing support.

Some code derived from Barlow Twins implementation of distributed computing:
https://github.com/facebookresearch/barlowtwins
"""

# TODO: Add ability to conduct hyper-parameter iterative variation experimentation.
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minerva.trainer import Trainer
from minerva.utils import CONFIG, utils, runner

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
        dist.init_process_group(  # type: ignore[attr-defined]
            backend="gloo",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print(f"INITIALISED PROCESS ON {args.rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(gpu=gpu, rank=args.rank, world_size=args.world_size, **CONFIG)

    if not CONFIG["eval"]:
        trainer.fit()

    if CONFIG["pre_train"] and gpu == 0:
        trainer.save_backbone()
        trainer.close()

    if not CONFIG["pre_train"]:
        trainer.test()


def main(args):
    if args.world_size <= 1:
        run(gpu=0, args=args)

    else:
        try:
            mp.spawn(run, (args,), args.ngpus_per_node)  # type: ignore[attr-defined]
        except KeyboardInterrupt:
            dist.destroy_process_group()  # type: ignore[attr-defined]


if __name__ == "__main__":
    # ---+ CLI +--------------------------------------------------------------+
    parser = argparse.ArgumentParser(parents=[runner.generic_parser])
    args = parser.parse_args()

    args.ngpus_per_node = torch.cuda.device_count()

    # Convert CLI arguments to dict.
    args_dict = vars(args)

    # Find which CLI arguments are not in the config.
    new_args = {key: args_dict[key] for key in args_dict if key not in CONFIG}

    # Updates the config with new arguments from the CLI.
    CONFIG.update(new_args)

    # Get seed from config.
    seed = CONFIG.get("seed", 42)

    # Set torch, numpy and inbuilt seeds for reproducibility.
    utils.set_seeds(seed)

    args = runner.config_env_vars(args)

    main(args)
