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

from minerva.trainer import Trainer
from minerva.utils import runner, CONFIG

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
def main(gpu: int, args) -> None:

    trainer = Trainer(gpu=gpu, rank=args.rank, world_size=args.world_size, **CONFIG)

    trainer.tsne_cluster()

    if gpu == 0:
        trainer.close()


if __name__ == "__main__":
    # ---+ CLI +--------------------------------------------------------------+
    parser = argparse.ArgumentParser(parents=[runner.generic_parser])

    # ------------ ADD EXTRA ARGS FOR THE PARSER HERE ------------------------+

    # Export args from CLI.
    args = parser.parse_args()

    # Configure the arguments and environment variables.
    runner.config_args(args)

    # Run the specified main with distributed computing and the arguments provided.
    runner.distributed_run(main, args)
