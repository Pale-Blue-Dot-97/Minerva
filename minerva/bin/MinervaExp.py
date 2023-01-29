#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
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
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse

import argcomplete
import wandb

from minerva.trainer import Trainer
from minerva.utils import CONFIG, runner


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(gpu: int, args) -> None:
    trainer = Trainer(gpu=gpu, rank=args.rank, world_size=args.world_size, **CONFIG)

    if not CONFIG["eval"]:
        trainer.fit()

    if CONFIG["pre_train"] and gpu == 0:
        trainer.save_backbone()
        trainer.close()

    if not CONFIG["pre_train"]:
        trainer.test()


if __name__ == "__main__":
    # ---+ CLI +--------------------------------------------------------------+
    parser = argparse.ArgumentParser(parents=[runner.GENERIC_PARSER], add_help=False)
    argcomplete.autocomplete(parser)
    # ------------ ADD EXTRA ARGS FOR THE PARSER HERE ------------------------+

    # Export args from CLI.
    cli_args = parser.parse_args()

    # Initialise Weights and Biases.
    wandb.init(project=cli_args.project_name)

    # Configure the arguments and environment variables.
    runner.config_args(cli_args)

    # Run the specified main with distributed computing and the arguments provided.
    runner.distributed_run(main, cli_args)
