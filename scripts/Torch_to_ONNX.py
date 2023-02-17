#!/usr/bin/env python
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
"""Converts :mod:`torch` model weights to ``ONNX`` format."""

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

from minerva.trainer import Trainer
from minerva.utils import CONFIG, runner, universal_path


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(gpu: int, args) -> None:

    trainer = Trainer(
        gpu=gpu, rank=args.rank, world_size=args.world_size, verbose=False, **CONFIG
    )

    weights_path = universal_path(CONFIG["dir"]["cache"]) / CONFIG["pre_train_name"]

    trainer.save_model(fn=weights_path, format="onnx")

    print(f"Model saved to --> {weights_path}.onnx")

    if gpu == 0:
        trainer.close()


if __name__ == "__main__":
    # ---+ CLI +--------------------------------------------------------------+
    parser = argparse.ArgumentParser(parents=[runner.GENERIC_PARSER], add_help=False)

    # ------------ ADD EXTRA ARGS FOR THE PARSER HERE ------------------------+

    # Export args from CLI.
    cli_args = parser.parse_args()

    # Configure the arguments and environment variables.
    runner.config_args(cli_args)

    # Run the specified main with distributed computing and the arguments provided.
    runner.distributed_run(main, cli_args)
