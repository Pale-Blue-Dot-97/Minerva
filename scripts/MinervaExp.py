#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
# Copyright (C) 2024 Harry Baker
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

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Optional

import hydra
from omegaconf import DictConfig
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from minerva.utils import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME, runner, utils


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
@hydra.main(
    version_base="1.3",
    config_path=str(DEFAULT_CONF_DIR_PATH),
    config_name=DEFAULT_CONFIG_NAME,
)
@runner.distributed_run
def main(gpu: int, wandb_run: Optional[Run | RunDisabled], cfg: DictConfig) -> None:

    # Due to the nature of multiprocessing and its interaction with hydra, wandb and SLURM,
    # the actual code excuted in the job is contained in `run_trainer` in `runner`.
    #
    # Any code placed here will not be executed with multiprocessing!

    pass


if __name__ == "__main__":
    # Print Minerva banner.
    utils._print_banner()

    with runner.WandbConnectionManager():
        # Run the specified main with distributed computing and the arguments provided.
        main()
