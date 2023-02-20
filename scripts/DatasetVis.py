#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
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
import matplotlib.pyplot as plt

from minerva.trainer import Trainer
from minerva.transforms import Normalise
from minerva.utils import CONFIG, runner


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(args) -> None:
    trainer = Trainer(
        gpu=0,
        **CONFIG,
    )

    transform = Normalise(255)

    for batch in trainer.loaders["train"]:
        sample = {key: batch[key][0] for key in batch}
        image = transform(sample["image"][0:3, :, :].permute(1, 2, 0))
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        plt.imshow(image)
        plt.axis("off")
        plt.savefig("pic1.png")

        break
        # dataset.plot({key: batch[key][0][1] for key in batch})


if __name__ == "__main__":
    # ---+ CLI +--------------------------------------------------------------+
    parser = argparse.ArgumentParser(parents=[runner.GENERIC_PARSER], add_help=False)
    argcomplete.autocomplete(parser)
    # ------------ ADD EXTRA ARGS FOR THE PARSER HERE ------------------------+

    # Export args from CLI.
    cli_args = parser.parse_args()

    main(cli_args)
