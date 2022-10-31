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
"""Module to handle generic functionality for running ``minerva`` scripts.
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
# ---+ Inbuilt +-------------------------------------------------------------------------------------------------------
import argparse

# ---+ Minerva +-------------------------------------------------------------------------------------------------------
from minerva.utils import master_parser

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
# ---+ CLI +--------------------------------------------------------------+
generic_parser = argparse.ArgumentParser(parents=[master_parser])

generic_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Set seed number",
)

generic_parser.add_argument(
    "--model-name",
    dest="model_name",
    type=str,
    help="Name of model."
    + " Sub-string before hyphen is taken as model class name."
    + " Sub-string past hyphen can be used to differeniate between versions.",
)

generic_parser.add_argument(
    "--model-type",
    dest="model_type",
    type=str,
    help="Type of model. Should be 'segmentation', 'scene_classifier', 'siamese' or 'mlp'",
)

generic_parser.add_argument(
    "--pre-train",
    action="store_false",
    help="Sets experiment type to pre-train. Will save model to cache at end of training.",
)

generic_parser.add_argument(
    "--fine-tune",
    action="store_false",
    help="Sets experiment type to fine-tune. Will load pre-trained backbone from file.",
)

generic_parser.add_argument(
    "--eval",
    action="store_false",
    help="Sets experiment type to pre-train. Will save model to cache at end of training.",
)

generic_parser.add_argument(
    "--balance",
    action="store_false",
    help="Activates class balancing."
    + " Depending on `model_type`, this will either be via sampling or weighting of the loss function.",
)

generic_parser.add_argument(
    "--class-elim",
    dest="elim",
    action="store_false",
    help="Eliminates classes that are specified in config but not present in the data.",
)

generic_parser.add_argument(
    "--sample-pairs",
    dest="sample_pairs",
    action="store_false",
    help="Used paired sampling. E.g. For Siamese models.",
)

generic_parser.add_argument(
    "--save-model",
    dest="save_model",
    type=str,
    default=False,
    help="Whether to save the model at end of testing. Must be 'true', 'false' or 'auto'."
    + " Setting 'auto' will automatically save the model to file."
    + " 'true' will ask the user whether to or not at runtime."
    + " 'false' will not save the model and will not ask the user at runtime.",
)

generic_parser.add_argument(
    "--run-tensorboard",
    dest="run_tensorboard",
    type=str,
    default=False,
    help="Whether to run the Tensorboard logs at end of testing. Must be 'true', 'false' or 'auto'."
    + " Setting 'auto' will automatically locate and run the logs on a local browser."
    + " 'true' will ask the user whether to or not at runtime."
    + " 'false' will not save the model and will not ask the user at runtime.",
)

generic_parser.add_argument(
    "--save-plots",
    dest="save",
    action="store_false",
    help="Whether to save plots created to file or not.",
)

generic_parser.add_argument(
    "--show-plots",
    dest="show",
    action="store_false",
    help="Whether to show plots created in a window or not."
    + " Warning: Do not use with a terminal-less operation, e.g. SLURM.",
)

generic_parser.add_argument(
    "--print-dist",
    dest="p_dist",
    action="store_false",
    help="Whether to print the distribution of classes within the data to `stdout`.",
)

generic_parser.add_argument(
    "--plot-last-epoch",
    dest="plot_last_epoch",
    action="store_false",
    help="Whether to plot the results from the final validation epoch.",
)
