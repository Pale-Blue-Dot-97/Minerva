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
"""Utility functionality, visualisation and configuration for :mod:`minerva`."""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import os
from pathlib import Path
from typing import Optional

from minerva.utils.config_load import check_paths, load_configs
from minerva.utils.config_load import universal_path as universal_path  # noqa: F401

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
# Objects to hold the config name and path.
CONFIG_NAME: Optional[str] = None
CONFIG_PATH: Optional[Path] = None

master_parser = argparse.ArgumentParser(add_help=False)
master_parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to the config file defining experiment",
)
master_parser.add_argument(
    "--use-default-conf-dir",
    dest="use_default_conf_dir",
    action="store_true",
    help="Set config path to default",
)
args, _ = master_parser.parse_known_args()

# Store the current working directory (i.e where script is being run from).
cwd = os.getcwd()

path, CONFIG_NAME, CONFIG_PATH = check_paths(args.config, args.use_default_conf_dir)

# Loads the configs from file using paths found in sys.args.
CONFIG, AUX_CONFIGS = load_configs(path)

# Change the working directory back to script location.
os.chdir(cwd)
