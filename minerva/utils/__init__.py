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

from minerva.utils.config_load import check_paths, chdir_to_default, load_configs

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

config_name: Optional[str] = None
config_path: Optional[Path] = None

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

#check_paths(args.config, args.use_default_conf_dir)

# Set the config path from the option found from args.
if args.config is not None:
    p = Path(args.config)
    head = p.parent
    tail = p.name
    
    if str(head) != "" or str(head) is not None:
        config_path = head
    elif head == "" or head is None:
        config_path = Path("")
    
    config_name = tail

# Overwrites the config path if option found in args regardless of -c args.
if args.default_config_dir:
    if config_path is not None:
        print(
            "Warning: Config path specified with `--default_config_dir` option."
            + "\nDefault config directory path will be used."
        )
    config_path = None

# Store the current working directory (i.e where script is being run from).
cwd = os.getcwd()

# If no config_path, change directory to the default config directory.
if config_path is None:
    config_name = chdir_to_default(config_name)

# Check the config specified exists at the path given. If not, assume its in the default directory.
else:
    if config_name is None:
        config_name = chdir_to_default(config_name)
    elif not (config_path / config_name).exists():
        config_name = chdir_to_default(config_name)
    else:
        pass

path = config_name
if config_path is not None and config_path != Path(""):
    path = str(config_path / config_name)

# Loads the configs from file using paths found in sys.args.
CONFIG, AUX_CONFIGS = load_configs(path)

# Change the working directory back to script location.
os.chdir(cwd)
