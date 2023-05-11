# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Utility functionality, visualisation and configuration for :mod:`minerva`.

Attributes:
    CONFIG_NAME (str): Name of the config to be used in the experiment.
    CONFIG_PATH (str): Path to the config.
    MASTER_PARSER (~argparse.ArgumentParser): Argparser for the CLI for the config loading.
    CONFIG (dict[str, Any]): The master config loaded by :mod:`config_load`.
    AUX_CONFIGS (dict[str, Any]): Dictionary containing the auxilary configs loaded by :mod:`config_load`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "universal_path",
    "CONFIG_NAME",
    "CONFIG_PATH",
    "MASTER_PARSER",
    "CONFIG",
    "AUX_CONFIGS",
]

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
#                                                     GLOBALS
# =====================================================================================================================
# Objects to hold the config name and path.
CONFIG_NAME: Optional[str]
CONFIG_PATH: Optional[Path]

MASTER_PARSER = argparse.ArgumentParser(add_help=False)
MASTER_PARSER.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to the config file defining experiment",
)
MASTER_PARSER.add_argument(
    "--use-default-conf-dir",
    dest="use_default_conf_dir",
    action="store_true",
    help="Set config path to default",
)
_args, _ = MASTER_PARSER.parse_known_args()

# Store the current working directory (i.e where script is being run from).
_cwd = os.getcwd()

_path, CONFIG_NAME, CONFIG_PATH = check_paths(_args.config, _args.use_default_conf_dir)

# Loads the configs from file using paths found in sys.args.
CONFIG, AUX_CONFIGS = load_configs(_path)

# Change the working directory back to script location.
os.chdir(_cwd)
