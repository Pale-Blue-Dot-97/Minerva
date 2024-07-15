# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
    MASTER_PARSER (~argparse.ArgumentParser): Argparser for the CLI for the config loading.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "universal_path",
    "DEFAULT_CONF_DIR_PATH",
    "DEFAULT_CONFIG_NAME",
    "MASTER_PARSER",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse

from minerva.utils.config_load import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME
from minerva.utils.config_load import universal_path as universal_path  # noqa: F401

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
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
