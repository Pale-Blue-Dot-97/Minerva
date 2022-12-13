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
"""Handles the loading of config files and checking paths.
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml
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
# Default values for the path to the config directory and config name.
DEFAULT_CONF_DIR_PATH = Path("../../inbuilt_cfgs/")
DEFAULT_CONFIG_NAME: str = "example_config.yml"

# Objects to hold the config name and path.
CONFIG_NAME: Optional[str] = None
CONFIG_PATH: Optional[Path] = None

# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def check_paths(config, use_default_conf_dir: bool):
    if config is not None:
        p = Path(config)
        head = p.parent
        tail = p.name
        
        if str(head) != "" or str(head) is not None:
            CONFIG_PATH = head
        elif head == "" or head is None:
            CONFIG_PATH = Path("")
        
        CONFIG_NAME = tail

    # Overwrites the config path if option found in args regardless of -c args.
    if use_default_conf_dir:
        if CONFIG_PATH is not None:
            print(
                "Warning: Config path specified with `--default_config_dir` option."
                + "\nDefault config directory path will be used."
            )
        CONFIG_PATH = None


def chdir_to_default(config_name: Optional[str] = None) -> str:

    this_abs_path = (Path(__file__).parent / DEFAULT_CONF_DIR_PATH).resolve()
    os.chdir(this_abs_path)

    if config_name is None:
        return DEFAULT_CONFIG_NAME
    elif not Path(config_name).exists():
        return DEFAULT_CONFIG_NAME
    else:
        return config_name


def load_configs(master_config_path: str) -> Tuple[Dict[str, Any], ...]:
    """Loads the master config from YAML. Finds other config paths within and loads them.

    Args:
        master_config_path (str): Path to the master config YAML file.

    Returns:
        Master config and any other configs found from paths in the master config.
    """

    def yaml_load(path: str) -> Any:
        """Loads YAML file from path as dict.
        Args:
            path(str): Path to YAML file.

        Returns:
            yml_file (dict): YAML file loaded as dict.
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def aux_config_load(paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Loads and returns config files from YAML as dicts.

        Args:
            paths (dict): Dictionary mapping config names to paths to their YAML files.

        Returns:
            Config dictionaries loaded from YAML from paths.
        """
        configs = {}
        for _config_name in paths.keys():
            # Loads config from YAML as dict.
            configs[_config_name] = yaml_load(paths[_config_name])
        return configs

    # First loads the master config.
    master_config = yaml_load(master_config_path)

    # Gets the paths for the other configs from master config.
    config_paths = master_config["dir"]["configs"]

    # Loads and returns the other configs along with master config.
    return master_config, aux_config_load(config_paths)
