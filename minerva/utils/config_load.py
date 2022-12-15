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
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Union

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

# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def check_paths(
    config: Optional[Union[str, PathLike[str]]], use_default_conf_dir: bool
) -> Tuple[str, Optional[str], Optional[Path]]:
    """Checks the path given for the config.

    Args:
        config (Optional[Union[str, PathLike[str]]]): Path to the config given from the CLI.
        use_default_conf_dir (bool): Assumes that ``config`` is in the default config directory if ``True``.

    Returns:
        Tuple[str, Optional[str], Optional[Path]]: Tuple of the path for :func:`load_configs` to use, the config name and path to config.
    """

    config_name: Optional[str] = None
    config_path: Optional[Path] = None

    if config is not None:
        p = Path(config)
        head = p.parent
        tail = p.name

        if str(head) != "" or str(head) is not None:
            config_path = head
        elif head == "" or head is None:
            config_path = Path("")

        config_name = tail

    # Overwrites the config path if option found in args regardless of -c args.
    if use_default_conf_dir:
        if config_path is not None:
            print(
                "Warning: Config path specified with `--default_config_dir` option."
                + "\nDefault config directory path will be used."
            )
        config_path = None

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

    return path, config_name, config_path


def chdir_to_default(config_name: Optional[str] = None) -> str:
    """Changes the current working directory to the default config directory.

    Args:
        config_name (Optional[str]): Optional; Name of the config in the default directory. Defaults to None.

    Returns:
        str: :var:`DEFAULT_CONFIG_NAME` if ``config_name`` not in default directory. ``config_name`` if it does exist.
    """

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
