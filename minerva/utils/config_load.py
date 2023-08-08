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
"""Handles the loading of config files and checking paths.

Attributes:
    DEFAULT_CONF_DIR_PATH (~pathlib.Path): Path to the default config directory.
    DEFAULT_CONFIG_NAME (str): Name of the default, example config.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "DEFAULT_CONF_DIR_PATH",
    "DEFAULT_CONFIG_NAME",
    "ToDefaultConfDir",
    "universal_path",
    "check_paths",
    "chdir_to_default",
    "load_configs",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Default values for the path to the config directory and config name.
DEFAULT_CONF_DIR_PATH = Path("../inbuilt_cfgs/")
DEFAULT_CONFIG_NAME: str = "example_config.yml"


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class ToDefaultConfDir:
    """Changes to the default config directory. Switches back to the previous CWD on close."""

    def __init__(self) -> None:
        self._cwd = os.getcwd()
        self._def_dir = (Path(__file__).parent / DEFAULT_CONF_DIR_PATH).resolve()

    def __enter__(self) -> None:
        os.chdir(self._def_dir)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        os.chdir(self._cwd)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def universal_path(path: Any) -> Path:
    """Creates a :class:`~pathlib.Path` object from :class:`str` or :class:`~typing.Iterable` inputs.

    Args:
        path (~typing.Any): Representation of a path to convert to :class:`~pathlib.Path` object.

    Returns:
        ~pathlib.Path: :class:`~pathlib.Path` object of the input ``path``.
    """
    if isinstance(path, Path):
        return path
    elif isinstance(path, str):
        return Path(path)
    else:
        return Path(*path)


def check_paths(
    config: Optional[Union[str, Path]] = None, use_default_conf_dir: bool = False
) -> Tuple[str, Optional[str], Optional[Path]]:
    """Checks the path given for the config.

    Args:
        config (str | ~pathlib.Path | None): Path to the config given from the CLI.
        use_default_conf_dir (bool): Assumes that ``config`` is in the default config directory if ``True``.

    Returns:
        tuple[str, ~typing.Optional[str], ~typing.Optional[~pathlib.Path]]: Tuple of the path for
        :func:`load_configs` to use, the config name and path to config.
    """

    config_name: Optional[str] = None
    config_path: Optional[Path] = None

    if config is not None:
        p = Path(config)
        head = p.parent
        tail = p.name

        if str(head) != "" or str(head) is not None:
            config_path = head
        elif str(head) == "" or head is None:
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
        if config_name is None or not (config_path / config_name).exists():
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
        config_name (str): Optional; Name of the config in the default directory. Defaults to None.

    Returns:
        str: :data:`DEFAULT_CONFIG_NAME` if ``config_name`` not in default directory. ``config_name`` if it does exist.
    """

    this_abs_path = (Path(__file__).parent / DEFAULT_CONF_DIR_PATH).resolve()
    os.chdir(this_abs_path)

    if config_name is None or not Path(config_name).exists():
        return DEFAULT_CONFIG_NAME
    else:
        return config_name


def load_configs(master_config_path: Union[str, Path]) -> Tuple[Dict[str, Any], ...]:
    """Loads the master config from ``YAML``. Finds other config paths within and loads them.

    Args:
        master_config_path (str): Path to the master config ``YAML`` file.

    Returns:
        tuple[dict[str, ~typing.Any], ...]: Master config and any other configs found from paths in the master config.
    """

    def yaml_load(path: Union[str, Path]) -> Any:
        """Loads ``YAML`` file from path as dict.
        Args:
            path(str | ~pathlib.Path): Path to ``YAML`` file.

        Returns:
            yml_file (dict): YAML file loaded as dict.
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def aux_config_load(paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Loads and returns config files from YAML as dicts.

        Args:
            paths (dict[str, str]): Dictionary mapping config names to paths to their ``YAML`` files.

        Returns:
            dict[str, dict[str, ~typing.Any]]: Config dictionaries loaded from ``YAML`` from paths.
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
