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
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "DEFAULT_CONF_DIR_PATH",
    "DEFAULT_CONFIG_NAME",
    "ToDefaultConfDir",
    "universal_path",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from pathlib import Path
from typing import Any

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Default values for the path to the config directory and config name.
DEFAULT_CONF_DIR_PATH = Path(__file__).parent / ".." / "inbuilt_cfgs"
DEFAULT_CONFIG_NAME: str = "example_config.yaml"


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
