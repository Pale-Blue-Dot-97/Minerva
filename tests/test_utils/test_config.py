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
r"""Tests for :mod:`minerva.utils.config_load`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
import os
from pathlib import Path

from minerva.utils.config_load import (
    DEFAULT_CONFIG_NAME,
    chdir_to_default,
    check_paths,
    load_configs,
    universal_path,
)


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_universal_path():
    path1 = "one/two/three/file.txt"
    path2 = ["one", "two", "three", "file.txt"]

    correct = Path(path1)

    assert universal_path(path1) == correct
    assert universal_path(path2) == correct


def test_config_path(config_root, config_here):
    assert str(Path("tmp/config")) in str(config_root)

    # Still works because we are relative to inbuilt_cfgs here
    base, aux = load_configs(config_root / "exp_mf_config.yml")
    assert base
    assert aux

    base, aux = load_configs(config_here / "exp_mf_config.yml")
    assert base
    assert aux


def test_check_paths(config_root: Path):
    path, config_name, config_path = check_paths(
        config_root, use_default_conf_dir=False
    )

    assert path == str(config_root)
    assert str(config_name) == config_root.name
    assert config_path == config_root.parent

    # Store the current working directory (i.e where script is being run from).
    cwd = os.getcwd()

    exp_config = "example_GeoCLR_config.yml"
    path2, config_name2, config_path2 = check_paths(
        exp_config, use_default_conf_dir=True
    )

    assert path2 == exp_config
    assert config_name2 == exp_config
    assert config_path2 is None

    nonexist_config_path = config_root / "non_existant_config.yml"
    assert not nonexist_config_path.exists()

    path3, config_name3, config_path3 = check_paths(nonexist_config_path)

    assert path3 == str(config_root / "example_config.yml")
    assert str(config_name3) == "example_config.yml"
    assert config_path3 == config_root

    # Change the working directory back to script location.
    os.chdir(cwd)


def test_chdir_to_default(inbuilt_cfg_root):
    def run_chdir(input, output):
        assert output == chdir_to_default(input)
        assert Path(os.getcwd()) == inbuilt_cfg_root
        os.chdir(cwd)

    cwd = os.getcwd()

    config_name1 = "example_GeoCLR_config.yml"

    run_chdir(config_name1, config_name1)
    run_chdir(None, DEFAULT_CONFIG_NAME)

    config_name2 = "wrong_config.yml"

    run_chdir(config_name2, DEFAULT_CONFIG_NAME)
