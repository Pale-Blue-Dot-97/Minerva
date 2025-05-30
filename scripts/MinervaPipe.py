#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Harry Baker
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
"""Script to handle the pre-training of model and its subsequent downstream task fine-tuning."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import shlex
import subprocess
import sys
from typing import Any

import yaml


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(config_path: str):
    with open(config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    for key in config.keys():
        print(
            f"\nExecuting {key} experiment + ====================================================================="
        )

        try:
            exit_code = subprocess.Popen(  # nosec B607, B602
                shlex.split(f"python MinervaExp.py -c {config[key]}"),
                shell=True,
            ).wait()

            if exit_code != 0:
                raise SystemExit()
        except KeyboardInterrupt as err:
            print(f"{err}: Skipping to next experiment...")

        except SystemExit as err:
            print(err)
            print(f"Error in {key} experiment -> ABORT")
            sys.exit(exit_code)  # type: ignore

        print(
            f"\n{key} experiment COMPLETE + ====================================================================="
        )

    print("\nPipeline COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    main(config_path=args.config_path)
