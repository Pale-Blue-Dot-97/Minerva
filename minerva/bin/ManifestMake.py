#!/usr/bin/env python
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
"""Script to create manifests of data for use in Minerva pre-processing to reduce computation time.

Attributes:
    config_path (str): Path to master config YAML file.
    config (dict): Master config defining how the experiment should be conducted.

TODO:
    * Re-engineer for use with torchvision style datasets
    * Consider use of parquet format rather than csv
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os

from minerva.utils import CONFIG, utils
from minerva.datasets import make_manifest

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    manifest = make_manifest()

    print(manifest)

    output_dir = os.sep.join(CONFIG["dir"]["cache"])

    fn = os.sep.join([output_dir, f"{utils.get_dataset_name()}_Manifest.csv"])

    print(f"MANIFEST TO FILE -----> {fn}")
    manifest.to_csv(fn)


if __name__ == "__main__":
    main()
