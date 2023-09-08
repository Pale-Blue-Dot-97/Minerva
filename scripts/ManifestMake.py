#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
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
"""Script to create manifests of data for use in Minerva pre-processing to reduce computation time."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from minerva.datasets import make_manifest
from minerva.utils import CONFIG, universal_path, utils


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    manifest = make_manifest(CONFIG)

    print(manifest)

    output_dir = universal_path(CONFIG["dir"]["cache"])

    fn = output_dir / f"{utils.get_dataset_name()}_Manifest.csv"

    print(f"MANIFEST TO FILE -----> {fn}")
    manifest.to_csv(fn)


if __name__ == "__main__":
    main()
