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
"""Module to store bespoke functions for converting between differently structured :mod:`torch` ``state_dict``."""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any

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


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def replace_keys(weights: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    for key in weights.keys():
        if old in key:
            weights[key.replace(old, new)] = weights.pop(key)

    return weights


def bt_hsic_to_minerva(weights: dict[str, Any]) -> dict[str, Any]:
    weights = replace_keys(weights, "f", "backbone.network")

    return weights
