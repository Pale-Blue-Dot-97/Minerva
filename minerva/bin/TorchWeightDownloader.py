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
"""Loads :mod:`torch` weights from Torch Hub into cache.

Attributes:
    resnets (List[str]): List of tags for ``pytorch`` resnet weights to download.
"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Optional

from torchvision.models._api import WeightsEnum

from minerva.models import get_torch_weights

resnets = [
    "ResNet101_Weights.IMAGENET1K_V1",
    "ResNet152_Weights.IMAGENET1K_V1",
    "ResNet18_Weights.IMAGENET1K_V1",
    "ResNet34_Weights.IMAGENET1K_V1",
    "ResNet50_Weights.IMAGENET1K_V1",
]


def main() -> None:
    for resnet in resnets:
        weights: Optional[WeightsEnum] = get_torch_weights(resnet)
        assert weights
        _ = weights.get_state_dict(True)


if __name__ == "__main__":
    main()
