# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.

# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
#
"""Module containing UNet models."""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from abc import ABC
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
    Sequence,
)

import torch
import torch.nn.modules as nn
from torch import Tensor

from .core import MinervaModel, bilinear_init
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"

__all__ = []

# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
