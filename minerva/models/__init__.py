# -*- coding: utf-8 -*-
# flake8: noqa: F401
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
""":mod:`models` contains several types of models designed to work within :mod:`minerva`."""
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
from .__depreciated import CNN as CNN
from .__depreciated import MLP as MLP
from .core import MinervaBackbone as MinervaBackbone
from .core import MinervaDataParallel as MinervaDataParallel
from .core import MinervaModel as MinervaModel
from .core import MinervaOnnxModel as MinervaOnnxModel
from .core import MinervaWrapper as MinervaWrapper
from .core import bilinear_init as bilinear_init
from .core import get_output_shape as get_output_shape
from .core import get_torch_weights as get_torch_weights
from .fcn import FCN8ResNet18 as FCN8ResNet18
from .fcn import FCN8ResNet34 as FCN8ResNet34
from .fcn import FCN8ResNet50 as FCN8ResNet50
from .fcn import FCN8ResNet101 as FCN8ResNet101
from .fcn import FCN8ResNet152 as FCN8ResNet152
from .fcn import FCN16ResNet18 as FCN16ResNet18
from .fcn import FCN16ResNet34 as FCN16ResNet34
from .fcn import FCN16ResNet50 as FCN16ResNet50
from .fcn import FCN32ResNet18 as FCN32ResNet18
from .fcn import FCN32ResNet34 as FCN32ResNet34
from .fcn import FCN32ResNet50 as FCN32ResNet50
from .resnet import ResNet18 as ResNet18
from .resnet import ResNet34 as ResNet34
from .resnet import ResNet50 as ResNet50
from .resnet import ResNet101 as ResNet101
from .resnet import ResNet152 as ResNet152
from .siamese import MinervaSiamese as MinervaSiamese
from .siamese import SimCLR18 as SimCLR18
from .siamese import SimCLR34 as SimCLR34
from .siamese import SimCLR50 as SimCLR50
from .siamese import SimSiam18 as SimSiam18
from .siamese import SimSiam34 as SimSiam34
from .siamese import SimSiam50 as SimSiam50
from .unet import UNet as UNet
from .unet import UNetR18 as UNetR18
from .unet import UNetR34 as UNetR34
from .unet import UNetR50 as UNetR50
from .unet import UNetR101 as UNetR101
from .unet import UNetR152 as UNetR152
