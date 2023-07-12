# -*- coding: utf-8 -*-
# flake8: noqa: F401
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
""":mod:`models` contains several types of models designed to work within :mod:`minerva`."""
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
from .siamese import SimConv as SimConv
from .siamese import SimSiam18 as SimSiam18
from .siamese import SimSiam34 as SimSiam34
from .siamese import SimSiam50 as SimSiam50
from .unet import UNet as UNet
from .unet import UNetR18 as UNetR18
from .unet import UNetR34 as UNetR34
from .unet import UNetR50 as UNetR50
from .unet import UNetR101 as UNetR101
from .unet import UNetR152 as UNetR152
