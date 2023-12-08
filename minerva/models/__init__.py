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
__all__ = [
    "MinervaBackbone",
    "MinervaDataParallel",
    "MinervaModel",
    "MinervaOnnxModel",
    "MinervaWrapper",
    "bilinear_init",
    "get_output_shape",
    "get_torch_weights",
    "is_minerva_model",
    "is_minerva_subtype",
    "extract_wrapped_model",
    "wrap_model",
    "FCN8ResNet18",
    "FCN8ResNet34",
    "FCN8ResNet50",
    "FCN8ResNet101",
    "FCN8ResNet152",
    "FCN16ResNet18",
    "FCN16ResNet34",
    "FCN16ResNet50",
    "FCN32ResNet18",
    "FCN32ResNet34",
    "FCN32ResNet50",
    "PSPEncoder",
    "ResNetX",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "MinervaSiamese",
    "SimCLR18",
    "SimCLR34",
    "SimCLR50",
    "SimConv",
    "SimSiam18",
    "SimSiam34",
    "SimSiam50",
    "UNet",
    "UNetR18",
    "UNetR34",
    "UNetR50",
    "UNetR101",
    "UNetR152",
]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from .__depreciated import CNN as CNN
from .__depreciated import MLP as MLP
from .core import (
    MinervaBackbone,
    MinervaDataParallel,
    MinervaModel,
    MinervaOnnxModel,
    MinervaWrapper,
    bilinear_init,
    extract_wrapped_model,
    get_output_shape,
    get_torch_weights,
    is_minerva_model,
    is_minerva_subtype,
    wrap_model,
)
from .fcn import (
    FCN8ResNet18,
    FCN8ResNet34,
    FCN8ResNet50,
    FCN8ResNet101,
    FCN8ResNet152,
    FCN16ResNet18,
    FCN16ResNet34,
    FCN16ResNet50,
    FCN32ResNet18,
    FCN32ResNet34,
    FCN32ResNet50,
)
from .psp import PSPEncoder
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNetX
from .siamese import (
    MinervaSiamese,
    SimCLR18,
    SimCLR34,
    SimCLR50,
    SimConv,
    SimSiam18,
    SimSiam34,
    SimSiam50,
)
from .unet import UNet, UNetR18, UNetR34, UNetR50, UNetR101, UNetR152
