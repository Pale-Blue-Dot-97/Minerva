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
"""Functionality for constructing datasets, samplers and :class:`~torch.utils.data.DataLoader` for :mod:`minerva`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["SSL4EOS12Sentinel2"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from torchgeo.datasets import Sentinel2


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class SSL4EOS12Sentinel2(Sentinel2):
    """Adapted version of :class:~`torchgeo.datasets.Sentinel2` that works .

    Attributes:
        filename_glob (str): Pattern for tiff files within dataset root to construct dataset from.
    """

    filename_glob = "{}.*"
    filename_regex = r"""(?P<band>B[^[0-1]?[0-9]|B[^[1]?[0-9][\dA])\..*$"""
    date_format = ""
    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    ]
    rgb_bands = ["B4", "B3", "B2"]
