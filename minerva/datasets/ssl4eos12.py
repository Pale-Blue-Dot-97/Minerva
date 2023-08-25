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
r"""Simple adaption of the :mod:`~torchgeo.datasets.Sentinel2` dataset for use with the SSL4EO-S12 dataset.
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
    """Adapted version of :class:~`torchgeo.datasets.Sentinel2` that works with the SSL4EO-S12 data format.

    Attributes:
        filename_glob (str): Adapted pattern from :class:`~torchgeo.datasets.Sentinel2` that looks just for band ID.
        filename_regex (str): Adapted regex from :class:`~torchgeo.datasets.Sentinel2` that looks just for band IDs
            with either ``B0x`` (like standard Sentinel2) or ``Bx`` (like SSL4EO-S12) format.
        all_bands (list[str]): Sentinel2 bands with the leading 0 ommitted.
        rgb_bands (list[str]): RGB Sentinel2 bands with the leading 0 omitted.
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
